"""Memory Lab — виды памяти ИИ на реальных Claude + mem0.
Запуск: uvicorn main:app --reload   →   http://127.0.0.1:8000

Работает только в реальном режиме (нужны ANTHROPIC_API_KEY в .env и пакеты из requirements.txt).
Если Claude не ответил, чат показывает ошибку, а не заглушку. Диагностика: GET /api/health

Память:
  • долговременные факты — mem0 + Qdrant (папка ./qdrant_data);
  • история диалогов, конспекты, сущности, эпизоды и факты — файл memory_lab_state.json,
    поэтому после перезапуска сервера чат и карточки не пустеют.
Ответ ИИ приходит потоком: POST /api/chat/stream (Server-Sent Events).
"""
import asyncio
import atexit
import json
import math
import os
import re
import threading
import time
import traceback
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()

# ---------- режим ----------
REAL = False
REAL_ERROR = ""
if os.getenv("USE_REAL_MEMORY", "true").lower() == "true":
    if not os.getenv("ANTHROPIC_API_KEY"):
        REAL_ERROR = "USE_REAL_MEMORY=true, но ANTHROPIC_API_KEY не задан в .env"
    else:
        try:
            import real
            REAL = True
        except Exception as e:  # не установлены пакеты и т.п.
            REAL_ERROR = f"Не удалось импортировать real.py: {type(e).__name__}: {e}"
else:
    REAL_ERROR = "USE_REAL_MEMORY выключен в .env"
if REAL_ERROR:
    print(f"[Memory Lab] Реальный режим недоступен: {REAL_ERROR}")
print(f"[Memory Lab] Режим: {'REAL (Claude + mem0)' if REAL else 'НЕ ЗАПУЩЕН — ' + REAL_ERROR}")


def safe_err(e: BaseException) -> str:
    """Текст ошибки без ключей API и с ограничением длины."""
    text = f"{type(e).__name__}: {e}"
    text = re.sub(r"(sk-[A-Za-z0-9_\-]{6,})", "sk-***", text)
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        key = os.getenv(var)
        if key:
            text = text.replace(key, "***")
    return text[:400]


@asynccontextmanager
async def lifespan(_app):
    try:
        yield
    except asyncio.CancelledError:
        pass  # Ctrl+C: останавливаем тихо, без длинного traceback в консоли
    finally:
        if REAL:
            real.close()  # закрыть Qdrant/SQLite до остановки интерпретатора


app = FastAPI(title="Memory Lab", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
BASE = Path(__file__).parent
if REAL:
    atexit.register(real.close)

WINDOW_K = (real.WINDOW_EXCHANGES * 2) if REAL else 4   # размер окна — в сообщениях, как в real.py

# --- лимиты, чтобы память сервера нельзя было раздуть запросами ---
MAX_MESSAGE = 2000        # символов в одном сообщении
MAX_MSGS_PER_THREAD = 200
MAX_THREADS_PER_USER = 30
MAX_USERS = 100
MAX_EPISODES = 300
MAX_FACTS = 100
RATE_LIMIT = 20           # запросов к /api/chat* за окно
RATE_WINDOW = 60          # секунд, с одного IP

ID_PATTERN = r"^[\w\-.]{1,40}$"
STATE_FILE = Path(os.getenv("STATE_FILE") or BASE / "memory_lab_state.json")

# short-term: живёт внутри (user_id, thread_id) — как checkpointer в LangGraph
threads = defaultdict(lambda: {"messages": [], "summary": "", "entities": {}})
# long-term: живёт по user_id между тредами (как mem0)
users = defaultdict(lambda: {"facts": [], "log": [], "episodes": []})
lock = threading.RLock()


# ---------- мини-эмбеддинги для эпизодической памяти ----------
def tokens(text: str):
    return [w[:5] for w in re.findall(r"[a-zа-яё0-9]+", text.lower()) if len(w) > 2]


def embed(text: str, dim: int = 128):
    v = [0.0] * dim
    for t in tokens(text):
        v[hash(t) % dim] += 1.0
    return v


def cosine(a, b):
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def n_tokens(text: str) -> int:
    return len(text.split())


# ---------- сохранение на диск ----------
def save_state():
    """Атомарно пишет всё состояние в JSON. Вызывать под lock."""
    try:
        data = {
            "threads": [{"user": u, "thread": t, "messages": th["messages"],
                         "summary": th["summary"], "entities": th["entities"]}
                        for (u, t), th in threads.items()],
            "users": {u: {"facts": [{"key": f["key"], "text": f["text"]} for f in us["facts"]],
                          "log": us["log"],
                          "episodes": [{k: e[k] for k in ("id", "ts", "thread", "user", "ai")}
                                       for e in us["episodes"]]}
                      for u, us in users.items()},
        }
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, STATE_FILE)
    except Exception as e:  # сохранение — best-effort, чат из-за него не падает
        print(f"[Memory Lab] не удалось сохранить состояние: {safe_err(e)}")


def load_state():
    if not STATE_FILE.exists():
        return
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        items = data.get("threads", [])
        for it in items:
            threads[(it["user"], it["thread"])] = {
                "messages": it.get("messages", []), "summary": it.get("summary", ""),
                "entities": it.get("entities", {})}
        for u, us in data.get("users", {}).items():
            users[u] = {
                "facts": [{**f, "vec": embed(f["text"])} for f in us.get("facts", [])],
                "log": us.get("log", []),
                "episodes": [{**e, "vec": embed(e["user"] + " " + e["ai"])} for e in us.get("episodes", [])]}
        if REAL:
            real.load_threads(items)
        print(f"[Memory Lab] Восстановлено с диска: тредов {len(items)}, пользователей {len(users)}")
    except Exception as e:
        print(f"[Memory Lab] Файл состояния повреждён, начинаем с нуля: {safe_err(e)}")
        try:
            STATE_FILE.replace(STATE_FILE.with_suffix(".bad"))
        except OSError:
            pass


load_state()

# ---------- защита: заголовки, лимиты, единый формат ошибок ----------
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
    ),
}

_hits = defaultdict(deque)


def rate_limited(ip: str) -> bool:
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > RATE_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        return True
    q.append(now)
    if len(_hits) > 5000:  # не даём словарю расти бесконечно
        for k in [k for k, v in _hits.items() if not v or now - v[-1] > RATE_WINDOW]:
            _hits.pop(k, None)
    return False


@app.middleware("http")
async def guard(request: Request, call_next):
    if request.url.path.startswith("/api/chat") and rate_limited(request.client.host if request.client else "?"):
        resp = JSONResponse({"detail": f"Слишком много сообщений. Подождите немного (лимит {RATE_LIMIT} в минуту)."},
                            status_code=429, headers={"Retry-After": str(RATE_WINDOW)})
    else:
        try:
            resp = await call_next(request)
        except Exception as e:  # любая необработанная ошибка → JSON, а не «Internal Server Error»
            traceback.print_exc()
            resp = JSONResponse({"detail": f"Ошибка сервера: {safe_err(e)}"}, status_code=500)
    for k, v in SECURITY_HEADERS.items():
        resp.headers.setdefault(k, v)
    resp.headers["Cache-Control"] = "no-store"
    return resp


# ---------- API ----------
class Chat(BaseModel):
    user_id: str = Field("anna", pattern=ID_PATTERN)
    thread_id: str = Field("thread-1", pattern=ID_PATTERN)
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE)


class Ref(BaseModel):
    user_id: str = Field("anna", pattern=ID_PATTERN)
    thread_id: str = Field("thread-1", pattern=ID_PATTERN)


class ApiError(Exception):
    def __init__(self, status: int, detail: str):
        self.status, self.detail = status, detail


_EMPTY_THREAD = {"messages": [], "summary": "", "entities": {}}
_EMPTY_USER = {"facts": [], "log": [], "episodes": []}


def user_threads(user_id: str):
    return [t for (u, t) in threads.keys() if u == user_id]


def threads_info(user_id: str):
    out = []
    for (u, t), th in threads.items():
        if u == user_id:
            first = next((m["text"] for m in th["messages"] if m["role"] == "human"), "")
            out.append({"id": t, "count": len(th["messages"]), "title": first[:70]})
    return out


def users_info():
    ids = sorted(set(users.keys()) | {u for (u, _t) in threads.keys()})
    out = []
    for u in ids:
        ts = [t for (uu, t) in threads.keys() if uu == u]
        out.append({"id": u, "threads": len(ts), "first": ts[0] if ts else "thread-1"})
    return out


def build_state(user_id: str, thread_id: str, hits=None, ep_hits=None):
    """Только чтение: не создаёт пустые записи для произвольных id."""
    th = threads.get((user_id, thread_id), _EMPTY_THREAD)
    us = users.get(user_id, _EMPTY_USER)
    msgs = th["messages"]
    window = msgs[-WINDOW_K:]
    full = " ".join(m["text"] for m in msgs)
    ent_ctx = " ".join(f"{k}: {'; '.join(v)}" for k, v in th["entities"].items())
    return {
        "mode": "real" if REAL else "off",
        "buffer": {"messages": msgs, "tokens": n_tokens(full)},
        "window": {"k": WINDOW_K, "kept": len(window), "dropped": len(msgs) - len(window),
                   "tokens": n_tokens(" ".join(m["text"] for m in window))},
        "summary": {"text": th["summary"], "tokens": n_tokens(th["summary"])},
        "entity": {"entities": th["entities"], "tokens": n_tokens(ent_ctx)},
        "short_term": {"thread_id": thread_id, "count": len(msgs),
                       "all_threads": user_threads(user_id), "threads_info": threads_info(user_id)},
        "users": users_info(),
        "long_term": {"user_id": user_id, "log": us["log"][-8:],
                      "facts": [{"key": f["key"], "text": f["text"]} for f in us["facts"]]},
        "semantic": {"facts": [f["text"] for f in us["facts"]], "hits": hits or []},
        "episodic": {"episodes": [{k: e[k] for k in ("id", "ts", "thread", "user", "ai")}
                                  for e in us["episodes"][-8:]], "hits": ep_hits or []},
        "warnings": [],
        "real_error": REAL_ERROR,
    }


def prepare(req: Chat):
    """Шаг 1 (под lock): проверки, поиск по mem0 и по эпизодам. Ответ ИИ ещё не запрашивали."""
    if not REAL:
        raise ApiError(503, "Реальный режим не запущен: " + (REAL_ERROR or "проверьте .env"))
    if (req.user_id not in users and len(users) >= MAX_USERS) or \
       ((req.user_id, req.thread_id) not in threads and len(user_threads(req.user_id)) >= MAX_THREADS_PER_USER):
        raise ApiError(400, "Достигнут лимит пользователей или тредов. Нажмите «Сбросить».")
    text = req.message.strip()
    if not text:
        raise ApiError(422, "Пустое сообщение.")

    warnings = []
    try:  # Semantic: настоящий mem0.search
        hits = real.search(req.user_id, text)
    except Exception as e:
        traceback.print_exc()
        hits = []
        warnings.append(f"mem0.search: {safe_err(e)}")

    q = embed(text)  # Episodic: похожие прошлые эпизоды с поправкой на свежесть
    now = time.time()
    ep_hits = []
    for e in users[req.user_id]["episodes"]:
        age_h = (now - e["ts"]) / 3600
        score = cosine(q, e["vec"]) * (1 / (1 + 0.1 * age_h))
        if score > 0.15:
            ep_hits.append({"id": e["id"], "text": e["user"], "thread": e["thread"], "score": round(score, 2)})
    ep_hits = sorted(ep_hits, key=lambda h: -h["score"])[:3]
    known = [f["text"] for f in users[req.user_id]["facts"]]
    return {"text": text, "hits": hits, "ep_hits": ep_hits, "known": known, "warnings": warnings, "now": now}


def commit(req: Chat, ctx: dict, reply: str):
    """Шаг 3 (под lock): записать сообщения, эпизод и обновить LangChain/mem0-память."""
    th, us = threads[(req.user_id, req.thread_id)], users[req.user_id]
    text, warnings = ctx["text"], ctx["warnings"]
    log_before = list(us["log"])

    th["messages"].append({"role": "human", "text": text})
    th["messages"].append({"role": "ai", "text": reply})
    del th["messages"][:-MAX_MSGS_PER_THREAD]

    us["episodes"].append({"id": (us["episodes"][-1]["id"] + 1) if us["episodes"] else 1, "ts": ctx["now"],
                           "thread": req.thread_id, "user": text, "ai": reply,
                           "vec": embed(text + " " + reply)})
    del us["episodes"][:-MAX_EPISODES]

    try:
        r = real.remember(req.user_id, req.thread_id, text, reply)
        th["summary"], th["entities"] = r["summary"], r["entities"]
        us["facts"] = [{**f, "vec": embed(f["text"])} for f in r["facts"]][:MAX_FACTS]
        us["log"] = (log_before + r["log"])[-50:]
    except Exception as e:
        traceback.print_exc()
        warnings.append(f"Ответ получен, но запись в mem0 не удалась — {safe_err(e)}")

    save_state()
    state = build_state(req.user_id, req.thread_id, ctx["hits"], ctx["ep_hits"])
    state["reply"] = reply
    state["warnings"] = warnings
    return state


def api_error_response(e: ApiError):
    return JSONResponse({"detail": e.detail}, status_code=e.status)


@app.post("/api/chat")
def chat(req: Chat):
    """Обычный запрос без стриминга (запасной вариант)."""
    try:
        with lock:
            ctx = prepare(req)
        try:
            reply = real.respond(req.user_id, req.thread_id, ctx["text"], ctx["hits"], ctx["known"])
        except Exception as e:
            traceback.print_exc()
            raise ApiError(502, f"Claude не ответил — {safe_err(e)}")
        if not reply:
            raise ApiError(502, "Claude вернул пустой ответ")
        with lock:
            return commit(req, ctx, reply)
    except ApiError as e:
        return api_error_response(e)


def sse(obj) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


@app.post("/api/chat/stream")
def chat_stream(req: Chat):
    """Ответ ИИ потоком (SSE): delta* → saving → state, либо error."""
    def gen():
        try:
            with lock:
                ctx = prepare(req)
            parts = []
            try:
                for piece in real.respond_stream(req.user_id, req.thread_id, ctx["text"], ctx["hits"], ctx["known"]):
                    parts.append(piece)
                    yield sse({"type": "delta", "text": piece})
            except Exception as e:
                traceback.print_exc()
                raise ApiError(502, f"Claude не ответил — {safe_err(e)}")
            reply = "".join(parts).strip()
            if not reply:
                raise ApiError(502, "Claude вернул пустой ответ")
            yield sse({"type": "saving"})  # ответ готов, дальше идёт запись в память (несколько секунд)
            with lock:
                state = commit(req, ctx, reply)
            yield sse({"type": "state", "state": state})
        except ApiError as e:
            yield sse({"type": "error", "detail": e.detail})
        except Exception as e:
            traceback.print_exc()
            yield sse({"type": "error", "detail": f"Ошибка сервера: {safe_err(e)}"})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.post("/api/state")
def state(ref: Ref):
    with lock:
        return build_state(ref.user_id, ref.thread_id)


def wipe_user(user_id: str):
    """Удаляет пользователя целиком: все его треды, эпизоды, факты и память mem0. Вызывать под lock."""
    warnings = []
    if REAL:
        try:
            real.reset(user_id)
        except Exception as e:
            warnings.append(f"mem0: {safe_err(e)}")
    for key in [k for k in threads if k[0] == user_id]:
        threads.pop(key, None)
    users.pop(user_id, None)
    save_state()
    return warnings


@app.post("/api/reset")
def reset(ref: Ref):
    """Сбрасывает только данные этого user_id, а не всех пользователей."""
    with lock:
        warnings = wipe_user(ref.user_id)
        s = build_state(ref.user_id, ref.thread_id)
        s["warnings"] = warnings
        return s


@app.post("/api/user/delete")
def delete_user(ref: Ref):
    """Удалить пользователя (имя) со всеми тредами и долговременной памятью."""
    with lock:
        if ref.user_id not in users and not any(k[0] == ref.user_id for k in threads):
            return JSONResponse({"detail": "Такого пользователя нет."}, status_code=404)
        warnings = wipe_user(ref.user_id)
        s = build_state(ref.user_id, ref.thread_id)
        s["warnings"] = warnings
        return s


@app.post("/api/thread/delete")
def delete_thread(ref: Ref):
    """Удалить один тред: сообщения, конспект, сущности и эпизоды этого треда.
    Факты о пользователе в mem0 остаются: они принадлежат пользователю, а не треду."""
    with lock:
        key = (ref.user_id, ref.thread_id)
        if key not in threads:
            return JSONResponse({"detail": "Такого треда нет."}, status_code=404)
        threads.pop(key, None)
        us = users.get(ref.user_id)
        if us:
            us["episodes"] = [e for e in us["episodes"] if e["thread"] != ref.thread_id]
        if REAL:
            real.delete_thread(ref.user_id, ref.thread_id)
        save_state()
        return build_state(ref.user_id, ref.thread_id)


@app.get("/api/health")
def health():
    """Пошаговая проверка REAL-режима: ключ → Claude → mem0. Показывает, на каком шаге ломается."""
    steps = [{"name": "Режим", "ok": REAL,
              "detail": "REAL" if REAL else (REAL_ERROR or "REAL не запущен")}]
    if REAL:
        for name, fn in (("Claude (ответ модели)", real.ping_llm), ("mem0 (память + эмбеддинги)", real.ping_mem0)):
            t0 = time.time()
            try:
                detail = fn()
                steps.append({"name": name, "ok": True, "detail": f"{detail} · {time.time() - t0:.1f} с"})
            except Exception as e:
                steps.append({"name": name, "ok": False, "detail": safe_err(e)})
    return {"ok": all(s["ok"] for s in steps), "mode": "real" if REAL else "off", "steps": steps}


@app.get("/")
def index():
    for p in (BASE / "static" / "index.html", BASE / "index.html"):
        if p.exists():
            return FileResponse(p)
    return JSONResponse({"detail": "index.html не найден рядом с main.py"}, status_code=404)
