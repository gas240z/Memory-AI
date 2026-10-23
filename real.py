"""Настоящие реализации: Claude + mem0, а четыре вида short-term памяти (buffer, window,
summary, entity) собраны напрямую поверх Claude — так же, как работали Conversation*Memory
из LangChain. Старый langchain.memory не импортируется на Python 3.14, поэтому он не нужен.
Подключается из main.py, если в .env задан ANTHROPIC_API_KEY и USE_REAL_MEMORY=true.
"""
import json
import logging
import os
import re
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("MEM0_TELEMETRY", "False")  # без отправки статистики mem0 (быстрее, без сети)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # без полосок загрузки в консоли
os.environ.setdefault("TQDM_DISABLE", "1")  # «Loading weights…»
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", message=".*get_sentence_embedding_dimension.*")
logging.getLogger("mem0").setLevel(logging.ERROR)  # предупреждения про spaCy и fastembed — необязательные расширения

from langchain_anthropic import ChatAnthropic  # noqa: E402
from mem0 import Memory  # noqa: E402

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
WINDOW_EXCHANGES = int(os.getenv("WINDOW_EXCHANGES", "2"))  # 2 обмена = 4 сообщения
TIMEOUT = float(os.getenv("LLM_TIMEOUT", "40"))  # секунд; чтобы чат не висел вечно

_llm = None
_mem0 = None
_threads = {}  # (user_id, thread_id) -> {"messages", "summary", "entities"} (short-term)
_init_lock = threading.Lock()  # mem0/Qdrant нельзя инициализировать из двух потоков сразу
_pool = ThreadPoolExecutor(max_workers=3)


def llm():
    global _llm
    if _llm is None:
        # temperature не передаём: новые модели Claude её не принимают
        _llm = ChatAnthropic(model=MODEL, max_tokens=1024, timeout=TIMEOUT, max_retries=1)
    return _llm


def mem0():
    """mem0: LLM = Claude, векторное хранилище = локальный Qdrant в ./qdrant_data."""
    global _mem0
    with _init_lock:
        if _mem0 is None:
            if os.getenv("MEM0_EMBEDDER", "huggingface") == "openai":
                embedder = {"provider": "openai", "config": {"model": "text-embedding-3-small"}}
                dims = 1536
            else:  # локально и бесплатно, понимает русский
                embedder = {"provider": "huggingface", "config": {
                    "model": os.getenv("MEM0_HF_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")}}
                dims = 384
            _mem0 = Memory.from_config({
                "llm": {"provider": "anthropic", "config": {"model": MODEL, "max_tokens": 1500,
                                                  "enable_sampling_parameters": False}},  # без temperature
                "embedder": embedder,
                "vector_store": {"provider": "qdrant", "config": {
                    "collection_name": "memory_lab", "path": "./qdrant_data", "embedding_model_dims": dims}},
                "history_db_path": "./mem0_history.db",
            })
    return _mem0


def close():
    """Аккуратно закрывает Qdrant и SQLite, чтобы при остановке не было ImportError в __del__."""
    global _mem0
    m, _mem0 = _mem0, None
    if m is None:
        return
    for closer in (lambda: m.vector_store.client.close(), lambda: m.close()):
        try:
            closer()
        except Exception:
            pass


def _text(content) -> str:
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content).strip()


def _chunk_text(content) -> str:
    """Текст куска стрима БЕЗ strip — пробелы между словами важны."""
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type", "text") == "text")
    return content if isinstance(content, str) else ""


def thread(user_id: str, thread_id: str) -> dict:
    """Short-term: у каждой пары (user_id, thread_id) своя память диалога."""
    return _threads.setdefault((user_id, thread_id),
                               {"messages": [], "summary": "", "entities": {}})


def load_threads(items):
    """Восстановить память диалогов после перезапуска сервера (данные читает main.py с диска)."""
    for it in items:
        _threads[(it["user"], it["thread"])] = {
            "messages": list(it.get("messages", [])),
            "summary": it.get("summary", ""),
            "entities": dict(it.get("entities", {})),
        }


def _results(res):
    return res.get("results", []) if isinstance(res, dict) else res


def search(user_id: str, text: str):
    """Semantic: mem0.search по смыслу (mem0 2.x принимает user_id только внутри filters)."""
    res = mem0().search(text, filters={"user_id": user_id}, top_k=3)
    return [{"text": r["memory"], "score": round(r.get("score") or 0, 2)} for r in _results(res)]


# ---------- сборка промпта из ВСЕХ видов памяти ----------
def _messages(user_id: str, thread_id: str, text: str, hits, known):
    th = thread(user_id, thread_id)
    window = th["messages"][-WINDOW_EXCHANGES * 2:]  # Window memory
    facts, seen = [], set()
    for t in [h["text"] for h in hits] + list(known or []):  # Semantic + Long-term (mem0)
        if t not in seen:
            seen.add(t)
            facts.append(f"- {t}")
    ents = "\n".join(f"- {k}: {'; '.join(v) if isinstance(v, list) else v}" for k, v in th["entities"].items())
    system = ("Ты дружелюбный ассистент с памятью. Отвечай кратко, на языке пользователя. "
              "Опирайся на память ниже. Если чего-то о пользователе в памяти нет — так и скажи, "
              "не выдумывай (например, не называй имя, которого не знаешь).\n\n"
              f"Долгосрочные факты о пользователе (mem0):\n{chr(10).join(facts) or '(пока ничего)'}\n\n"
              f"Конспект этого разговора (summary):\n{th['summary'] or '(пусто)'}\n\n"
              f"Сущности из разговора (entity):\n{ents or '(нет)'}")
    messages = [("system", system)]
    messages += [("human" if m["role"] == "human" else "ai", m["text"]) for m in window]
    messages.append(("human", text))
    return messages


def respond(user_id: str, thread_id: str, text: str, hits, known=None):
    return _text(llm().invoke(_messages(user_id, thread_id, text, hits, known)).content)


def respond_stream(user_id: str, thread_id: str, text: str, hits, known=None):
    """Генератор: отдаёт ответ Claude кусками по мере генерации."""
    for chunk in llm().stream(_messages(user_id, thread_id, text, hits, known)):
        piece = _chunk_text(chunk.content)
        if piece:
            yield piece


# ---------- Summary и Entity: то, что раньше делали ConversationSummary/EntityMemory ----------
def _summarize(prev: str, text: str, reply: str) -> str:
    prompt = ("Постепенно обновляй краткое резюме разговора. Пиши кратко, на языке пользователя, "
              "только факты.\n\n"
              f"Текущее резюме:\n{prev or '(пусто)'}\n\n"
              f"Новые реплики:\nПользователь: {text}\nИИ: {reply}\n\n"
              "Новое резюме:")
    return _text(llm().invoke([("human", prompt)]).content)


def _extract_entities(text: str, known: dict):
    prompt = ("Найди в реплике пользователя людей, места, организации и проекты (сущности) и коротко, "
              "одним предложением, что о них сказано. Ответь ТОЛЬКО JSON-объектом "
              '{"Сущность": "факт"}. Если сущностей нет — {}.\n'
              f"Уже известные сущности: {', '.join(known) or 'нет'}\n\nРеплика: {text}")
    raw = _text(llm().invoke([("human", prompt)]).content)
    m = re.search(r"\{.*\}", raw, re.S)
    data = json.loads(m.group(0)) if m else {}
    return {str(k): str(v) for k, v in data.items() if v}


def remember(user_id: str, thread_id: str, text: str, reply: str):
    th = thread(user_id, thread_id)
    th["messages"] += [{"role": "human", "text": text}, {"role": "ai", "text": reply}]  # Buffer

    # три независимых вызова модели идут параллельно, чтобы ответ не ждал каждый по очереди
    f_sum = _pool.submit(_summarize, th["summary"], text, reply)
    f_ent = _pool.submit(_extract_entities, text, th["entities"])
    f_mem = _pool.submit(mem0().add, [{"role": "user", "content": text},
                                       {"role": "assistant", "content": reply}], user_id=user_id)
    errors = []
    try:
        th["summary"] = f_sum.result()
    except Exception as e:
        errors.append(f"summary: {type(e).__name__}: {e}")
    try:
        for k, v in f_ent.result().items():
            bucket = th["entities"].setdefault(k, [])
            if v not in bucket:
                bucket.append(v)
    except Exception as e:
        errors.append(f"entity: {type(e).__name__}: {e}")
    added = f_mem.result()  # ошибку mem0 отдаём наверх: без неё долговременной памяти нет

    log = [{"op": r.get("event", "ADD"), "text": r.get("memory", ""), "old": r.get("previous_memory")}
           for r in _results(added)]
    facts = [{"key": f["id"], "text": f["memory"]}
             for f in _results(mem0().get_all(filters={"user_id": user_id}, top_k=50))]
    if errors:
        print("[Memory Lab] частичные ошибки памяти:", "; ".join(errors))
    return {"summary": th["summary"], "entities": th["entities"], "log": log, "facts": facts}


def ping_llm():
    """Диагностика: один короткий запрос к Claude."""
    out = _text(llm().invoke([("human", "Ответь одним словом: ок")]).content)
    return f"модель {MODEL} ответила: {out[:40]!r}"


def ping_mem0():
    """Диагностика: инициализация mem0 (скачивает эмбеддинги при первом запуске) и поиск."""
    mem0().search("тест", filters={"user_id": "__health__"}, top_k=1)
    return "mem0 и Qdrant работают"


def delete_thread(user_id: str, thread_id: str):
    _threads.pop((user_id, thread_id), None)


def reset(user_id: str):
    for key in [k for k in _threads if k[0] == user_id]:
        _threads.pop(key, None)
    try:
        mem0().delete_all(user_id=user_id)
    except Exception:
        pass
