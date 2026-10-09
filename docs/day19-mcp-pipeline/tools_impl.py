"""
День 19. Логика трёх инструментов — без зависимости от MCP.

mcp_server.py только регистрирует эти функции как MCP-инструменты.
Такое разделение позволяет тестировать логику напрямую.

  search(query, top_k)            — ПОЛУЧАЕТ данные из базы статей
  summarize(text, max_sentences)  — ОБРАБАТЫВАЕТ текст (выжимка)
  save_to_file(filename, content) — СОХРАНЯЕТ результат в папку output/

Для проверки передачи данных каждый инструмент возвращает sha256
того, что он получил или записал: клиент сравнивает эти хеши со своими.
"""

import hashlib
import os
import re
from collections import Counter
from datetime import datetime

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 1. search — источник данных
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE = [
    {
        "id": "kb-1",
        "title": "Что такое MCP",
        "text": (
            "Model Context Protocol (MCP) — открытый протокол, который стандартизирует подключение "
            "языковых моделей к внешним инструментам и данным. MCP-сервер публикует инструменты, "
            "ресурсы и промпты, а MCP-клиент подключается к серверу и вызывает их. "
            "Общение идёт по JSON-RPC поверх транспорта stdio или Streamable HTTP. "
            "Каждый инструмент описывается именем, описанием и JSON Schema входных параметров. "
            "Благодаря единому протоколу один и тот же сервер можно подключить к разным агентам."
        ),
    },
    {
        "id": "kb-2",
        "title": "Композиция MCP-инструментов",
        "text": (
            "Композиция инструментов — это передача результата одного инструмента на вход другому. "
            "Типичный пайплайн состоит из трёх шагов: получение данных, обработка и сохранение. "
            "Оркестрацию может выполнять обычный код клиента или сама языковая модель. "
            "Детерминированный код надёжнее, а модель гибче, но может ошибиться в аргументах. "
            "Важно проверять, что данные между шагами передаются без потерь и искажений. "
            "Для этого удобно сравнивать контрольные суммы на каждом шаге цепочки."
        ),
    },
    {
        "id": "kb-3",
        "title": "RAG: поиск плюс генерация",
        "text": (
            "Retrieval-Augmented Generation (RAG) дополняет запрос к модели найденными документами. "
            "Сначала система ищет релевантные фрагменты в базе знаний, затем передаёт их модели как контекст. "
            "Это снижает количество выдуманных фактов и позволяет работать со свежими данными. "
            "Качество RAG сильно зависит от качества поиска и нарезки документов на фрагменты. "
            "Поиск бывает по ключевым словам, по эмбеддингам или гибридным."
        ),
    },
    {
        "id": "kb-4",
        "title": "Окно контекста и токены",
        "text": (
            "Окно контекста ограничивает, сколько токенов модель обрабатывает за один запрос. "
            "Длинная история диалога быстро расходует окно и увеличивает стоимость запросов. "
            "Для экономии применяют скользящее окно, сжатие истории в summary и выделение ключевых фактов. "
            "Суммаризация перед передачей данных в модель уменьшает расход токенов в пайплайнах."
        ),
    },
    {
        "id": "kb-5",
        "title": "Function calling",
        "text": (
            "Function calling позволяет модели запросить вызов функции вместо текстового ответа. "
            "Модель получает список функций с JSON Schema параметров и возвращает имя функции и аргументы. "
            "Приложение выполняет функцию и передаёт результат обратно модели. "
            "MCP-инструменты легко переводятся в формат function calling, поэтому агент может вызывать их через модель."
        ),
    },
]

_WORD_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)


def _words(text: str) -> list:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _stem(word: str) -> str:
    """Очень грубое приведение к основе: отрезаем окончание, чтобы «инструментов» ≈ «инструменты»."""
    return word[:6] if len(word) > 6 else word


def search(query: str, top_k: int = 3) -> dict:
    """Поиск статей по ключевым словам. Возвращает документы, отсортированные по релевантности."""
    if not query.strip():
        raise ValueError("Пустой поисковый запрос")
    if not 1 <= top_k <= 10:
        raise ValueError("top_k должен быть от 1 до 10")

    q_stems = {_stem(w) for w in _words(query) if len(w) > 2}
    scored = []
    for doc in KNOWLEDGE_BASE:
        title_stems = Counter(_stem(w) for w in _words(doc["title"]))
        text_stems = Counter(_stem(w) for w in _words(doc["text"]))
        score = sum(title_stems[s] * 3 + text_stems[s] for s in q_stems)
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda x: (-x[0], x[1]["id"]))

    results = [{**doc, "score": score} for score, doc in scored[:top_k]]
    combined = "\n\n".join(f"{d['title']}. {d['text']}" for d in results)
    return {
        "query": query,
        "found": len(results),
        "results": results,
        "combined_text": combined,              # готовый вход для следующего шага
        "combined_sha256": sha256(combined),    # контрольная сумма для проверки передачи
    }


# ---------------------------------------------------------------------------
# 2. summarize — обработка
# ---------------------------------------------------------------------------

_STOPWORDS = set("""и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только
ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до
вас нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем
была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой совсем ним
здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при наконец два об
другой хоть после над больше тот через эти нас про всего них какая много разве три эту моя впрочем хорошо
свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю между это которые который
также затем сначала""".split())


def _split_sentences(text: str) -> list:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 20]


def _extractive_summary(text: str, max_sentences: int) -> list:
    """Выбираем самые «весомые» предложения по частоте значимых слов, сохраняя исходный порядок."""
    sentences = _split_sentences(text)
    freq = Counter(_stem(w) for w in _words(text) if w not in _STOPWORDS and len(w) > 3)
    if not freq:
        return sentences[:max_sentences]
    top = max(freq.values())

    def score(sentence):
        stems = [_stem(w) for w in _words(sentence) if w not in _STOPWORDS and len(w) > 3]
        return sum(freq[s] / top for s in stems) / (len(stems) ** 0.5) if stems else 0

    best = sorted(range(len(sentences)), key=lambda i: -score(sentences[i]))[:max_sentences]
    return [sentences[i] for i in sorted(best)]


def _llm_summary(text: str, max_sentences: int) -> list:
    key = os.environ.get("AITUNNEL_API_KEY")
    if not key:
        raise ValueError("Режим llm требует AITUNNEL_API_KEY")
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ValueError("Для режима llm нужен пакет openai: pip install openai") from e

    client = OpenAI(api_key=key, base_url="https://api.aitunnel.ru/v1/")
    try:
        resp = client.chat.completions.create(
            model=os.environ.get("AITUNNEL_MODEL", "gpt-4o-mini"),
            messages=[{
                "role": "user",
                "content": f"Сделай выжимку текста ровно из {max_sentences} предложений, на русском, "
                           f"только факты из текста, каждое предложение с новой строки:\n\n{text}",
            }],
        )
    except Exception as e:  # noqa: BLE001 — 401, сеть, неизвестная модель → понятная ошибка шага
        raise ValueError(f"AITunnel не ответил: {type(e).__name__}: {e}") from e
    lines = [ln.strip(" -•\t") for ln in (resp.choices[0].message.content or "").splitlines()]
    sentences = [ln for ln in lines if ln][:max_sentences]
    if not sentences:
        raise ValueError("Модель вернула пустую выжимку")
    return sentences


def summarize(text: str, max_sentences: int = 4, mode: str = "extractive") -> dict:
    """Сжать текст до max_sentences предложений. mode: extractive (локально) или llm (через AITunnel)."""
    if not text.strip():
        raise ValueError("Нечего суммаризировать: пустой текст")
    if not 1 <= max_sentences <= 20:
        raise ValueError("max_sentences должен быть от 1 до 20")
    if mode not in ("extractive", "llm"):
        raise ValueError("mode должен быть 'extractive' или 'llm'")

    sentences = _extractive_summary(text, max_sentences) if mode == "extractive" \
        else _llm_summary(text, max_sentences)
    summary = " ".join(sentences)
    return {
        "mode": mode,
        "summary": summary,
        "sentences": sentences,
        "input_chars": len(text),
        "output_chars": len(summary),
        "compression": round(len(summary) / len(text), 3),
        "input_sha256": sha256(text),  # что инструмент реально получил — для проверки передачи
    }


# ---------------------------------------------------------------------------
# 3. save_to_file — сохранение
# ---------------------------------------------------------------------------

def _safe_filename(filename: str) -> str:
    """Только имя файла, без путей: защищает от записи за пределы папки output/."""
    name = os.path.basename(filename.strip().replace("\\", "/"))
    name = re.sub(r"[^\w.\-]", "_", name)
    if not name or name.startswith("."):
        raise ValueError("Некорректное имя файла")
    if not name.lower().endswith((".md", ".txt")):
        name += ".md"
    return name


def save_to_file(filename: str, content: str) -> dict:
    """Сохранить текст в папку output/. Возвращает путь, размер и sha256 записанного."""
    if not content.strip():
        raise ValueError("Пустое содержимое — сохранять нечего")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, _safe_filename(filename))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    with open(path, "r", encoding="utf-8", newline="") as f:
        written = f.read()
    return {
        "path": path,
        "bytes": len(written.encode("utf-8")),
        "sha256": sha256(written),  # хеш того, что реально лежит на диске
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
