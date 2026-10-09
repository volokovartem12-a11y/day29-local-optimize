"""
База знаний сервиса: поиск статей и выжимка текста.
Логика для MCP-сервера kb (поиск и суммаризация взяты из дня 19).
"""

import hashlib
import re
from collections import Counter


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


KNOWLEDGE_BASE = [
    {"id": "kb-1", "title": "Шиномонтаж и сезонная смена шин", "text": (
        "Сезонную смену шин рекомендуют делать, когда среднесуточная температура держится около плюс семи градусов. "
        "Шиномонтаж включает снятие колёс, перебортовку, балансировку и проверку давления. "
        "После шиномонтажа через пятьдесят километров пробега нужно протянуть колёсные болты. "
        "Клиенту стоит напомнить о записи заранее: в сезон очереди на шиномонтаж достигают недели.")},
    {"id": "kb-2", "title": "Хранение шин", "text": (
        "Шины хранят в сухом прохладном помещении без прямого солнца. "
        "Шины на дисках хранят в подвешенном состоянии или стопкой, без дисков — вертикально с поворотом раз в месяц. "
        "Сервис может предложить клиенту сезонное хранение шин как дополнительную услугу.")},
    {"id": "kb-3", "title": "Замена масла", "text": (
        "Моторное масло меняют каждые десять–пятнадцать тысяч километров или раз в год. "
        "При тяжёлых условиях эксплуатации интервал сокращают до семи тысяч километров. "
        "Вместе с маслом обязательно меняют масляный фильтр.")},
    {"id": "kb-4", "title": "Диагностика подвески", "text": (
        "Диагностика подвески выявляет износ амортизаторов, шаровых опор и сайлентблоков. "
        "Стук на неровностях и увод автомобиля в сторону — повод записаться на диагностику. "
        "После замены элементов подвески обязательна регулировка углов установки колёс.")},
    {"id": "kb-5", "title": "Тормозные колодки", "text": (
        "Передние тормозные колодки обычно служат двадцать–сорок тысяч километров. "
        "Скрип при торможении и увеличение тормозного пути говорят об износе колодок. "
        "Колодки меняют комплектом на одной оси.")},
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


def summarize(text: str, max_sentences: int = 3) -> dict:
    """Сжать текст до max_sentences ключевых предложений (локально, без LLM)."""
    if not text.strip():
        raise ValueError("Нечего суммаризировать: пустой текст")
    if not 1 <= max_sentences <= 20:
        raise ValueError("max_sentences должен быть от 1 до 20")
    sentences = _extractive_summary(text, max_sentences)
    summary = " ".join(sentences)
    return {"summary": summary, "sentences": sentences, "input_chars": len(text),
            "output_chars": len(summary), "input_sha256": sha256(text)}
