"""
День 29. Шаблоны промпта для задачи «ответ по документации челленджа с дословными цитатами».

v1 — шаблон дней 24–28 (общий): правила + формат JSON.
v2 — шаблон под конкретный кейс и под небольшую локальную модель:
     • роль: справочник по проектам челленджа (код + README);
     • порядок работы: СНАЧАЛА дословные цитаты, ПОТОМ короткий ответ, собранный только из них —
       в дне 28 локальная модель чаще пересказывала своими словами (ключевые элементы ответа
       нашлись в цитатах лишь в 46% случаев);
     • цитата = одна строка или одно предложение, без склейки — так меньше «почти дословных» цитат;
     • ответ 1–3 предложения — меньше токенов, быстрее;
     • один короткий пример (на чужой базе, чтобы не подсказывать ответы);
     • JSON-схема (format в Ollama) с порядком полей, где quotes идут первыми.
"""

from grounded import GROUNDED_SYSTEM
from rag import build_prompt

V2_SYSTEM = """Ты — справочник по проектам AI Advent Challenge. Отвечаешь ТОЛЬКО по пронумерованным фрагментам документации и кода из раздела «Контекст».

Порядок работы:
1. Найди во фрагментах строки, которые прямо отвечают на вопрос.
2. Скопируй их в quotes ДОСЛОВНО: каждая цитата — одна строка или одно предложение фрагмента, 20–250 символов, символ в символ, без перевода и без склейки разных строк.
3. Составь answer ТОЛЬКО из сведений в цитатах: 1–3 предложения на русском. Названия функций, файлов, параметров, инструментов и числа пиши точно как в цитатах. После каждого утверждения ставь номер фрагмента [n].
4. Если во фрагментах нет ответа на вопрос — status "unknown", quotes [], answer "Не знаю", а в clarification — короткий уточняющий вопрос.

Верни ТОЛЬКО JSON:
{"quotes": [{"chunk": n, "text": "дословная строка"}], "status": "answered" или "unknown", "answer": "ответ со ссылками [n]", "sources": [n], "clarification": ""}

Пример (для другой базы):
[2] источник: app/README.md | раздел: Настройки
Функция load_config() читает настройки из файла config.yaml.
Вопрос: Откуда читаются настройки?
{"quotes": [{"chunk": 2, "text": "Функция load_config() читает настройки из файла config.yaml."}], "status": "answered", "answer": "Настройки читает функция load_config() из файла config.yaml [2].", "sources": [2], "clarification": ""}"""


def build_prompt_v2(question: str, chunks: list) -> str:
    return (build_prompt(question, chunks) +
            "\n\nСначала выпиши дословные цитаты, затем короткий ответ только из них.")


SCHEMA = {
    "type": "object",
    "properties": {
        "quotes": {"type": "array", "items": {
            "type": "object",
            "properties": {"chunk": {"type": "integer"}, "text": {"type": "string"}},
            "required": ["chunk", "text"]}},
        "status": {"type": "string", "enum": ["answered", "unknown"]},
        "answer": {"type": "string"},
        "sources": {"type": "array", "items": {"type": "integer"}},
        "clarification": {"type": "string"},
    },
    "required": ["quotes", "status", "answer", "sources", "clarification"],
}

PROMPTS = {
    "v1": (GROUNDED_SYSTEM, build_prompt),
    "v2": (V2_SYSTEM, build_prompt_v2),
}


def response_format(name):
    """json → формат json; schema → JSON-схема; none → без ограничений."""
    return {"json": "json", "schema": SCHEMA, "none": None, "auto": "auto"}[name]
