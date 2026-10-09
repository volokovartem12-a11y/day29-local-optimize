"""
День 23. Генерация ответа по найденному контексту (база из дня 22).

Отличие от дня 22: если второй этап поиска отсёк ВСЕ фрагменты, модель не
вызывается — агент сразу отвечает, что в базе нет информации. Это экономит
запрос и исключает выдумку на пустом контексте.
"""

import os
import re

from build_index import build, index_path
from index_store import Index

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = "gpt-4o-mini"
NO_CONTEXT_ANSWER = "В базе нет информации для ответа на этот вопрос: подходящих фрагментов не найдено."

RAG_SYSTEM = (
    "Ты отвечаешь на вопросы по базе документации проектов. Используй ТОЛЬКО фрагменты "
    "из раздела «Контекст». После каждого утверждения ставь ссылку на фрагмент в виде [n]. "
    "Приводи точные названия, числа и имена из контекста. Если в контексте нет ответа — "
    "прямо скажи, что в базе нет ответа на этот вопрос, и ничего не придумывай. "
    "Отвечай кратко, на русском."
)

REFUSAL = re.compile(r"не знаю|нет информации|не располагаю|не могу (точно )?(ответить|сказать)|"
                     r"в базе нет|в контексте нет|нет данных|недостаточно (данных|информации)|"
                     r"не упоминается|не указан|не найден|отсутству", re.IGNORECASE)


class LLM:
    """Обёртка над AITunnel (OpenAI-совместимый API)."""

    def __init__(self, model: str = None):
        key = os.environ.get("AITUNNEL_API_KEY")
        if not key:
            raise SystemExit("Не задан AITUNNEL_API_KEY — ключ нужен для запросов к модели.")
        from openai import OpenAI
        self.model = model or os.environ.get("AITUNNEL_MODEL", DEFAULT_MODEL)
        self.client = OpenAI(api_key=key, base_url="https://api.aitunnel.ru/v1/")

    def chat(self, system: str, user: str, history: list = None) -> tuple:
        """history — продолжение диалога после user: [{"role": "assistant"|"user", "content": ...}]."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}] + (history or [])
        try:
            resp = self.client.chat.completions.create(model=self.model, messages=messages)
        except Exception as e:  # noqa: BLE001
            text = str(e).lower()
            if "model" in text and ("not" in text or "не найден" in text or "404" in text):
                raise SystemExit(f"Модель {self.model} недоступна в AITunnel: {e}\n"
                                 f"Попробуйте другую, например -Model gpt-4o-mini") from e
            raise
        usage = getattr(resp, "usage", None)
        return (resp.choices[0].message.content or "").strip(), {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        }


def load_index(backend: str = "aitunnel", strategy: str = "structure") -> Index:
    path = index_path(strategy, backend)
    if not os.path.exists(path):
        print(f"Индекс {os.path.relpath(path, BASE)} не найден — собираю ...")
        from types import SimpleNamespace as NS
        build(NS(docs=os.path.join(BASE, "docs"), backend=backend, model=None,
                 size=1000, overlap=150, max_chars=1500, min_chars=200))
    return Index(path)


def build_prompt(question: str, chunks: list) -> str:
    blocks = [f"[{i}] источник: {c['source']} | раздел: {c['section']}\n{c['text']}"
              for i, c in enumerate(chunks, 1)]
    return "Контекст:\n\n" + "\n\n---\n\n".join(blocks) + f"\n\nВопрос: {question}"


def cited_sources(answer: str, chunks: list) -> list:
    nums = {int(n) for group in re.findall(r"\[([\d,\s]+)\]", answer) for n in re.findall(r"\d+", group)}
    return sorted({chunks[n - 1]["source"] for n in nums if 1 <= n <= len(chunks)})


def is_refusal(answer: str) -> bool:
    return bool(REFUSAL.search(answer))


def answer(llm, question: str, chunks: list) -> dict:
    if not chunks:
        return {"answer": NO_CONTEXT_ANSWER, "cited": [], "short_circuit": True,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0}, "prompt_chars": 0}
    prompt = build_prompt(question, chunks)
    text, usage = llm.chat(RAG_SYSTEM, prompt)
    return {"answer": text, "cited": cited_sources(text, chunks), "short_circuit": False,
            "usage": usage, "prompt_chars": len(prompt)}
