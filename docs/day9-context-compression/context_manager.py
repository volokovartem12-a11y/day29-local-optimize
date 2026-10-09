"""
Менеджер контекста с сжатием истории диалога.

Идея:
- Храним ПОЛНУЮ историю (full_history) — для отладки и сравнения.
- В запрос к модели подставляем НЕ всю историю, а:
    [system с summary] + [последние N сообщений "как есть"]
- Каждые SUMMARIZE_EVERY сообщений старые сообщения "сворачиваются"
  в summary отдельным вызовом LLM, а сами сообщения из "рабочего окна" удаляются.
- Summary хранится отдельно и на каждом сжатии дополняется (не перезаписывается с нуля),
  чтобы не терять факты из более ранних сверток.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Optional


def approx_tokens(text: str) -> int:
    """Грубая оценка числа токенов (~4 символа на токен для англ/рус смеси).
    Для точных цифр в бою используйте response.usage из API-ответа."""
    return max(1, len(text) // 4)


@dataclass
class ContextManager:
    keep_last_n: int = 6            # сколько последних сообщений хранить "как есть"
    summarize_every: int = 10       # раз в сколько сообщений делать сжатие
    summarizer: Optional[Callable[[List[Dict]], str]] = None  # функция вызова LLM для саммари

    full_history: List[Dict] = field(default_factory=list)   # вся история без сжатия (для сравнения)
    working_history: List[Dict] = field(default_factory=list)  # рабочее окно (то, что не свернуто)
    summary: str = ""               # текущее накопленное summary
    messages_since_summary: int = 0
    summary_calls: int = 0

    def add_message(self, role: str, content: str) -> None:
        msg = {"role": role, "content": content}
        self.full_history.append(msg)
        self.working_history.append(msg)
        self.messages_since_summary += 1

        if self.messages_since_summary >= self.summarize_every:
            self._compress()

    def _compress(self) -> None:
        """Сворачиваем всё, кроме последних keep_last_n сообщений, в summary."""
        if len(self.working_history) <= self.keep_last_n:
            self.messages_since_summary = 0
            return

        to_summarize = self.working_history[:-self.keep_last_n]
        keep = self.working_history[-self.keep_last_n:]

        if self.summarizer is None:
            raise RuntimeError("Не задан summarizer — функция для вызова LLM-суммаризации")

        new_summary_piece = self.summarizer(to_summarize)
        self.summary_calls += 1

        if self.summary:
            self.summary = self.summary + "\n" + new_summary_piece
        else:
            self.summary = new_summary_piece

        self.working_history = keep
        self.messages_since_summary = 0

    def get_context_compressed(self, system_prompt: str = "") -> Dict:
        """Контекст ДЛЯ ЗАПРОСА в модель: system(+summary) + последние N сообщений."""
        sys_text = system_prompt
        if self.summary:
            sys_text = (sys_text + "\n\n" if sys_text else "") + \
                f"Краткое содержание более ранней части диалога:\n{self.summary}"
        return {"system": sys_text, "messages": list(self.working_history)}

    def get_context_full(self, system_prompt: str = "") -> Dict:
        """Контекст БЕЗ сжатия — вся история как есть (для сравнения)."""
        return {"system": system_prompt, "messages": list(self.full_history)}

    def token_estimate(self, system_prompt: str = "", compressed: bool = True) -> int:
        ctx = self.get_context_compressed(system_prompt) if compressed \
            else self.get_context_full(system_prompt)
        total = approx_tokens(ctx["system"])
        for m in ctx["messages"]:
            total += approx_tokens(m["content"])
        return total


def default_summarizer_factory(llm_call: Callable[[str], str]) -> Callable[[List[Dict]], str]:
    """
    Оборачивает обычный вызов LLM (принимает строку-промпт, возвращает строку-ответ)
    в summarizer для ContextManager.
    """
    def summarizer(messages: List[Dict]) -> str:
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        prompt = (
            "Сожми следующий фрагмент диалога в краткое summary. "
            "Сохрани все важные факты, решения, имена, числа и договорённости. "
            "Не добавляй ничего от себя. Пиши компактно, списком если уместно.\n\n"
            f"{transcript}"
        )
        return llm_call(prompt)
    return summarizer
