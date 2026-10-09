"""
День 10. Три стратегии управления контекстом (без summary).

1) SlidingWindowStrategy — храним только последние N сообщений, остальное отбрасываем.
2) FactsStrategy         — отдельный блок "facts" (ключ-значение) с важными данными
                            диалога (цель, ограничения, предпочтения, решения),
                            обновляется после каждого сообщения пользователя;
                            в запрос уходит facts + последние N сообщений.
3) BranchingStrategy     — дерево сообщений: можно поставить checkpoint, создать
                            от него несколько веток и вести их независимо,
                            переключаясь между ветками.

Все стратегии провайдер-агностичны: вызовы LLM передаются снаружи как callable.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Optional
import json


def approx_tokens(text: str) -> int:
    """Грубая оценка числа токенов (~4 символа на токен). Для точных цифр
    в бою используйте response.usage из API-ответа."""
    if not text:
        return 1
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# 1. Sliding Window
# ---------------------------------------------------------------------------

@dataclass
class SlidingWindowStrategy:
    window_size: int = 6
    full_history: List[Dict] = field(default_factory=list)  # полный лог — для справки/сравнения

    def add_message(self, role: str, content: str) -> None:
        self.full_history.append({"role": role, "content": content})

    def get_context(self, system_prompt: str = "") -> Dict:
        window = self.full_history[-self.window_size:]
        return {"system": system_prompt, "messages": list(window)}

    def token_estimate(self, system_prompt: str = "") -> int:
        ctx = self.get_context(system_prompt)
        return approx_tokens(ctx["system"]) + sum(approx_tokens(m["content"]) for m in ctx["messages"])


# ---------------------------------------------------------------------------
# 2. Sticky Facts / Key-Value Memory
# ---------------------------------------------------------------------------

DEFAULT_FACTS_PROMPT = (
    "Ты — модуль извлечения фактов для ассистента, который собирает ТЗ с пользователем.\n"
    "Текущие известные факты (JSON):\n{facts_json}\n\n"
    "Новое сообщение пользователя:\n{message}\n\n"
    "Извлеки из сообщения новые или изменившиеся факты — цель проекта, ограничения, "
    "бюджет, сроки, предпочтения по стеку, принятые решения, договорённости. "
    "Верни ТОЛЬКО валидный JSON-объект вида {{\"ключ\": \"значение\"}} с новыми/изменившимися "
    "фактами (не повторяй факты, которые не изменились). Если новых фактов нет — верни {{}}. "
    "Никакого текста кроме JSON."
)


def default_facts_updater_factory(llm_call: Callable[[str], str]) -> Callable[[Dict, str], Dict]:
    """
    Оборачивает вызов LLM (строка-промпт -> строка-ответ) в facts_updater
    для FactsStrategy: (текущие facts, новое сообщение) -> обновления (dict).
    """
    def updater(current_facts: Dict, message: str) -> Dict:
        prompt = DEFAULT_FACTS_PROMPT.format(
            facts_json=json.dumps(current_facts, ensure_ascii=False),
            message=message,
        )
        raw = llm_call(prompt)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        return {}
    return updater


@dataclass
class FactsStrategy:
    window_size: int = 4
    facts_updater: Optional[Callable[[Dict, str], Dict]] = None  # (facts, new_user_msg) -> updates

    full_history: List[Dict] = field(default_factory=list)
    facts: Dict[str, str] = field(default_factory=dict)
    facts_update_calls: int = 0

    def add_message(self, role: str, content: str) -> None:
        self.full_history.append({"role": role, "content": content})
        if role == "user":
            self._update_facts(content)

    def _update_facts(self, message: str) -> None:
        if self.facts_updater is None:
            raise RuntimeError("Не задан facts_updater — функция для извлечения фактов через LLM")
        updates = self.facts_updater(self.facts, message)
        self.facts_update_calls += 1
        if updates:
            self.facts.update(updates)

    def get_context(self, system_prompt: str = "") -> Dict:
        sys_text = system_prompt
        if self.facts:
            facts_block = "\n".join(f"- {k}: {v}" for k, v in self.facts.items())
            sys_text = (sys_text + "\n\n" if sys_text else "") + f"Известные факты по проекту:\n{facts_block}"
        window = self.full_history[-self.window_size:]
        return {"system": sys_text, "messages": list(window)}

    def token_estimate(self, system_prompt: str = "") -> int:
        ctx = self.get_context(system_prompt)
        return approx_tokens(ctx["system"]) + sum(approx_tokens(m["content"]) for m in ctx["messages"])


# ---------------------------------------------------------------------------
# 3. Branching (ветки диалога)
# ---------------------------------------------------------------------------

class BranchingStrategy:
    """
    Дерево сообщений. Каждое сообщение — узел с указателем на родителя.
    "Ветка" — это просто указатель (branch_heads[name]) на узел-лист;
    история ветки восстанавливается подъёмом по родителям до корня.
    Checkpoint — просто именованная ссылка на узел, от которой можно
    создать новую ветку.
    """

    def __init__(self):
        self._nodes: Dict[int, Dict] = {}  # id -> {"role","content","parent"}
        self._next_id = 1
        self._nodes[0] = {"role": "root", "content": None, "parent": None}
        self.branch_heads: Dict[str, int] = {"main": 0}
        self.checkpoints: Dict[str, int] = {}
        self.active_branch = "main"

    def add_message(self, role: str, content: str, branch: Optional[str] = None) -> int:
        branch = branch or self.active_branch
        parent = self.branch_heads[branch]
        node_id = self._next_id
        self._next_id += 1
        self._nodes[node_id] = {"role": role, "content": content, "parent": parent}
        self.branch_heads[branch] = node_id
        return node_id

    def checkpoint(self, name: str, branch: Optional[str] = None) -> None:
        branch = branch or self.active_branch
        self.checkpoints[name] = self.branch_heads[branch]

    def create_branch(self, new_branch: str, from_checkpoint: Optional[str] = None,
                       from_branch: Optional[str] = None) -> None:
        if from_checkpoint is not None:
            start = self.checkpoints[from_checkpoint]
        else:
            start = self.branch_heads[from_branch or self.active_branch]
        self.branch_heads[new_branch] = start

    def switch(self, branch: str) -> None:
        if branch not in self.branch_heads:
            raise KeyError(f"Ветка '{branch}' не существует")
        self.active_branch = branch

    def list_branches(self) -> List[str]:
        return list(self.branch_heads.keys())

    def get_history(self, branch: Optional[str] = None) -> List[Dict]:
        branch = branch or self.active_branch
        node_id = self.branch_heads[branch]
        chain = []
        while node_id is not None and self._nodes[node_id]["content"] is not None:
            chain.append(self._nodes[node_id])
            node_id = self._nodes[node_id]["parent"]
        chain.reverse()
        return chain

    def get_context(self, system_prompt: str = "", branch: Optional[str] = None) -> Dict:
        history = self.get_history(branch)
        messages = [{"role": m["role"], "content": m["content"]} for m in history]
        return {"system": system_prompt, "messages": messages}

    def token_estimate(self, system_prompt: str = "", branch: Optional[str] = None) -> int:
        ctx = self.get_context(system_prompt, branch)
        return approx_tokens(ctx["system"]) + sum(approx_tokens(m["content"]) for m in ctx["messages"])
