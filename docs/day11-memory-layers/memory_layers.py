"""
День 11. Модель памяти агента: три слоя памяти (memory layers).

МОДЕЛЬ ПАМЯТИ
=============

1) ShortTermMemory — КРАТКОСРОЧНАЯ (текущий диалог)
   Что хранит: последние N реплик диалога дословно.
   Время жизни: до вытеснения из окна (скользящее окно).
   Зачем: связность разговора — местоимения, "а что если", уточнения.
   Сбрасывается: при /reset, при старте новой сессии.

2) WorkingMemory — РАБОЧАЯ (данные текущей задачи)
   Что хранит: слоты текущей задачи — что за задача, какие параметры
   уже собраны, что ещё не заполнено, промежуточные результаты.
   Время жизни: пока задача не закрыта (/task done).
   Зачем: агент не теряет нить задачи, даже если реплики уже вышли
   из окна краткосрочной памяти.
   Сбрасывается: при закрытии/смене задачи.

3) LongTermMemory — ДОЛГОВРЕМЕННАЯ (профиль, решения, знания)
   Что хранит: три раздела —
     - profile:   устойчивое про пользователя (имя, роль, стек, часовой пояс)
     - decisions: принятые решения и договорённости (с меткой времени)
     - knowledge: факты о предметной области/проекте
   Время жизни: постоянно, переживает перезапуск (файл memory_store.json).
   Зачем: агент помнит пользователя и прошлые решения между сессиями.
   Сбрасывается: только явной командой /forget.

МАРШРУТИЗАЦИЯ (что и куда сохраняется)
======================================
Есть два режима, и оба явные — это принципиально для задания:

  - РУЧНОЙ: пользователь сам командует, куда писать
      /remember profile стек=FastAPI
      /task set бюджет=400000
  - АВТО: LLM-роутер на каждую реплику решает, в какой слой
      положить извлечённые данные, и ВОЗВРАЩАЕТ ОБОСНОВАНИЕ.
      Решение роутера всегда видно (route_log), его можно
      посмотреть командой /routing — то есть маршрутизация
      остаётся явной, а не "магией внутри".

Каждый слой умеет отдавать свой фрагмент контекста (render_for_prompt),
и MemoryManager собирает из них итоговый system-промпт. Так видно,
какой именно слой повлиял на ответ.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional
import json
import os


def approx_tokens(text: str) -> int:
    """Грубая оценка токенов (~4 символа на токен). Для точных цифр
    используйте usage из ответа API."""
    if not text:
        return 0
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Слой 1: краткосрочная память
# ---------------------------------------------------------------------------

@dataclass
class ShortTermMemory:
    """Текущий диалог: последние N реплик дословно."""
    window_size: int = 6
    messages: List[Dict] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def get_window(self) -> List[Dict]:
        return self.messages[-self.window_size:]

    def clear(self) -> None:
        self.messages = []

    def render_for_prompt(self) -> str:
        # краткосрочная память уходит не в system, а в messages —
        # здесь только для отладочного показа
        return "\n".join(f"{m['role']}: {m['content']}" for m in self.get_window())

    def stats(self) -> Dict:
        window = self.get_window()
        return {
            "всего реплик": len(self.messages),
            "в окне": len(window),
            "токенов в окне (оценка)": sum(approx_tokens(m["content"]) for m in window),
        }


# ---------------------------------------------------------------------------
# Слой 2: рабочая память
# ---------------------------------------------------------------------------

@dataclass
class WorkingMemory:
    """Данные текущей задачи: название, слоты, промежуточные результаты."""
    task_name: Optional[str] = None
    slots: Dict[str, str] = field(default_factory=dict)
    required_slots: List[str] = field(default_factory=list)
    scratch: List[str] = field(default_factory=list)

    def start_task(self, name: str, required: Optional[List[str]] = None) -> None:
        self.task_name = name
        self.slots = {}
        self.required_slots = list(required or [])
        self.scratch = []

    def set_slot(self, key: str, value: str) -> None:
        self.slots[key] = value

    def add_scratch(self, note: str) -> None:
        self.scratch.append(note)

    def missing_slots(self) -> List[str]:
        return [s for s in self.required_slots if s not in self.slots]

    def finish_task(self) -> Dict:
        """Закрывает задачу и возвращает её итог (его обычно переносят в долговременную)."""
        result = {"task": self.task_name, "slots": dict(self.slots)}
        self.task_name = None
        self.slots = {}
        self.required_slots = []
        self.scratch = []
        return result

    def clear(self) -> None:
        self.finish_task()

    def render_for_prompt(self) -> str:
        if not self.task_name and not self.slots:
            return ""
        lines = []
        if self.task_name:
            lines.append(f"Текущая задача: {self.task_name}")
        if self.slots:
            lines.append("Собранные данные задачи:")
            lines += [f"  - {k}: {v}" for k, v in self.slots.items()]
        missing = self.missing_slots()
        if missing:
            lines.append(f"Ещё не заполнено: {', '.join(missing)}")
        if self.scratch:
            lines.append("Промежуточные заметки:")
            lines += [f"  - {s}" for s in self.scratch]
        return "\n".join(lines)

    def stats(self) -> Dict:
        return {
            "задача": self.task_name or "(нет активной)",
            "слотов заполнено": len(self.slots),
            "не заполнено": len(self.missing_slots()),
            "заметок": len(self.scratch),
        }


# ---------------------------------------------------------------------------
# Слой 3: долговременная память
# ---------------------------------------------------------------------------

@dataclass
class LongTermMemory:
    """Профиль, решения, знания. Переживает перезапуск — хранится в файле."""
    storage_path: str = "memory_store.json"
    profile: Dict[str, str] = field(default_factory=dict)
    decisions: List[Dict] = field(default_factory=list)
    knowledge: Dict[str, str] = field(default_factory=dict)

    def load(self) -> None:
        if not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.profile = data.get("profile", {})
            self.decisions = data.get("decisions", [])
            self.knowledge = data.get("knowledge", {})
        except (json.JSONDecodeError, OSError):
            pass  # битый или недоступный файл — начинаем с пустой памяти

    def save(self) -> None:
        data = {"profile": self.profile, "decisions": self.decisions, "knowledge": self.knowledge}
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def set_profile(self, key: str, value: str) -> None:
        self.profile[key] = value
        self.save()

    def add_decision(self, text: str) -> None:
        self.decisions.append({"text": text, "ts": datetime.now().isoformat(timespec="seconds")})
        self.save()

    def set_knowledge(self, key: str, value: str) -> None:
        self.knowledge[key] = value
        self.save()

    def forget(self, section: Optional[str] = None) -> None:
        if section in (None, "all"):
            self.profile, self.decisions, self.knowledge = {}, [], {}
        elif section == "profile":
            self.profile = {}
        elif section == "decisions":
            self.decisions = []
        elif section == "knowledge":
            self.knowledge = {}
        self.save()

    def render_for_prompt(self) -> str:
        blocks = []
        if self.profile:
            blocks.append("Профиль пользователя:\n" + "\n".join(f"  - {k}: {v}" for k, v in self.profile.items()))
        if self.decisions:
            recent = self.decisions[-10:]
            blocks.append("Ранее принятые решения:\n" + "\n".join(f"  - {d['text']}" for d in recent))
        if self.knowledge:
            blocks.append("Знания о проекте:\n" + "\n".join(f"  - {k}: {v}" for k, v in self.knowledge.items()))
        return "\n\n".join(blocks)

    def stats(self) -> Dict:
        return {
            "профиль (полей)": len(self.profile),
            "решений": len(self.decisions),
            "знаний": len(self.knowledge),
            "файл": self.storage_path,
        }


# ---------------------------------------------------------------------------
# Авто-роутер: LLM решает, в какой слой положить данные
# ---------------------------------------------------------------------------

ROUTER_PROMPT = """Ты — маршрутизатор памяти ассистента. У ассистента три слоя памяти:

- "working" — данные ТЕКУЩЕЙ задачи: параметры, требования, цифры по задаче,
  которую сейчас решают (бюджет проекта, сроки, объём работ, технические параметры).
- "long_profile" — устойчивое о пользователе: имя, роль, компания, стек,
  часовой пояс, постоянные предпочтения.
- "long_decisions" — принятые решения и договорённости ("решили делать X",
  "договорились, что Y").
- "long_knowledge" — факты о предметной области/проекте, которые полезны
  и после завершения текущей задачи.
- "none" — болтовня, уточняющие вопросы, ничего для сохранения.

Текущая задача ассистента: {task}

Сообщение пользователя:
{message}

Верни ТОЛЬКО валидный JSON вида:
{{"items": [{{"layer": "...", "key": "...", "value": "...", "reason": "почему сюда"}}]}}

Для "long_decisions" поле key можно оставить пустым — важен value (текст решения).
Если сохранять нечего — верни {{"items": []}}. Никакого текста кроме JSON."""


def default_router_factory(llm_call: Callable[[str], str]) -> Callable[[str, Optional[str]], List[Dict]]:
    """Оборачивает LLM-вызов в роутер: (сообщение, имя задачи) -> список решений о маршрутизации."""
    def router(message: str, task_name: Optional[str]) -> List[Dict]:
        prompt = ROUTER_PROMPT.format(task=task_name or "(задача не задана)", message=message)
        raw = llm_call(prompt)
        cleaned = (raw or "").strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
        try:
            parsed = json.loads(cleaned)
            items = parsed.get("items", [])
            return items if isinstance(items, list) else []
        except (json.JSONDecodeError, ValueError, AttributeError):
            return []
    return router


# ---------------------------------------------------------------------------
# Менеджер: собирает контекст из трёх слоёв
# ---------------------------------------------------------------------------

class MemoryManager:
    """Объединяет три слоя и собирает из них итоговый контекст запроса."""

    def __init__(self, window_size: int = 6, storage_path: str = "memory_store.json",
                 router: Optional[Callable[[str, Optional[str]], List[Dict]]] = None):
        self.short = ShortTermMemory(window_size=window_size)
        self.working = WorkingMemory()
        self.long = LongTermMemory(storage_path=storage_path)
        self.long.load()
        self.router = router
        self.route_log: List[Dict] = []
        self.router_calls = 0

    # --- маршрутизация ---

    def route_message(self, message: str) -> List[Dict]:
        """Авто-режим: спрашиваем у LLM, что и куда сохранить, и применяем."""
        if self.router is None:
            return []
        items = self.router(message, self.working.task_name)
        self.router_calls += 1
        applied = []
        for item in items:
            layer = item.get("layer")
            key = (item.get("key") or "").strip()
            value = (item.get("value") or "").strip()
            reason = item.get("reason", "")
            if not value or layer in (None, "none"):
                continue
            if layer == "working":
                self.working.set_slot(key or value[:30], value)
            elif layer == "long_profile":
                self.long.set_profile(key or value[:30], value)
            elif layer == "long_decisions":
                self.long.add_decision(value)
            elif layer == "long_knowledge":
                self.long.set_knowledge(key or value[:30], value)
            else:
                continue
            record = {"layer": layer, "key": key, "value": value, "reason": reason,
                       "message": message[:80]}
            applied.append(record)
            self.route_log.append(record)
        return applied

    # --- сборка контекста ---

    def build_context(self, system_prompt: str = "") -> Dict:
        parts = [system_prompt] if system_prompt else []

        long_block = self.long.render_for_prompt()
        if long_block:
            parts.append("=== ДОЛГОВРЕМЕННАЯ ПАМЯТЬ ===\n" + long_block)

        working_block = self.working.render_for_prompt()
        if working_block:
            parts.append("=== РАБОЧАЯ ПАМЯТЬ (текущая задача) ===\n" + working_block)

        return {"system": "\n\n".join(parts), "messages": self.short.get_window()}

    def context_breakdown(self, system_prompt: str = "") -> Dict:
        """Сколько токенов даёт каждый слой — чтобы видеть вклад слоёв в промпт."""
        long_block = self.long.render_for_prompt()
        working_block = self.working.render_for_prompt()
        window = self.short.get_window()
        return {
            "system_prompt": approx_tokens(system_prompt),
            "долговременная": approx_tokens(long_block),
            "рабочая": approx_tokens(working_block),
            "краткосрочная": sum(approx_tokens(m["content"]) for m in window),
        }

    def add_message(self, role: str, content: str) -> None:
        self.short.add(role, content)

    def reset_session(self) -> None:
        """Новая сессия: краткосрочная и рабочая очищаются, долговременная остаётся."""
        self.short.clear()
        self.working.clear()

    def stats(self) -> Dict:
        return {
            "краткосрочная": self.short.stats(),
            "рабочая": self.working.stats(),
            "долговременная": self.long.stats(),
            "вызовов роутера": self.router_calls,
        }
