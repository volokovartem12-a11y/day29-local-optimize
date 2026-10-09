"""Задачи менеджера (mock-трекер): логика для MCP-сервера tasks."""

from copy import deepcopy
from datetime import date

_TASKS = []


def create_task(title: str, client_id: int = None, due_date: str = None) -> dict:
    """Создать открытую задачу. due_date — в формате ГГГГ-ММ-ДД."""
    if not title.strip():
        raise ValueError("Название задачи не может быть пустым")
    if due_date:
        try:
            date.fromisoformat(due_date)
        except ValueError as e:
            raise ValueError(f"Дата '{due_date}' не в формате ГГГГ-ММ-ДД") from e
    task = {"id": len(_TASKS) + 1, "title": title.strip(), "client_id": client_id,
            "due_date": due_date, "status": "open", "created": date.today().isoformat()}
    _TASKS.append(task)
    return {"created": True, "task": deepcopy(task)}


def list_tasks(client_id: int = None, status: str = "open") -> dict:
    """Список задач с фильтром по клиенту и статусу (open / done / all)."""
    if status not in ("open", "done", "all"):
        raise ValueError("status должен быть open, done или all")
    tasks = [deepcopy(t) for t in _TASKS
             if (client_id is None or t["client_id"] == client_id) and (status == "all" or t["status"] == status)]
    return {"count": len(tasks), "tasks": tasks}


def complete_task(task_id: int) -> dict:
    for t in _TASKS:
        if t["id"] == task_id:
            t["status"] = "done"
            return {"completed": True, "task": deepcopy(t)}
    raise ValueError(f"Задача с id={task_id} не найдена")
