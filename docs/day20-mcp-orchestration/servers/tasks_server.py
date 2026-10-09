"""MCP-сервер tasks: задачи менеджера."""
import sys
from typing import Annotated, Literal, Optional

from pydantic import Field
from mcp.server.fastmcp import FastMCP

import tasks_impl

mcp = FastMCP("tasks")


@mcp.tool()
def create_task(
    title: Annotated[str, Field(description="Что сделать, например 'Позвонить клиенту и напомнить о записи'")],
    client_id: Annotated[Optional[int], Field(description="id клиента из CRM, если задача про клиента")] = None,
    due_date: Annotated[Optional[str], Field(description="Срок в формате ГГГГ-ММ-ДД")] = None,
) -> dict:
    """Создать задачу для менеджера."""
    print(f"[tasks] create_task({title!r}, {client_id}, {due_date})", file=sys.stderr, flush=True)
    return tasks_impl.create_task(title, client_id, due_date)


@mcp.tool()
def list_tasks(
    client_id: Annotated[Optional[int], Field(description="Только задачи этого клиента")] = None,
    status: Annotated[Literal["open", "done", "all"], Field(description="Статус задач")] = "open",
) -> dict:
    """Список задач менеджера."""
    print(f"[tasks] list_tasks({client_id}, {status})", file=sys.stderr, flush=True)
    return tasks_impl.list_tasks(client_id, status)


@mcp.tool()
def complete_task(task_id: Annotated[int, Field(description="id задачи", ge=1)]) -> dict:
    """Отметить задачу выполненной."""
    return tasks_impl.complete_task(task_id)


if __name__ == "__main__":
    mcp.run()
