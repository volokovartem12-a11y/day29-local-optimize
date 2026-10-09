"""MCP-сервер crm: клиенты и заказы."""
import sys
from typing import Annotated, Literal, Optional

from pydantic import Field
from mcp.server.fastmcp import FastMCP

import crm_api

mcp = FastMCP("crm")


def _run(fn, *args):
    print(f"[crm] {fn.__name__}{args}", file=sys.stderr, flush=True)
    try:
        return fn(*args)
    except crm_api.CrmError as e:
        raise ValueError(str(e)) from e


@mcp.tool()
def search_clients(query: Annotated[str, Field(description="Часть имени, телефона или города клиента")]) -> dict:
    """Найти клиентов CRM по имени, телефону или городу. Возвращает клиентов с их id."""
    clients = _run(crm_api.search_clients, query)
    return {"query": query, "found": len(clients), "clients": clients}


@mcp.tool()
def get_client_orders(
    client_id: Annotated[int, Field(description="id клиента из search_clients", ge=1)],
    status: Annotated[Optional[Literal["new", "in_progress", "done", "cancelled"]],
                      Field(description="Фильтр по статусу; не указывать — все заказы")] = None,
) -> dict:
    """Заказы клиента CRM и сумма без отменённых."""
    return _run(crm_api.get_client_orders, client_id, status)


@mcp.tool()
def create_order(
    client_id: Annotated[int, Field(description="id клиента", ge=1)],
    item: Annotated[str, Field(description="Название услуги")],
    amount: Annotated[int, Field(description="Сумма в рублях", gt=0)],
) -> dict:
    """Создать заказ в CRM со статусом new."""
    return {"created": True, "order": _run(crm_api.create_order, client_id, item, amount)}


if __name__ == "__main__":
    mcp.run()
