"""
День 17. Собственный MCP-сервер вокруг CRM API.

Что здесь закрывает ТЗ:
  - РЕГИСТРАЦИЯ инструмента   — декоратор @mcp.tool();
  - ОПИСАНИЕ входных параметров — типы + Annotated[..., Field(description=...)],
    из них SDK сам строит JSON Schema (inputSchema), которую видит клиент;
  - ВОЗВРАТ результата         — функция возвращает dict/list, SDK отдаёт его
    клиенту как JSON. Ошибки бизнес-логики уходят клиенту как isError=True.

Транспорт — stdio: агент сам запускает этот файл как дочерний процесс и
общается с ним через stdin/stdout. Поэтому в stdout здесь писать НЕЛЬЗЯ —
он занят протоколом; логи идут в stderr.

Отдельно сервер можно запустить так (он будет ждать клиента на stdin):
    python mcp_server.py
"""

import sys
from typing import Annotated, Literal, Optional

from pydantic import Field
from mcp.server.fastmcp import FastMCP

import crm_api

mcp = FastMCP("crm-server")


def log(message: str) -> None:
    print(f"[crm-server] {message}", file=sys.stderr, flush=True)


@mcp.tool()
def search_clients(
    query: Annotated[str, Field(description="Часть имени, телефона или города клиента, например 'Иван' или 'Москва'")],
) -> dict:
    """Найти клиентов в CRM по имени, телефону или городу. Возвращает список клиентов с их id."""
    log(f"search_clients(query={query!r})")
    try:
        clients = crm_api.search_clients(query)
    except crm_api.CrmError as e:
        raise ValueError(str(e)) from e
    return {"query": query, "found": len(clients), "clients": clients}


@mcp.tool()
def get_client_orders(
    client_id: Annotated[int, Field(description="id клиента из CRM (можно узнать через search_clients)", ge=1)],
    status: Annotated[
        Optional[Literal["new", "in_progress", "done", "cancelled"]],
        Field(description="Фильтр по статусу заказа. Не указывать — вернуть все заказы"),
    ] = None,
) -> dict:
    """Получить заказы клиента и сводку: количество и сумму без отменённых заказов."""
    log(f"get_client_orders(client_id={client_id}, status={status})")
    try:
        return crm_api.get_client_orders(client_id, status)
    except crm_api.CrmError as e:
        raise ValueError(str(e)) from e


@mcp.tool()
def create_order(
    client_id: Annotated[int, Field(description="id клиента, для которого создаётся заказ", ge=1)],
    item: Annotated[str, Field(description="Название услуги, например 'Замена масла'")],
    amount: Annotated[int, Field(description="Сумма заказа в рублях, больше нуля", gt=0)],
) -> dict:
    """Создать новый заказ для клиента. Заказ получает статус 'new'."""
    log(f"create_order(client_id={client_id}, item={item!r}, amount={amount})")
    try:
        order = crm_api.create_order(client_id, item, amount)
    except crm_api.CrmError as e:
        raise ValueError(str(e)) from e
    return {"created": True, "order": order}


if __name__ == "__main__":
    log("запущен, жду MCP-клиента на stdio")
    mcp.run()  # транспорт по умолчанию — stdio
