"""
День 17. Прямой вызов MCP-инструментов из приложения — без LLM.

Проверяет серверную часть ТЗ, не требуя ключа AITunnel:
  1) подключение к своему MCP-серверу;
  2) список зарегистрированных инструментов и их входные параметры
     (то, что сервер описал через JSON Schema);
  3) вызов инструментов и получение результата;
  4) использование результата в коде: id из первого вызова передаётся
     во второй — цепочка search_clients -> get_client_orders;
  5) обработка ошибки: заведомо неверный вызов возвращает isError.

Запуск:
    python demo_call.py
"""

import asyncio
import json
import sys

from mcp_client import connect, pretty, result_to_text


def show_schema(tool) -> None:
    schema = tool.inputSchema or {}
    required = set(schema.get("required", []))
    print(f"• {tool.name} — {tool.description}")
    for name, info in schema.get("properties", {}).items():
        typ = info.get("type") or " | ".join(
            str(x.get("type") or x.get("enum")) for x in info.get("anyOf", [])) or "?"
        req = "обязательный" if name in required else "необязательный"
        print(f"    - {name} ({typ}, {req}): {info.get('description', '')}")


async def main() -> None:
    print("Запускаю свой MCP-сервер (mcp_server.py) и подключаюсь по stdio ...")
    async with connect() as session:
        print("Соединение установлено ✅\n")

        tools = (await session.list_tools()).tools
        print(f"=== Зарегистрировано инструментов: {len(tools)} ===")
        for t in tools:
            show_schema(t)

        print("\n=== Вызов 1: search_clients(query='Иван') ===")
        r1 = await session.call_tool("search_clients", {"query": "Иван"})
        text1 = result_to_text(r1)
        print(pretty(text1))

        # используем результат в коде: берём id первого найденного клиента
        data1 = json.loads(text1)
        if not data1.get("clients"):
            print("Клиенты не найдены — дальше идти не с чем.")
            return
        client = data1["clients"][0]
        print(f"\n→ Из результата взят client_id={client['id']} ({client['name']})")

        print(f"\n=== Вызов 2: get_client_orders(client_id={client['id']}) ===")
        r2 = await session.call_tool("get_client_orders", {"client_id": client["id"]})
        data2 = json.loads(result_to_text(r2))
        for o in data2["orders"]:
            print(f"  #{o['id']}  {o['item']:<24} {o['amount']:>7} ₽  [{o['status']}]")
        print(f"→ Итог по клиенту без отменённых: {data2['total_amount_without_cancelled']} ₽")

        print("\n=== Вызов 3 (проверка ошибки): get_client_orders(client_id=999) ===")
        r3 = await session.call_tool("get_client_orders", {"client_id": 999})
        print(f"isError={r3.isError} → {result_to_text(r3)}")

    print("\nГотово: инструменты зарегистрированы, вызываются и возвращают результат ✅")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main())
