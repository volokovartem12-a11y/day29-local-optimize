"""
День 16. Подключение к MCP и получение списка инструментов (официальный MCP SDK).

По умолчанию подключается к публичному MCP-серверу DeepWiki —
он работает без ключей и регистрации:
    https://mcp.deepwiki.com/mcp   (транспорт Streamable HTTP)

Что делает скрипт:
  1) устанавливает MCP-соединение (initialize);
  2) печатает, к какому серверу подключились (имя, версия, версия протокола);
  3) запрашивает tools/list и выводит список инструментов с описанием
     и входными параметрами.

Два варианта подключения:
  - удалённый (по умолчанию) — публичный сервер DeepWiki по интернету;
  - локальный (--local)      — скрипт сам запускает local_server.py и
                               подключается к нему по stdio, интернет не нужен.

Запуск:
    python list_tools.py              # удалённый сервер DeepWiki
    python list_tools.py --local      # свой локальный сервер
    python list_tools.py --url <адрес SSE-сервера> --transport sse
    python list_tools.py --url <любой другой публичный MCP>
"""

import argparse
import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
try:
    from mcp.client.streamable_http import streamablehttp_client
except ImportError:  # в некоторых версиях SDK функция называется иначе
    from mcp.client.streamable_http import streamable_http_client as streamablehttp_client

DEFAULT_URL = "https://mcp.deepwiki.com/mcp"
LOCAL_SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_server.py")


def root_cause(exc: BaseException) -> BaseException:
    """SDK заворачивает ошибки в ExceptionGroup — достаём настоящую причину."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


def print_tools(tools) -> None:
    print(f"\nДоступно инструментов: {len(tools)}\n")
    for i, tool in enumerate(tools, 1):
        print(f"{i}. {tool.name}")
        description = (tool.description or "(без описания)").strip().splitlines()[0]
        print(f"   Описание: {description}")
        schema = tool.inputSchema or {}
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        if props:
            print("   Параметры:")
            for pname, pinfo in props.items():
                ptype = pinfo.get("type") or " | ".join(
                    str(x.get("type")) for x in pinfo.get("anyOf", [])) or "?"
                mark = " (обязательный)" if pname in required else ""
                print(f"     - {pname}: {ptype}{mark}")
        else:
            print("   Параметры: нет")
        print()


async def run_session(read, write) -> None:
    async with ClientSession(read, write) as session:
        init = await session.initialize()
        info = init.serverInfo
        print("Соединение установлено ✅")
        print(f"Сервер: {info.name} {info.version}")
        print(f"Версия протокола MCP: {init.protocolVersion}")

        result = await session.list_tools()
        print_tools(result.tools)


async def main(url: str, transport: str, local: bool) -> None:
    if local:
        print("Запускаю локальный MCP-сервер (local_server.py) и подключаюсь по stdio ...")
        params = StdioServerParameters(command=sys.executable, args=[LOCAL_SERVER],
                                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        async with stdio_client(params) as (read, write):
            await run_session(read, write)
        return

    print(f"Подключаюсь к {url} (транспорт: {transport}) ...")
    if transport == "sse":
        async with sse_client(url) as (read, write):
            await run_session(read, write)
    else:
        async with streamablehttp_client(url) as (read, write, _get_session_id):
            await run_session(read, write)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Подключиться к MCP и вывести список инструментов")
    parser.add_argument("--url", default=DEFAULT_URL, help="адрес MCP-сервера")
    parser.add_argument("--transport", choices=["http", "sse"], default="http",
                        help="http = Streamable HTTP (по умолчанию), sse = старый SSE-транспорт")
    parser.add_argument("--local", action="store_true",
                        help="поднять свой локальный MCP-сервер и подключиться к нему (без интернета)")
    args = parser.parse_args()

    try:
        asyncio.run(main(args.url, args.transport, args.local))
    except Exception as e:  # noqa: BLE001 — показываем понятную ошибку вместо длинного трейсбека
        cause = root_cause(e)
        print(f"\nНе удалось подключиться к MCP ❌\n{type(cause).__name__}: {cause}")
        if not args.local:
            print("Проверьте интернет и адрес сервера. Запасные варианты:")
            print("  python list_tools.py --local     (свой сервер, без интернета)")
            print("  python list_tools_raw.py         (без SDK)")
        sys.exit(1)
