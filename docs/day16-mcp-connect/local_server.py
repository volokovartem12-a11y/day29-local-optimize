"""
День 16. Локальный MCP-сервер — вариант «поднять MCP-сервер у себя».

Минимальный сервер на официальном SDK с тремя инструментами.
Работает без интернета: клиент list_tools.py --local сам запускает
этот файл как дочерний процесс и общается с ним через stdin/stdout.

stdout занят протоколом MCP, поэтому здесь ничего не печатаем в stdout.
"""

from datetime import datetime

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("local-demo-server")


@mcp.tool()
def add(a: float, b: float) -> float:
    """Сложить два числа."""
    return a + b


@mcp.tool()
def echo(text: str) -> str:
    """Вернуть переданный текст без изменений."""
    return text


@mcp.tool()
def current_time() -> str:
    """Текущие дата и время на компьютере, где запущен сервер."""
    return datetime.now().isoformat(timespec="seconds")


if __name__ == "__main__":
    mcp.run()  # транспорт stdio
