"""
Общие функции MCP-клиента для pipeline.py.

- connect()          — запускает mcp_server.py как дочерний процесс и
                        открывает MCP-сессию по stdio;
- result_to_text()   — достаёт текст результата вызова инструмента;
- tools_to_openai()  — переводит список MCP-инструментов в формат
                        function calling, который понимает LLM.
"""

import json
import os
import sys
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")


@asynccontextmanager
async def connect():
    """Запускает MCP-сервер тем же Python и отдаёт готовую сессию."""
    params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_PATH],
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


def result_to_text(result) -> str:
    """Текст из результата call_tool. Ошибки инструмента помечаются явно."""
    parts = [c.text for c in (result.content or []) if getattr(c, "type", None) == "text"]
    if parts:
        text = "\n".join(parts)
    elif getattr(result, "structuredContent", None) is not None:
        text = json.dumps(result.structuredContent, ensure_ascii=False)
    else:
        text = "(пустой результат)"
    if getattr(result, "isError", False):
        return f"ОШИБКА ИНСТРУМЕНТА: {text}"
    return text


def pretty(text: str) -> str:
    """Красиво форматирует JSON-ответ (кириллица без \\u-кодов)."""
    try:
        return json.dumps(json.loads(text), ensure_ascii=False, indent=2)
    except (json.JSONDecodeError, TypeError):
        return text


def tools_to_openai(mcp_tools) -> list:
    """MCP tools -> формат tools для chat.completions (function calling)."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema or {"type": "object", "properties": {}},
            },
        }
        for t in mcp_tools
    ]
