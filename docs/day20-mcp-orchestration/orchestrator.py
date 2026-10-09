"""
День 20. Оркестратор нескольких MCP-серверов.

1) РЕГИСТРАЦИЯ   — список серверов берётся из servers.json;
2) ПОДКЛЮЧЕНИЕ   — к каждому серверу открывается своя MCP-сессия (stdio);
3) КАТАЛОГ       — инструменты всех серверов собираются в один список под
                   именами «сервер__инструмент» (kb__search, files__search):
                   одноимённые инструменты разных серверов не конфликтуют;
4) МАРШРУТИЗАЦИЯ — вызов «kb__search» уходит именно на сервер kb под
                   исходным именем search;
5) ЖУРНАЛ        — каждый вызов пишется в trace: ход, сервер, инструмент,
                   аргументы, результат, время. По нему потом проверяется
                   корректность выбора и порядка вызовов.
"""

import json
import os
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEP = "__"


@dataclass
class ServerSpec:
    name: str
    script: str
    description: str = ""


@dataclass
class Route:
    server: str
    tool: str
    description: str
    schema: dict


@dataclass
class TraceEntry:
    n: int
    turn: int
    server: str
    tool: str
    args: dict
    ok: bool
    result: str
    ms: int
    data: dict = field(default=None)


def load_registry(path: str = os.path.join(BASE_DIR, "servers.json")) -> list:
    with open(path, encoding="utf-8") as f:
        items = json.load(f)["servers"]
    specs, seen = [], set()
    for item in items:
        spec = ServerSpec(item["name"], item["script"], item.get("description", ""))
        if spec.name in seen:
            raise ValueError(f"Сервер '{spec.name}' зарегистрирован дважды")
        if SEP in spec.name:
            raise ValueError(f"Имя сервера не должно содержать '{SEP}'")
        if not os.path.exists(os.path.join(BASE_DIR, spec.script)):
            raise FileNotFoundError(f"Нет файла сервера {spec.name}: {spec.script}")
        seen.add(spec.name)
        specs.append(spec)
    return specs


async def open_stdio_session(spec: ServerSpec, stack: AsyncExitStack):
    """Запустить MCP-сервер дочерним процессом и открыть сессию."""
    script = os.path.join(BASE_DIR, spec.script)
    if not os.path.exists(script):
        raise FileNotFoundError(f"Нет файла сервера {spec.name}: {script}")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=sys.executable, args=[script],
                                   env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    read, write = await stack.enter_async_context(stdio_client(params))
    session = await stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    return session


def result_to_text(result) -> tuple:
    parts = [c.text for c in (result.content or []) if getattr(c, "type", None) == "text"]
    text = "\n".join(parts) if parts else json.dumps(getattr(result, "structuredContent", None), ensure_ascii=False)
    return text, bool(getattr(result, "isError", False))


class Orchestrator:
    def __init__(self, specs: list, open_session=open_stdio_session):
        self.specs = specs
        self._open_session = open_session
        self.sessions = {}
        self.routes = {}
        self.trace = []
        self.turn = 0

    async def __aenter__(self):
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        try:
            for spec in self.specs:
                session = await self._open_session(spec, self._stack)
                self.sessions[spec.name] = session
                for t in (await session.list_tools()).tools:
                    self.routes[f"{spec.name}{SEP}{t.name}"] = Route(
                        spec.name, t.name, t.description or "", t.inputSchema or {"type": "object", "properties": {}})
        except BaseException:
            await self._stack.__aexit__(*sys.exc_info())
            raise
        return self

    async def __aexit__(self, *exc):
        return await self._stack.__aexit__(*exc)

    def catalog(self) -> dict:
        """Инструменты, сгруппированные по серверам."""
        out = {s.name: [] for s in self.specs}
        for full, r in self.routes.items():
            out[r.server].append((full, r))
        return out

    def llm_tools(self) -> list:
        """Каталог в формате function calling. Описание начинается с имени сервера."""
        return [{"type": "function",
                 "function": {"name": full, "description": f"[сервер {r.server}] {r.description}",
                              "parameters": r.schema}}
                for full, r in self.routes.items()]

    async def call(self, full_name: str, args: dict) -> tuple:
        """Маршрутизировать вызов на сервер-владелец. Возвращает (текст, ошибка?)."""
        n = len(self.trace) + 1
        route = self.routes.get(full_name)
        if route is None:
            text = f"Неизвестный инструмент '{full_name}'. Доступны: {', '.join(self.routes)}"
            self.trace.append(TraceEntry(n, self.turn, "?", full_name, args, False, text, 0))
            return text, True

        started = time.perf_counter()
        try:
            result = await self.sessions[route.server].call_tool(route.tool, args)
            text, is_error = result_to_text(result)
        except Exception as e:  # noqa: BLE001 — сбой транспорта не должен ронять весь флоу
            text, is_error = f"{type(e).__name__}: {e}", True
        ms = int((time.perf_counter() - started) * 1000)

        data = None
        if not is_error:
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                pass
        self.trace.append(TraceEntry(n, self.turn, route.server, route.tool, args, not is_error, text, ms, data))
        return (f"ОШИБКА ИНСТРУМЕНТА: {text}" if is_error else text), is_error
