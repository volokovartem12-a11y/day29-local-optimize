"""
День 16. То же самое, но БЕЗ SDK — голым JSON-RPC поверх HTTP.

Зачем: показать, что происходит «под капотом» MCP, и иметь запасной
вариант, если SDK почему-то не ставится. Нужен только стандартный Python,
никаких pip install.

Протокол (транспорт Streamable HTTP) — три шага:
  1) POST initialize                  → сервер отвечает своими данными
                                         и (обычно) выдаёт Mcp-Session-Id
  2) POST notifications/initialized   → говорим серверу, что готовы
  3) POST tools/list                  → получаем список инструментов

Сервер может ответить либо обычным JSON, либо потоком SSE
("event: message / data: {...}") — обрабатываются оба варианта.

Запуск:
    python list_tools_raw.py
    python list_tools_raw.py --url <другой MCP-сервер>
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

DEFAULT_URL = "https://mcp.deepwiki.com/mcp"
PROTOCOL_VERSION = "2025-06-18"


class McpHttpClient:
    def __init__(self, url: str):
        self.url = url
        self.session_id = None
        self.protocol_version = PROTOCOL_VERSION
        self._next_id = 1

    def _post(self, payload: dict):
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": self.protocol_version,
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id

        req = urllib.request.Request(
            self.url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            sid = resp.headers.get("Mcp-Session-Id")
            if sid:
                self.session_id = sid
            content_type = resp.headers.get("Content-Type", "")
            if "text/event-stream" in content_type:
                # читаем поток построчно и выходим, как только пришёл наш ответ —
                # не ждём, пока сервер закроет соединение
                return self._read_sse(resp, payload.get("id"))
            body = resp.read().decode("utf-8")

        if not body.strip():
            return None  # у уведомлений ответа нет (обычно HTTP 202)
        return json.loads(body)

    @staticmethod
    def _read_sse(resp, request_id):
        """Достаём из SSE-потока JSON-RPC ответ на наш запрос."""
        data_lines = []
        for raw_line in resp:
            line = raw_line.decode("utf-8").rstrip("\r\n")
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())
            elif line == "" and data_lines:  # пустая строка = конец события
                message = json.loads("\n".join(data_lines))
                data_lines = []
                if message.get("id") == request_id:
                    return message
        if data_lines:  # поток закрылся без финальной пустой строки
            message = json.loads("\n".join(data_lines))
            if message.get("id") == request_id:
                return message
        raise RuntimeError("В SSE-ответе не нашлось ответа на запрос")

    def request(self, method: str, params: dict = None) -> dict:
        req_id = self._next_id
        self._next_id += 1
        message = self._post({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}})
        if message is None:
            raise RuntimeError(f"Пустой ответ на {method}")
        if "error" in message:
            raise RuntimeError(f"Сервер вернул ошибку на {method}: {message['error']}")
        return message["result"]

    def notify(self, method: str, params: dict = None) -> None:
        self._post({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def initialize(self) -> dict:
        result = self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "day16-raw-client", "version": "1.0"},
        })
        # дальше общаемся на той версии протокола, которую выбрал сервер
        self.protocol_version = result.get("protocolVersion", PROTOCOL_VERSION)
        self.notify("notifications/initialized")
        return result

    def list_tools(self) -> list:
        tools, cursor = [], None
        while True:  # список может приходить страницами
            params = {"cursor": cursor} if cursor else {}
            result = self.request("tools/list", params)
            tools.extend(result.get("tools", []))
            cursor = result.get("nextCursor")
            if not cursor:
                return tools


def main():
    # чтобы ✅ и кириллица не роняли скрипт при выводе в файл на Windows
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="MCP tools/list без SDK")
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()

    client = McpHttpClient(args.url)
    print(f"Подключаюсь к {args.url} (без SDK, JSON-RPC over HTTP) ...")

    try:
        init = client.initialize()
        info = init.get("serverInfo", {})
        print("Соединение установлено ✅")
        print(f"Сервер: {info.get('name', '?')} {info.get('version', '')}")
        print(f"Версия протокола MCP: {client.protocol_version}")
        print(f"Session ID: {client.session_id or '(сервер не выдал)'}")

        tools = client.list_tools()
    except (urllib.error.URLError, RuntimeError, json.JSONDecodeError) as e:
        print(f"\nНе удалось получить список инструментов ❌\n{type(e).__name__}: {e}")
        sys.exit(1)

    print(f"\nДоступно инструментов: {len(tools)}\n")
    for i, tool in enumerate(tools, 1):
        desc = (tool.get("description") or "(без описания)").strip().splitlines()[0]
        props = tool.get("inputSchema", {}).get("properties", {})
        print(f"{i}. {tool['name']}")
        print(f"   Описание: {desc}")
        print(f"   Параметры: {', '.join(props) if props else 'нет'}\n")


if __name__ == "__main__":
    main()
