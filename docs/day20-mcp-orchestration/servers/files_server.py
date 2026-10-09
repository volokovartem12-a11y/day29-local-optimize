"""MCP-сервер files: сохранение отчётов и поиск по ним."""
import sys
from typing import Annotated

from pydantic import Field
from mcp.server.fastmcp import FastMCP

import files_impl

mcp = FastMCP("files")


@mcp.tool()
def save_to_file(
    filename: Annotated[str, Field(description="Имя файла без пути, например 'client_2_report.md'")],
    content: Annotated[str, Field(description="Текст отчёта в Markdown")],
) -> dict:
    """Сохранить отчёт в папку output/. Возвращает имя, размер и sha256 файла."""
    print(f"[files] save_to_file({filename!r}, len={len(content)})", file=sys.stderr, flush=True)
    return files_impl.save_to_file(filename, content)


@mcp.tool()
def search(query: Annotated[str, Field(description="Текст для поиска в именах и содержимом файлов")]) -> dict:
    """Поиск по СОХРАНЁННЫМ ФАЙЛАМ-отчётам. Не ищет в базе знаний."""
    print(f"[files] search({query!r})", file=sys.stderr, flush=True)
    return files_impl.search(query)


@mcp.tool()
def list_files() -> dict:
    """Список сохранённых файлов."""
    return files_impl.list_files()


@mcp.tool()
def read_file(filename: Annotated[str, Field(description="Имя сохранённого файла")]) -> dict:
    """Прочитать сохранённый файл."""
    return files_impl.read_file(filename)


if __name__ == "__main__":
    mcp.run()
