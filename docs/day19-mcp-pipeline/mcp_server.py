"""
День 19. MCP-сервер с тремя инструментами для пайплайна.

  search       — получает данные (статьи из базы знаний)
  summarize    — обрабатывает данные (выжимка)
  save_to_file — сохраняет результат (Markdown-файл в папке output/)

Логика инструментов — в tools_impl.py, здесь только регистрация
и описание параметров. Транспорт stdio: сервер запускает pipeline.py.
"""

import sys
from typing import Annotated, Literal

from pydantic import Field
from mcp.server.fastmcp import FastMCP

import tools_impl

mcp = FastMCP("pipeline-server")


def log(message: str) -> None:
    print(f"[pipeline-server] {message}", file=sys.stderr, flush=True)  # stdout занят протоколом


@mcp.tool()
def search(
    query: Annotated[str, Field(description="Поисковый запрос, например 'композиция MCP инструментов'")],
    top_k: Annotated[int, Field(description="Сколько статей вернуть (1–10)", ge=1, le=10)] = 3,
) -> dict:
    """Найти статьи в базе знаний. Возвращает документы и combined_text — готовый текст для summarize."""
    log(f"search(query={query!r}, top_k={top_k})")
    return tools_impl.search(query, top_k)


@mcp.tool()
def summarize(
    text: Annotated[str, Field(description="Текст для сжатия (например, combined_text из search)")],
    max_sentences: Annotated[int, Field(description="Сколько предложений оставить (1–20)", ge=1, le=20)] = 4,
    mode: Annotated[
        Literal["extractive", "llm"],
        Field(description="extractive — локальная выжимка без сети; llm — через AITunnel (нужен ключ)"),
    ] = "extractive",
) -> dict:
    """Сделать краткую выжимку текста. Возвращает summary и sha256 полученного текста."""
    log(f"summarize(len(text)={len(text)}, max_sentences={max_sentences}, mode={mode})")
    return tools_impl.summarize(text, max_sentences, mode)


@mcp.tool()
def save_to_file(
    filename: Annotated[str, Field(description="Имя файла без пути, например 'report.md'")],
    content: Annotated[str, Field(description="Текст для сохранения")],
) -> dict:
    """Сохранить текст в папку output/. Возвращает путь, размер и sha256 записанного файла."""
    log(f"save_to_file(filename={filename!r}, len(content)={len(content)})")
    return tools_impl.save_to_file(filename, content)


if __name__ == "__main__":
    log("запущен, жду клиента на stdio")
    mcp.run()
