"""MCP-сервер kb: база знаний сервиса — поиск статей и выжимка."""
import sys
from typing import Annotated

from pydantic import Field
from mcp.server.fastmcp import FastMCP

import kb_impl

mcp = FastMCP("kb")


@mcp.tool()
def search(
    query: Annotated[str, Field(description="Тема, например 'шиномонтаж' или 'замена масла'")],
    top_k: Annotated[int, Field(description="Сколько статей вернуть (1–10)", ge=1, le=10)] = 2,
) -> dict:
    """Поиск в БАЗЕ ЗНАНИЙ сервиса: регламенты и советы по услугам. Не ищет сохранённые файлы.
    Возвращает статьи и combined_text — текст для summarize."""
    print(f"[kb] search({query!r}, {top_k})", file=sys.stderr, flush=True)
    return kb_impl.search(query, top_k)


@mcp.tool()
def summarize(
    text: Annotated[str, Field(description="Текст для выжимки, например combined_text из search")],
    max_sentences: Annotated[int, Field(description="Длина выжимки в предложениях (1–20)", ge=1, le=20)] = 3,
) -> dict:
    """Сделать краткую выжимку текста (ключевые предложения)."""
    print(f"[kb] summarize(len={len(text)}, {max_sentences})", file=sys.stderr, flush=True)
    return kb_impl.summarize(text, max_sentences)


if __name__ == "__main__":
    mcp.run()
