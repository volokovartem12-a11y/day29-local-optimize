"""
Загрузка документов: README / статьи / код / PDF → текст.

Поддерживаются .md, .txt, .py, .pdf. PDF переводится в текст постранично,
каждая страница помечается заголовком «## Страница N», чтобы структурный
chunking мог резать PDF по страницам.
"""

import os
import re
from dataclasses import dataclass

SUPPORTED = (".md", ".txt", ".py", ".pdf")
CHARS_PER_PAGE = 1800  # условная страница машинописного текста


@dataclass
class Document:
    source: str   # путь относительно папки docs/, например day9-context-compression/README.md
    file: str     # имя файла
    kind: str     # markdown | python | text | pdf
    title: str
    text: str


def _read_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError("Для PDF установите пакет: pip install pypdf") from e
    pages = []
    for i, page in enumerate(PdfReader(path).pages, 1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"## Страница {i}\n\n{text}")
    return "\n\n".join(pages)


def _title(kind: str, text: str, file: str) -> str:
    if kind == "markdown":
        m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        if m:
            return m.group(1).strip()
    if kind == "python":
        m = re.match(r'\s*(?:"""|\'\'\')\s*\n?\s*(.+)', text)
        if m:
            return m.group(1).strip().rstrip(".")
    if kind == "pdf":
        for line in text.splitlines():
            if line.strip() and not line.startswith("## Страница"):
                return line.strip()[:120]
    return file


def load_documents(root: str) -> list:
    docs = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith((".", "__")))
        for name in sorted(filenames):
            ext = os.path.splitext(name)[1].lower()
            if ext not in SUPPORTED:
                continue
            path = os.path.join(dirpath, name)
            if ext == ".pdf":
                text = _read_pdf(path)
            else:
                with open(path, encoding="utf-8", errors="replace") as f:
                    text = f.read()
            text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
            if not text:
                continue
            kind = {".md": "markdown", ".py": "python", ".pdf": "pdf"}.get(ext, "text")
            source = os.path.relpath(path, root).replace("\\", "/")
            docs.append(Document(source, name, kind, _title(kind, text, name), text))
    return docs


def corpus_stats(docs: list) -> dict:
    chars = sum(len(d.text) for d in docs)
    return {
        "documents": len(docs),
        "by_kind": {k: sum(1 for d in docs if d.kind == k) for k in sorted({d.kind for d in docs})},
        "chars": chars,
        "words": sum(len(d.text.split()) for d in docs),
        "pages": round(chars / CHARS_PER_PAGE, 1),
    }
