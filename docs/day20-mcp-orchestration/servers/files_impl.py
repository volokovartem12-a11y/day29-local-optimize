"""Файловое хранилище отчётов: логика для MCP-сервера files."""

import hashlib
import os
import re

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_filename(filename: str) -> str:
    """Только имя файла, без путей: запись возможна лишь внутрь output/."""
    name = os.path.basename(filename.strip().replace("\\", "/"))
    name = re.sub(r"[^\w.\-]", "_", name)
    if not name or name.startswith("."):
        raise ValueError("Некорректное имя файла")
    if not name.lower().endswith((".md", ".txt")):
        name += ".md"
    return name


def save_to_file(filename: str, content: str) -> dict:
    if not content.strip():
        raise ValueError("Пустое содержимое — сохранять нечего")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    name = _safe_filename(filename)
    path = os.path.join(OUTPUT_DIR, name)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    with open(path, "r", encoding="utf-8", newline="") as f:
        written = f.read()
    return {"saved": True, "filename": name, "path": path,
            "bytes": len(written.encode("utf-8")), "sha256": sha256(written)}


def list_files() -> dict:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    names = sorted(n for n in os.listdir(OUTPUT_DIR) if n.lower().endswith((".md", ".txt")))
    return {"count": len(names), "files": names}


def read_file(filename: str) -> dict:
    path = os.path.join(OUTPUT_DIR, _safe_filename(filename))
    if not os.path.exists(path):
        raise ValueError(f"Файл '{filename}' не найден")
    with open(path, encoding="utf-8") as f:
        content = f.read()
    return {"filename": os.path.basename(path), "content": content, "sha256": sha256(content)}


def search(query: str) -> dict:
    """Поиск по СОХРАНЁННЫМ файлам: по имени и содержимому (без учёта регистра)."""
    q = query.strip().lower()
    if not q:
        raise ValueError("Пустой поисковый запрос")
    matches = []
    for name in list_files()["files"]:
        with open(os.path.join(OUTPUT_DIR, name), encoding="utf-8") as f:
            content = f.read()
        pos = content.lower().find(q)
        if q in name.lower() or pos >= 0:
            snippet = content[max(0, pos - 40): pos + 80].replace("\n", " ") if pos >= 0 else content[:120]
            matches.append({"filename": name, "snippet": snippet.strip()})
    return {"query": query, "found": len(matches), "files": matches}
