"""
Хранение индекса в SQLite (один файл на стратегию) + JSON-экспорт метаданных.

Таблица chunks: chunk_id, strategy, source, file, title, section, position,
char_start, char_end, n_chars, text, embedding (float32 BLOB).
Таблица meta: стратегия, параметры, состояние эмбеддера, дата сборки.
"""

import json
import os
import sqlite3
from dataclasses import asdict
from datetime import datetime

import numpy as np

from embeddings import embedder_from_state

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE chunks (
    chunk_id   TEXT PRIMARY KEY,
    strategy   TEXT NOT NULL,
    source     TEXT NOT NULL,
    file       TEXT NOT NULL,
    title      TEXT NOT NULL,
    section    TEXT NOT NULL,
    position   INTEGER NOT NULL,
    char_start INTEGER NOT NULL,
    char_end   INTEGER NOT NULL,
    n_chars    INTEGER NOT NULL,
    text       TEXT NOT NULL,
    embedding  BLOB NOT NULL
);
CREATE INDEX idx_chunks_source ON chunks(source);
"""

META_FIELDS = ("chunk_id", "strategy", "source", "file", "title", "section", "position", "char_start", "char_end")


def save_index(path: str, chunks: list, vectors: np.ndarray, meta: dict) -> None:
    if len(chunks) != len(vectors):
        raise ValueError("Число чанков и векторов не совпадает")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        os.remove(path)  # индекс пересобирается целиком
    con = sqlite3.connect(path)
    try:
        con.executescript(SCHEMA)
        meta = {**meta, "created": datetime.now().isoformat(timespec="seconds"),
                "chunks": len(chunks), "dim": int(vectors.shape[1])}
        con.executemany("INSERT INTO meta VALUES (?, ?)", [(k, json.dumps(v, ensure_ascii=False)) for k, v in meta.items()])
        con.executemany(
            "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [(c.chunk_id, c.strategy, c.source, c.file, c.title, c.section, c.position, c.char_start,
              c.char_end, len(c.text), c.text, v.astype(np.float32).tobytes()) for c, v in zip(chunks, vectors)])
        con.commit()
    finally:
        con.close()

    with open(os.path.splitext(path)[0] + ".chunks.json", "w", encoding="utf-8") as f:
        json.dump({"meta": {k: v for k, v in meta.items() if k != "embedder"},
                   "embedder": meta["embedder"]["label"],
                   "chunks": [{**{k: asdict(c)[k] for k in META_FIELDS}, "n_chars": len(c.text),
                               "preview": c.text[:160]} for c in chunks]},
                  f, ensure_ascii=False, indent=2)


class Index:
    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Индекс не найден: {path}. Сначала соберите его: python build_index.py")
        con = sqlite3.connect(path)
        try:
            self.meta = {k: json.loads(v) for k, v in con.execute("SELECT key, value FROM meta")}
            rows = con.execute("SELECT * FROM chunks ORDER BY source, position").fetchall()
            cols = [d[0] for d in con.execute("SELECT * FROM chunks LIMIT 0").description]
        finally:
            con.close()
        self.chunks = [dict(zip(cols, r)) for r in rows]
        dim = self.meta["dim"]
        self.matrix = np.stack([np.frombuffer(c.pop("embedding"), dtype=np.float32, count=dim) for c in self.chunks])
        self.embedder = embedder_from_state(self.meta["embedder"])

    def search(self, query: str, k: int = 3) -> list:
        q = self.embedder.embed([query])[0]
        scores = self.matrix @ q
        top = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]
