"""
День 21. Пайплайн индексации документов.

    docs/ ──загрузка──► текст ──chunking──► чанки + метаданные ──эмбеддинги──► index/*.sqlite

Собирает по индексу на каждую стратегию chunking:
    index/index_fixed_<backend>.sqlite      (+ .chunks.json)
    index/index_structure_<backend>.sqlite  (+ .chunks.json)

Запуск:
    python build_index.py                         # эмбеддинги локально, без сети
    python build_index.py --backend aitunnel      # эмбеддинги AITunnel (нужен ключ)
    python build_index.py --size 800 --overlap 100 --max-chars 1500
"""

import argparse
import os
import sys
import time

from chunking import STRATEGIES
from embeddings import make_embedder
from index_store import save_index
from loaders import load_documents, corpus_stats

BASE = os.path.dirname(os.path.abspath(__file__))


def index_path(strategy: str, backend: str) -> str:
    return os.path.join(BASE, "index", f"index_{strategy}_{backend}.sqlite")


def build(args) -> dict:
    docs = load_documents(args.docs)
    if not docs:
        raise SystemExit(f"В папке {args.docs} нет документов (.md, .txt, .py, .pdf)")
    stats = corpus_stats(docs)
    print(f"Корпус: {stats['documents']} документов {stats['by_kind']}, {stats['chars']:,} символов, "
          f"{stats['words']:,} слов ≈ {stats['pages']} стр.".replace(",", " "))

    params = {"fixed": {"size": args.size, "overlap": args.overlap},
              "structure": {"max_chars": args.max_chars, "min_chars": args.min_chars}}
    built = {}
    for strategy, fn in STRATEGIES.items():
        t0 = time.perf_counter()
        chunks = [c for d in docs for c in fn(d, **params[strategy])]
        embedder = make_embedder(args.backend, args.model).fit([c.text for c in chunks])
        vectors = embedder.embed([c.text for c in chunks])
        path = index_path(strategy, args.backend)
        save_index(path, chunks, vectors, {"strategy": strategy, "params": params[strategy],
                                           "embedder": embedder.state(), "corpus": stats})
        sec = time.perf_counter() - t0
        print(f"  ✅ {strategy:<9} чанков: {len(chunks):>4}, вектор: {vectors.shape[1]}, "
              f"эмбеддинги: {embedder.label}, {sec:.1f} с → {os.path.relpath(path, BASE)}")
        built[strategy] = path
    return built


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Индексация документов: chunking → эмбеддинги → SQLite")
    p.add_argument("--docs", default=os.path.join(BASE, "docs"), help="папка с документами")
    p.add_argument("--backend", choices=["local", "aitunnel"], default="local")
    p.add_argument("--model", help="модель эмбеддингов для aitunnel (по умолчанию text-embedding-3-small)")
    p.add_argument("--size", type=int, default=1000, help="fixed: размер чанка в символах")
    p.add_argument("--overlap", type=int, default=150, help="fixed: перекрытие в символах")
    p.add_argument("--max-chars", type=int, default=1500, help="structure: максимальный размер чанка")
    p.add_argument("--min-chars", type=int, default=200, help="structure: минимальный размер раздела")
    build(p.parse_args())
