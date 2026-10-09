"""
Поиск по индексу — чтобы своими глазами увидеть, что находит каждая стратегия.

    python search.py "Как защититься от записи за пределы папки output?"
    python search.py "sha256 пайплайн" --strategy structure --k 5
    python search.py "вопрос" --backend aitunnel
"""

import argparse
import sys

from build_index import index_path
from index_store import Index


def main(args) -> None:
    strategies = ("fixed", "structure") if args.strategy == "both" else (args.strategy,)
    print(f"Запрос: {args.query}")
    for s in strategies:
        idx = Index(index_path(s, args.backend))
        print(f"\n=== {s} ({idx.meta['chunks']} чанков, {idx.meta['embedder']['label']}) ===")
        for i, r in enumerate(idx.search(args.query, args.k), 1):
            preview = " ".join(r["text"].split())[:args.preview]
            print(f"{i}. score={r['score']:.3f}  {r['chunk_id']}")
            print(f"   файл: {r['source']}  |  раздел: {r['section']}  |  {r['n_chars']} симв., "
                  f"позиция {r['char_start']}–{r['char_end']}")
            print(f"   {preview}…")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Поиск по индексу документов")
    p.add_argument("query")
    p.add_argument("--strategy", choices=["fixed", "structure", "both"], default="both")
    p.add_argument("--backend", choices=["local", "aitunnel"], default="local")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--preview", type=int, default=160)
    main(p.parse_args())
