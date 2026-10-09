"""
День 21. Сравнение двух стратегий chunking по собранным индексам.

Метрики:
  Размер       — число чанков, средний / медианный / мин / макс размер, разброс;
  Границы      — сколько чанков начинаются с начала раздела, сколько обрываются
                 посреди строки, сколько разрывают блок кода ``` в Markdown;
  Метаданные   — у скольких чанков заполнены source / file / title / section;
  Поиск        — по контрольным вопросам (eval_questions.json):
                 hit@1, hit@3 — нужный файл в топе; MRR — средняя обратная
                 позиция; section@3 — найден нужный раздел, а не просто файл;
  Контекст     — сколько символов уйдёт в модель, если брать топ-3 чанка.

Запуск:
    python compare.py                    # индексы с локальными эмбеддингами
    python compare.py --backend aitunnel
"""

import argparse
import json
import os
import statistics
import sys

from build_index import BASE, index_path
from chunking import section_markers
from index_store import Index
from loaders import load_documents

STRATS = ("fixed", "structure")


def size_stats(chunks) -> dict:
    sizes = [c["n_chars"] for c in chunks]
    return {"chunks": len(sizes), "avg": round(statistics.mean(sizes)), "median": round(statistics.median(sizes)),
            "min": min(sizes), "max": max(sizes), "stdev": round(statistics.pstdev(sizes)),
            "tiny_pct": round(100 * sum(s < 200 for s in sizes) / len(sizes), 1)}


def boundary_stats(chunks, docs) -> dict:
    by_src = {d.source: d for d in docs}
    starts = {d.source: {off for off, _ in section_markers(d)} for d in docs}
    at_start = mid_line = broken_fence = md_total = 0
    for c in chunks:
        doc = by_src[c["source"]]
        text = doc.text
        # начало чанка = начало раздела (с точностью до пробелов)
        if any(abs(c["char_start"] - s) <= 2 for s in starts[c["source"]]):
            at_start += 1
        if c["char_end"] < len(text) and text[c["char_end"]] != "\n":
            mid_line += 1
        if doc.kind == "markdown":
            md_total += 1
            broken_fence += c["text"].count("```") % 2
    n = len(chunks)
    return {"starts_at_section_pct": round(100 * at_start / n, 1),
            "cut_mid_line_pct": round(100 * mid_line / n, 1),
            "broken_code_fence_pct": round(100 * broken_fence / md_total, 1) if md_total else 0.0}


def metadata_stats(chunks) -> dict:
    fields = ("chunk_id", "source", "file", "title", "section")
    full = sum(all(str(c[f]).strip() for f in fields) for c in chunks)
    return {"complete_pct": round(100 * full / len(chunks), 1), "fields": list(fields) + ["position", "char_start", "char_end"]}


def retrieval_stats(index, questions, k: int = 5) -> dict:
    hit1 = hit3 = mrr = sec_hits = sec_total = ctx = 0
    details = []
    for item in questions:
        results = index.search(item["q"], k)
        ranks = [i for i, r in enumerate(results, 1) if any(e in r["source"] for e in item["expected"])]
        rank = ranks[0] if ranks else None
        hit1 += rank == 1
        hit3 += bool(rank and rank <= 3)
        mrr += 1 / rank if rank else 0
        sec_ok = None
        if item.get("section"):
            sec_total += 1
            sec_ok = any(any(e in r["source"] for e in item["expected"]) and item["section"] in r["section"]
                         for r in results[:3])
            sec_hits += sec_ok
        ctx += sum(r["n_chars"] for r in results[:3])
        details.append({"q": item["q"], "rank": rank, "section_ok": sec_ok,
                        "top1": f"{results[0]['source']} [{results[0]['section']}]"})
    n = len(questions)
    return {"hit@1": round(hit1 / n, 3), "hit@3": round(hit3 / n, 3), "mrr": round(mrr / n, 3),
            "section@3": round(sec_hits / sec_total, 3) if sec_total else None,
            "context_chars_top3": round(ctx / n), "details": details}


def verdict(r: dict) -> list:
    f, s = r["fixed"], r["structure"]
    lines = []

    def cmp(name, a, b, higher_better=True, unit=""):
        if a == b:
            lines.append(f"- {name}: одинаково ({a}{unit})")
            return
        win = "structure" if (b > a) == higher_better else "fixed"
        lines.append(f"- {name}: лучше **{win}** (fixed {a}{unit} vs structure {b}{unit})")

    cmp("Точность поиска hit@1", f["retrieval"]["hit@1"], s["retrieval"]["hit@1"])
    cmp("Полнота поиска hit@3", f["retrieval"]["hit@3"], s["retrieval"]["hit@3"])
    cmp("MRR", f["retrieval"]["mrr"], s["retrieval"]["mrr"])
    if f["retrieval"]["section@3"] is not None:
        cmp("Нужный раздел в топ-3", f["retrieval"]["section@3"], s["retrieval"]["section@3"])
    cmp("Чанков начинается с начала раздела", f["boundaries"]["starts_at_section_pct"],
        s["boundaries"]["starts_at_section_pct"], unit="%")
    cmp("Обрыв посреди строки", f["boundaries"]["cut_mid_line_pct"], s["boundaries"]["cut_mid_line_pct"],
        higher_better=False, unit="%")
    cmp("Разорванные блоки кода в Markdown", f["boundaries"]["broken_code_fence_pct"],
        s["boundaries"]["broken_code_fence_pct"], higher_better=False, unit="%")
    cmp("Разброс размеров чанков (stdev)", f["size"]["stdev"], s["size"]["stdev"], higher_better=False, unit=" симв.")
    cmp("Контекст для топ-3", f["retrieval"]["context_chars_top3"], s["retrieval"]["context_chars_top3"],
        higher_better=False, unit=" симв.")
    return lines


def render_report(r: dict, backend: str, corpus: dict) -> str:
    f, s = r["fixed"], r["structure"]

    def row(label, a, b):
        return f"| {label} | {a} | {b} |"

    rows = [
        row("Чанков", f["size"]["chunks"], s["size"]["chunks"]),
        row("Размер: средний / медиана", f"{f['size']['avg']} / {f['size']['median']}", f"{s['size']['avg']} / {s['size']['median']}"),
        row("Размер: мин / макс", f"{f['size']['min']} / {f['size']['max']}", f"{s['size']['min']} / {s['size']['max']}"),
        row("Разброс размеров (stdev)", f["size"]["stdev"], s["size"]["stdev"]),
        row("Мелких чанков (< 200 симв.)", f"{f['size']['tiny_pct']}%", f"{s['size']['tiny_pct']}%"),
        row("Начинаются с начала раздела", f"{f['boundaries']['starts_at_section_pct']}%", f"{s['boundaries']['starts_at_section_pct']}%"),
        row("Обрыв посреди строки", f"{f['boundaries']['cut_mid_line_pct']}%", f"{s['boundaries']['cut_mid_line_pct']}%"),
        row("Разорванные блоки ``` (Markdown)", f"{f['boundaries']['broken_code_fence_pct']}%", f"{s['boundaries']['broken_code_fence_pct']}%"),
        row("Метаданные заполнены", f"{f['metadata']['complete_pct']}%", f"{s['metadata']['complete_pct']}%"),
        row("Поиск hit@1", f["retrieval"]["hit@1"], s["retrieval"]["hit@1"]),
        row("Поиск hit@3", f["retrieval"]["hit@3"], s["retrieval"]["hit@3"]),
        row("Поиск MRR", f["retrieval"]["mrr"], s["retrieval"]["mrr"]),
        row("Нужный раздел в топ-3 (section@3)", f["retrieval"]["section@3"], s["retrieval"]["section@3"]),
        row("Контекст топ-3, символов", f["retrieval"]["context_chars_top3"], s["retrieval"]["context_chars_top3"]),
    ]
    q_rows = "\n".join(
        f"| {i} | {fd['q']} | {fd['rank'] or '—'} | {sd['rank'] or '—'} |"
        for i, (fd, sd) in enumerate(zip(f["retrieval"]["details"], s["retrieval"]["details"]), 1))
    return (
        f"# Сравнение стратегий chunking\n\n"
        f"Корпус: {corpus['documents']} документов {corpus['by_kind']}, {corpus['chars']} символов "
        f"≈ {corpus['pages']} страниц. Эмбеддинги: {r['embedder']}.\n\n"
        f"Параметры: fixed — {r['params']['fixed']}; structure — {r['params']['structure']}.\n\n"
        f"## Метрики\n\n| Метрика | fixed | structure |\n|---|---|---|\n" + "\n".join(rows) +
        f"\n\n## Итог\n\n" + "\n".join(verdict(r)) +
        f"\n\n## Позиция правильного файла по каждому вопросу (— = нет в топ-5)\n\n"
        f"| # | Вопрос | fixed | structure |\n|---|---|---|---|\n{q_rows}\n\n"
        f"## Пример: один и тот же файл в двух стратегиях\n\n"
        f"`{r['example']['source']}`\n\n"
        f"- fixed: {' · '.join(r['example']['fixed'])}\n"
        f"- structure: {' · '.join(r['example']['structure'])}\n"
    )


def main(args) -> int:
    with open(os.path.join(BASE, "eval_questions.json"), encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    docs = load_documents(os.path.join(BASE, "docs"))
    indexes = {s: Index(index_path(s, args.backend)) for s in STRATS}

    result = {"embedder": indexes["fixed"].meta["embedder"]["label"],
              "params": {s: indexes[s].meta["params"] for s in STRATS}}
    for s in STRATS:
        idx = indexes[s]
        result[s] = {"size": size_stats(idx.chunks), "boundaries": boundary_stats(idx.chunks, docs),
                     "metadata": metadata_stats(idx.chunks), "retrieval": retrieval_stats(idx, questions)}

    example = "day20-mcp-orchestration/orchestrator.py"
    result["example"] = {"source": example, **{
        s: [f"[{c['section']}] {c['n_chars']}" for c in indexes[s].chunks if c["source"] == example] for s in STRATS}}

    report = render_report(result, args.backend, indexes["fixed"].meta["corpus"])
    os.makedirs(os.path.join(BASE, "output"), exist_ok=True)
    md_path = os.path.join(BASE, "output", f"chunking_comparison_{args.backend}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.splitext(md_path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(report)
    print(f"\nОтчёт сохранён: {os.path.relpath(md_path, BASE)}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Сравнение стратегий chunking")
    p.add_argument("--backend", choices=["local", "aitunnel"], default="local")
    sys.exit(main(p.parse_args()))
