"""
Дни 23–24. Подбор порогов по данным, а не на глаз.

Порог «не знаю» (день 24, answer_min): по лучшей близости каждого вопроса
отделяет вопросы с ответом от вопросов без ответа и расплывчатых. Главное
условие — не отказывать там, где ответ есть; затем — поймать как можно больше
вопросов без ответа; порог ставится посередине зазора между группами.

Порог фильтра (день 23):

Для каждого контрольного вопроса берутся 20 кандидатов. Релевантными считаются
чанки из ожидаемых файлов (поле sources), для вопросов без ответа в базе
(type = absent) — все кандидаты нерелевантны. Общие вопросы в калибровке
не участвуют: для них нет «правильных» файлов.

Перебираются пары (min_score, rel_ratio):
  - обязательное условие: у КАЖДОГО вопроса с ответом остаётся хотя бы один
    релевантный чанк (фильтр не должен убивать ответ);
  - среди таких пар выбирается лучшая по F1 «оставлен ↔ релевантен».

Оговорка: порог подбирается на тех же вопросах, на которых потом идёт
сравнение, — оценка получается оптимистичной. Для честной проверки порог
стоит проверить на новых вопросах.

Запуск:  python calibrate.py --backend aitunnel   (или local)
"""

import argparse
import json
import statistics
import sys

from rag import BASE, load_index
from retrieval import save_thresholds

N_CANDIDATES = 20


def collect(index, questions) -> list:
    data = []
    for q in questions:
        if q["type"] == "general":
            continue
        cands = index.search(q["q"], N_CANDIDATES)
        labels = [q["type"] not in ("absent", "vague") and any(e in c["source"] for e in q["sources"]) for c in cands]
        data.append({"id": q["id"], "type": q["type"], "scores": [c["score"] for c in cands], "labels": labels})
    return data


def evaluate(data, min_score: float, rel_ratio: float) -> dict:
    tp = fp = fn = 0
    answered = answerable = absent_cleared = absent_total = 0
    for d in data:
        best = d["scores"][0]
        kept = [s >= min_score and s >= rel_ratio * best for s in d["scores"]]
        tp += sum(k and l for k, l in zip(kept, d["labels"]))
        fp += sum(k and not l for k, l in zip(kept, d["labels"]))
        fn += sum(not k and l for k, l in zip(kept, d["labels"]))
        if d["type"] in ("absent", "vague"):
            absent_total += 1
            absent_cleared += not any(kept)
        else:
            answerable += 1
            answered += any(k and l for k, l in zip(kept, d["labels"]))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "precision": prec, "recall": rec,
            "questions_kept": f"{answered}/{answerable}", "all_answerable": answered == answerable,
            "absent_cleared": f"{absent_cleared}/{absent_total}"}


def calibrate(index, questions) -> tuple:
    data = collect(index, questions)
    all_scores = sorted({round(s, 3) for d in data for s in d["scores"]})
    grid_min = [all_scores[int(i * (len(all_scores) - 1) / 40)] for i in range(41)]
    grid_rel = [round(0.30 + 0.05 * i, 2) for i in range(14)]  # 0.30 … 0.95
    best = None
    for m in grid_min:
        for r in grid_rel:
            res = evaluate(data, m, r)
            key = (res["all_answerable"], res["f1"], m)  # ответы важнее, затем F1, затем строже порог
            if best is None or key > best[0]:
                best = (key, m, r, res)
    _, m, r, res = best
    return {"min_score": round(m, 3), "rel_ratio": r}, res, data


def calibrate_gate(index, questions) -> tuple:
    """Порог «не знаю» по лучшей близости вопроса."""
    ans = sorted(index.search(q["q"], 1)[0]["score"] for q in questions if q["type"] in ("project", "general"))
    unk = sorted(index.search(q["q"], 1)[0]["score"] for q in questions if q["type"] in ("absent", "vague"))
    points = sorted(set(ans + unk))
    cands = [(a + b) / 2 for a, b in zip(points, points[1:])] + [points[0] - 1e-3]
    def key(t):
        return (sum(s >= t for s in ans), sum(s < t for s in unk), -abs(t - (max(unk) + min(ans)) / 2))
    best = max(cands, key=key)
    stats = {"answerable_kept": f"{sum(s >= best for s in ans)}/{len(ans)}",
             "unknown_caught": f"{sum(s < best for s in unk)}/{len(unk)}",
             "answerable_range": (round(ans[0], 3), round(ans[-1], 3)),
             "unknown_range": (round(unk[0], 3), round(unk[-1], 3))}
    return round(best, 4), stats


def main(args) -> int:
    with open(f"{BASE}/eval_questions.json", encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    index = load_index(args.backend, "structure")
    thresholds, res, data = calibrate(index, questions)

    rel = [s for d in data for s, l in zip(d["scores"], d["labels"]) if l]
    irr = [s for d in data for s, l in zip(d["scores"], d["labels"]) if not l]
    top_ans = [d["scores"][0] for d in data if d["type"] not in ("absent", "vague")]
    top_abs = [d["scores"][0] for d in data if d["type"] in ("absent", "vague")]
    print(f"Эмбеддинги: {index.meta['embedder']['label']}")
    print(f"Близость релевантных чанков:   среднее {statistics.mean(rel):.3f}, медиана {statistics.median(rel):.3f}")
    print(f"Близость нерелевантных чанков: среднее {statistics.mean(irr):.3f}, медиана {statistics.median(irr):.3f}")
    print(f"Лучший кандидат: у вопросов с ответом {min(top_ans):.3f}–{max(top_ans):.3f}, "
          f"у вопросов без ответа {', '.join(f'{s:.3f}' for s in top_abs)}")
    print(f"\nПодобранный порог: min_score = {thresholds['min_score']}, rel_ratio = {thresholds['rel_ratio']}")
    print(f"  F1 {res['f1']:.2f} (точность {res['precision']:.2f}, полнота {res['recall']:.2f}); "
          f"вопросов с ответом сохранено: {res['questions_kept']}; "
          f"вопросов без ответа отсечено полностью: {res['absent_cleared']}")
    gate, gstats = calibrate_gate(index, questions)
    thresholds["answer_min"] = gate
    print(f"\nПорог «не знаю»: answer_min = {gate}")
    print(f"  лучшая близость: у вопросов с ответом {gstats['answerable_range'][0]}–{gstats['answerable_range'][1]}, "
          f"без ответа / расплывчатых {gstats['unknown_range'][0]}–{gstats['unknown_range'][1]}")
    print(f"  вопросов с ответом пропущено к модели: {gstats['answerable_kept']}; "
          f"вопросов без ответа остановлено порогом: {gstats['unknown_caught']}")
    if not args.dry_run:
        save_thresholds(args.backend, thresholds)
        print(f"Сохранено в config.json → thresholds.{args.backend}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Подбор порога отсечения")
    ap.add_argument("--backend", choices=["aitunnel", "local", "ollama"], default="aitunnel")
    ap.add_argument("--dry-run", action="store_true", help="только показать, не сохранять")
    sys.exit(main(ap.parse_args()))
