"""
День 28. Сравнение локальной и облачной модели в одном RAG-пайплайне.

    вопрос ─► ПОИСК (локально: bge-m3 в Ollama + фильтр + реранкер) ─► ОДИН И ТОТ ЖЕ контекст
                                                                         ├─► локальная модель (Ollama)
                                                                         └─► облачная модель (AITunnel), если есть ключ

Контекст для обеих моделей одинаковый, поэтому разница в ответах — это разница в моделях,
а не в поиске. Каждый вопрос прогоняется N раз (стабильность). Результаты каждого вызова
сразу пишутся в output/runs.jsonl — прерванный прогон продолжается ключом --resume.

Оцениваются три вещи:
  КАЧЕСТВО     покрытие обязательных фактов (поле must), доля ответов со статусом «answered»,
               нужные источники, ключевые элементы ответа в цитатах, правильный отказ на вопросах
               без ответа; по желанию — LLM-судья (--judge).
  СКОРОСТЬ     время поиска, время ответа, токены/с (у локальной модели — по счётчикам Ollama),
               загрузка модели в память (холодный старт).
  СТАБИЛЬНОСТЬ доля ответов, прошедших проверку цитат с первой попытки; совпадение статуса и
               разброс покрытия/времени между прогонами; сбои; стресс-тест — вопросы без ответа
               при принудительно слабом контексте (отказ по порогу выключен).

Запуск:
    python evaluate_local.py                    # локально + облако (если задан AITUNNEL_API_KEY)
    python evaluate_local.py --no-cloud         # только локальная модель
    python evaluate_local.py --runs 3 --resume  # продолжить прерванный прогон
"""

import argparse
import copy
import json
import os
import statistics as st
import sys
import time
from datetime import datetime

from grounded import generate, normalize, prepare
from local_llm import OllamaLLM, TimedLLM, has_model, ollama_models
from rag import BASE, LLM, load_index
from retrieval import load_config
from verify import judge, lexical_grounding

OUT = os.path.join(BASE, "output")
CLOUD_DEFAULT = "gpt-5.6-luna"


# ---------------------------------------------------------------------------
# Метрики одного ответа
# ---------------------------------------------------------------------------

def must_coverage(answer: str, must: list):
    """Доля групп обязательных фактов, у которых в ответе есть хотя бы один вариант."""
    if not must:
        return None
    hay = normalize(answer)
    hay_ns = hay.replace(" ", "")
    hit = 0
    for group in must:
        for v in group:
            nv = normalize(v)
            if nv in hay or nv.replace(" ", "") in hay_ns:
                hit += 1
                break
    return hit / len(must)


def source_ok(sources: list, expected: list):
    if not expected:
        return None
    return any(e in s for s in sources for e in expected)


def make_record(res, q, gen, model, run, stress, retr_sec, timed, error=None) -> dict:
    sources = sorted({s["source"] for s in res.sources})
    answered = res.status == "answered"
    must = q.get("must") or []
    rec = {
        "run": run, "qid": q["id"], "type": q["type"], "gen": gen, "model": model, "stress": stress,
        "status": "error" if error else res.status, "reason": "error" if error else res.reason,
        "model_called": bool(timed and timed.calls), "answer": res.answer if not error else "",
        "sources": sources, "n_quotes": len(res.quotes), "n_rejected": len(res.rejected_quotes),
        "attempts": res.attempts, "first_try_ok": res.first_try_ok,
        "coverage": must_coverage(res.answer, must) if answered and q["type"] in ("project", "general") else
                    (0.0 if must and q["type"] in ("project", "general") else None),
        "source_ok": source_ok(sources, q.get("sources")) if answered else (False if q.get("sources") else None),
        "grounding": (lexical_grounding(res.answer, res.quotes)["score"] if answered else None),
        "retrieval_sec": round(retr_sec, 3), "top_score": res.top_score, "n_chunks": len(res.chunks),
        "context_sources": sorted({c["source"] for c in res.chunks}),
        "quotes": [{"n": x["n"], "text": x["text"]} for x in res.quotes],
        "error": error, "ts": datetime.now().isoformat(timespec="seconds"),
    }
    rec.update(timed.summary() if timed else {})
    return rec


def key_of(r: dict) -> tuple:
    return (r["run"], r["qid"], r["gen"], r["model"], bool(r["stress"]))


# ---------------------------------------------------------------------------
# Прогон
# ---------------------------------------------------------------------------

def warmup(local: TimedLLM, index) -> dict:
    """Холодный старт: первая загрузка моделей в память. Меряется отдельно, чтобы не портить скорость."""
    out = {}
    t0 = time.perf_counter()
    index.embedder.embed(["прогрев"])
    out["embed_model_load_sec"] = round(time.perf_counter() - t0, 2)
    t0 = time.perf_counter()
    local.llm.chat("Ответь одним словом.", "Привет")
    out["llm_cold_start_sec"] = round(time.perf_counter() - t0, 2)
    local.reset()
    return out


def run_all(args, questions, index, cfg, gens: dict) -> list:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "runs.jsonl")
    done, records = set(), []
    if args.resume and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    records.append(r)
                    done.add(key_of(r))
        print(f"Продолжаю: уже готово {len(records)} записей из {path}")
    else:
        open(path, "w", encoding="utf-8").close()

    total = args.runs * len(gens) * sum(1 + (args.stress and q["type"] in ("absent", "vague")) for q in questions)
    n = 0
    with open(path, "a", encoding="utf-8") as out:
        for run in range(1, args.runs + 1):
            for q in questions:
                variants = [False] + ([True] if args.stress and q["type"] in ("absent", "vague") else [])
                for stress in variants:
                    t0 = time.perf_counter()
                    try:
                        res0, chunks = prepare(index, q["q"], cfg, args.backend, args.mode, None, force=stress)
                        retr_err = None
                    except Exception as e:  # noqa: BLE001
                        res0, chunks, retr_err = None, None, str(e)
                    retr_sec = time.perf_counter() - t0
                    for gname, timed in gens.items():
                        n += 1
                        model = timed.model
                        probe = {"run": run, "qid": q["id"], "gen": gname, "model": model, "stress": stress}
                        if key_of(probe) in done:
                            continue
                        timed.reset()
                        err = retr_err
                        if res0 is None:
                            from grounded import Grounded
                            res = Grounded(q["q"], "unknown", "")
                        else:
                            res = copy.deepcopy(res0)
                            if chunks is not None and not err:
                                try:
                                    generate(timed, index, q["q"], res, chunks)
                                except Exception as e:  # noqa: BLE001
                                    err = str(e)
                        rec = make_record(res, q, gname, model, run, stress, retr_sec, timed, err)
                        records.append(rec)
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        out.flush()
                        tag = "стресс " if stress else ""
                        sec = rec.get("gen_wall_sec")
                        print(f"[{n}/{total}] прогон {run} · вопрос {q['id']:>2} {tag}· {gname:<5} · "
                              f"{rec['status']:<10} покрытие={('%.0f%%' % (rec['coverage'] * 100)) if rec['coverage'] is not None else '-'} "
                              f"· {('%.1f с' % sec) if sec else 'без модели'}"
                              + (f" · ОШИБКА: {err[:80]}" if err else ""), flush=True)
    return records


def add_judge(args, records, questions, judge_llm) -> None:
    by_q = {q["id"]: q for q in questions}
    todo = [r for r in records if r["status"] == "answered" and not r["stress"] and r["run"] == 1
            and "judge" not in r]
    print(f"\nСудья ({args.judge}) проверяет ответы первого прогона: {len(todo)} шт.")
    for r in todo:
        try:
            j = judge(judge_llm, by_q[r["qid"]]["q"], r["answer"], r["quotes"])
            r["judge"] = j["verdict"]
        except Exception as e:  # noqa: BLE001
            r["judge"] = "error"
            print(f"  судья: вопрос {r['qid']} {r['gen']}: {str(e)[:80]}")


# ---------------------------------------------------------------------------
# Сводка
# ---------------------------------------------------------------------------

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def _pct(num, den):
    return round(num / den, 3) if den else None


def summarize(records: list, gen: str) -> dict:
    rs = [r for r in records if r["gen"] == gen]
    normal = [r for r in rs if not r["stress"]]
    ans_q = [r for r in normal if r["type"] in ("project", "general")]
    unans = [r for r in normal if r["type"] in ("absent", "vague")]
    stress = [r for r in rs if r["stress"]]
    called = [r for r in normal if r["model_called"]]
    ok_calls = [r for r in called if r["status"] != "error"]

    q = {
        "answerable_answered": _pct(sum(r["status"] == "answered" for r in ans_q), len(ans_q)),
        "coverage_mean": _mean(r["coverage"] for r in ans_q),
        "coverage_project_only": _mean(r["coverage"] for r in ans_q if r["type"] == "project"),
        "source_ok": _pct(sum(bool(r["source_ok"]) for r in ans_q if r["source_ok"] is not None),
                          len([r for r in ans_q if r["source_ok"] is not None])),
        "grounding_mean": _mean(r["grounding"] for r in ans_q),
        "judge_supported": _pct(sum(r.get("judge") == "supported" for r in ans_q),
                                len([r for r in ans_q if r.get("judge") in ("supported", "partial", "unsupported")])),
        "refusal_correct": _pct(sum(r["status"] != "answered" for r in unans), len(unans)),
        "stress_refusal_correct": _pct(sum(r["status"] not in ("answered",) for r in stress), len(stress)),
        "stress_hallucinated": sum(r["status"] == "answered" for r in stress),
    }

    secs = sorted(r["gen_wall_sec"] for r in ok_calls if r.get("gen_wall_sec"))
    s = {
        "retrieval_sec_mean": _mean(r["retrieval_sec"] for r in normal),
        "answer_sec_mean": _mean(secs),
        "answer_sec_median": round(st.median(secs), 2) if secs else None,
        "answer_sec_p90": round(secs[int(0.9 * (len(secs) - 1))], 2) if secs else None,
        "tok_per_sec_mean": _mean(r.get("tok_per_sec") for r in ok_calls),
        "tok_per_sec_kind": next((r.get("tok_per_sec_kind") for r in ok_calls), None),
        "completion_tokens_mean": _mean(r.get("completion_tokens") for r in ok_calls),
        "prompt_tokens_mean": _mean(r.get("prompt_tokens") for r in ok_calls),
        "model_load_sec_total": round(sum(r.get("load_sec", 0) or 0 for r in ok_calls), 2),
        "prompt_truncated": sum(bool(r.get("truncated_prompt")) for r in ok_calls),
    }

    # стабильность: сравниваем прогоны одного и того же вопроса
    agree, cov_sd, lat_cv = [], [], []
    for qid in sorted({r["qid"] for r in normal}):
        grp = [r for r in normal if r["qid"] == qid]
        if len(grp) < 2:
            continue
        statuses = [r["status"] for r in grp]
        agree.append(max(statuses.count(x) for x in set(statuses)) / len(statuses))
        covs = [r["coverage"] for r in grp if r["coverage"] is not None]
        if len(covs) > 1:
            cov_sd.append(st.pstdev(covs))
        lat = [r["gen_wall_sec"] for r in grp if r.get("gen_wall_sec") and r["status"] != "error"]
        if len(lat) > 1 and st.mean(lat) > 0:
            lat_cv.append(st.pstdev(lat) / st.mean(lat))
    stab = {
        "first_try_ok": _pct(sum(r["first_try_ok"] for r in ok_calls), len(ok_calls)),
        "retry_used": _pct(sum(r["attempts"] == 2 for r in ok_calls), len(ok_calls)),
        "unverified": sum(r["status"] == "unverified" for r in rs),
        "errors": sum(r["status"] == "error" for r in rs),
        "status_agreement": _mean(agree),
        "coverage_sd": _mean(cov_sd),
        "latency_cv": _mean(lat_cv),
    }
    return {"n_records": len(rs), "quality": q, "speed": s, "stability": stab}


def fmt(v, suffix="", pct=False):
    if v is None:
        return "—"
    if pct:
        return f"{v * 100:.0f}%"
    return f"{v}{suffix}"


def write_report(records, questions, gens: dict, meta: dict, summ: dict) -> str:
    names = list(gens)
    label = {g: f"{g} ({gens[g].model})" for g in names}
    L = ["# День 28. Локальная LLM + RAG: сравнение локальной и облачной модели", ""]
    L += [f"Сформировано: {meta['created']}  ", f"Поиск (локально): эмбеддинги `{meta['embedder']}`, режим `{meta['mode']}`, "
          f"структурный индекс, {meta['chunks']} чанков  ",
          f"Прогонов на вопрос: {meta['runs']}; вопросов: {meta['questions']}; "
          f"модели: " + ", ".join(label.values()), ""]
    if len(names) == 1:
        L += ["> Облачная модель в этом прогоне не участвовала (нет ключа AITunnel или указан `--no-cloud`).", ""]
    if meta.get("warmup"):
        w = meta["warmup"]
        L += [f"Холодный старт локальных моделей: загрузка эмбеддингов {w['embed_model_load_sec']} с, "
              f"загрузка генератора {w['llm_cold_start_sec']} с (в замеры скорости не входит).", ""]

    def table(title, rows):
        L.append(f"### {title}")
        L.append("| Метрика | " + " | ".join(label[g] for g in names) + " |")
        L.append("|---|" + "---|" * len(names))
        for name, getter in rows:
            L.append(f"| {name} | " + " | ".join(getter(summ[g]) for g in names) + " |")
        L.append("")

    L.append("## 1. Качество")
    table("Сводка", [
        ("Вопросы с ответом: доля со статусом «answered»", lambda s: fmt(s["quality"]["answerable_answered"], pct=True)),
        ("Покрытие обязательных фактов (все вопросы с ответом)", lambda s: fmt(s["quality"]["coverage_mean"], pct=True)),
        ("Покрытие фактов (только вопросы по проектам)", lambda s: fmt(s["quality"]["coverage_project_only"], pct=True)),
        ("Нужный источник назван", lambda s: fmt(s["quality"]["source_ok"], pct=True)),
        ("Ключевые элементы ответа найдены в цитатах", lambda s: fmt(s["quality"]["grounding_mean"], pct=True)),
        ("LLM-судья: «supported»", lambda s: fmt(s["quality"]["judge_supported"], pct=True)),
        ("Правильный отказ на вопросах без ответа (обычный режим)", lambda s: fmt(s["quality"]["refusal_correct"], pct=True)),
        ("Стресс (слабый контекст): правильный отказ", lambda s: fmt(s["quality"]["stress_refusal_correct"], pct=True)),
        ("Стресс: выдуманных ответов", lambda s: fmt(s["quality"]["stress_hallucinated"])),
    ])
    L.append("## 2. Скорость")
    table("Сводка", [
        ("Поиск, среднее (эмбеддинг запроса + фильтр), с", lambda s: fmt(s["speed"]["retrieval_sec_mean"])),
        ("Ответ модели, среднее, с", lambda s: fmt(s["speed"]["answer_sec_mean"])),
        ("Ответ модели, медиана, с", lambda s: fmt(s["speed"]["answer_sec_median"])),
        ("Ответ модели, 90-й процентиль, с", lambda s: fmt(s["speed"]["answer_sec_p90"])),
        ("Токенов в секунду", lambda s: f"{fmt(s['speed']['tok_per_sec_mean'])} ({s['speed']['tok_per_sec_kind'] or '—'})"),
        ("Токенов ответа в среднем", lambda s: fmt(s["speed"]["completion_tokens_mean"])),
        ("Токенов запроса в среднем", lambda s: fmt(s["speed"]["prompt_tokens_mean"])),
        ("Догрузок модели в память за прогон, с", lambda s: fmt(s["speed"]["model_load_sec_total"])),
        ("Запросов, не влезших в контекстное окно", lambda s: fmt(s["speed"]["prompt_truncated"])),
    ])
    L.append("## 3. Стабильность")
    table("Сводка", [
        ("Ответ прошёл проверку цитат с первой попытки", lambda s: fmt(s["stability"]["first_try_ok"], pct=True)),
        ("Понадобилась повторная попытка", lambda s: fmt(s["stability"]["retry_used"], pct=True)),
        ("Совпадение статуса между прогонами", lambda s: fmt(s["stability"]["status_agreement"], pct=True)),
        ("Разброс покрытия между прогонами (σ)", lambda s: fmt(s["stability"]["coverage_sd"])),
        ("Разброс времени ответа (коэфф. вариации)", lambda s: fmt(s["stability"]["latency_cv"])),
        ("«Не подтверждено цитатами» (unverified)", lambda s: fmt(s["stability"]["unverified"])),
        ("Сбои вызова (ошибки)", lambda s: fmt(s["stability"]["errors"])),
    ])

    L.append("## 4. Автоматические наблюдения")
    L += auto_notes(summ, names)
    L.append("")

    L.append("## 5. Ответы по вопросам (прогон 1)")
    for q in questions:
        L.append(f"### {q['id']}. {q['q']}  *(тип: {q['type']})*")
        if q.get("expected"):
            L.append(f"Эталон: {q['expected']}")
        for g in names:
            rs = [r for r in records if r["gen"] == g and r["qid"] == q["id"] and r["run"] == 1 and not r["stress"]]
            if not rs:
                continue
            r = rs[0]
            cov = f", покрытие {r['coverage']:.0%}" if r["coverage"] is not None else ""
            t = f", {r['gen_wall_sec']:.1f} с" if r.get("gen_wall_sec") else ", модель не вызывалась"
            L.append(f"- **{label[g]}** — {r['status']}{cov}{t}")
            L.append(f"  > {(r['answer'] or '(нет ответа)')[:600]}")
            if r["sources"]:
                L.append(f"  > Источники: {', '.join(r['sources'])}")
        L.append("")

    stress = [r for r in records if r["stress"] and r["run"] == 1]
    if stress:
        L.append("## 6. Стресс-тест (вопросы без ответа, отказ по порогу выключен)")
        for r in stress:
            qq = next(x for x in questions if x["id"] == r["qid"])
            L.append(f"- вопрос {r['qid']} «{qq['q']}» · {label[r['gen']]}: **{r['status']}** — "
                     f"{(r['answer'] or '')[:200]}")
        L.append("")

    L += ["## Ограничения", "",
          "- Вопросов мало (см. выше), поэтому доли — оценка, а не статистика; различия в несколько процентов — шум.",
          "- Пороги поиска подобраны `calibrate.py` на тех же вопросах — поиск немного «подогнан» под набор. "
          "На сравнение моделей это не влияет: контекст у обеих одинаковый.",
          "- Скорость локальной модели зависит от железа (CPU/GPU, RAM), облачной — от сети и нагрузки сервиса; "
          "токены/с у облака посчитаны вместе с сетевой задержкой, у локальной — по счётчикам Ollama.",
          "- Покрытие фактов — проверка по ключевым словам: она не видит смысловых ошибок. Для этого есть LLM-судья (`--judge`)."]
    return "\n".join(L)


def auto_notes(summ: dict, names: list) -> list:
    notes = []
    loc = summ.get("local")
    cl = summ.get("cloud")
    if loc and loc["stability"]["errors"]:
        notes.append(f"- Локальная модель: сбоев вызова — {loc['stability']['errors']}.")
    if loc and loc["speed"]["prompt_truncated"]:
        notes.append(f"- ВНИМАНИЕ: у локальной модели {loc['speed']['prompt_truncated']} запросов упёрлись в контекстное окно — "
                     f"часть контекста могла быть обрезана (увеличьте num_ctx).")
    if loc and cl:
        a, b = loc["speed"]["answer_sec_mean"], cl["speed"]["answer_sec_mean"]
        if a and b:
            notes.append(f"- Скорость: локальная отвечает в среднем за {a} с, облачная за {b} с "
                         f"({'локальная медленнее в %.1f раза' % (a / b) if a > b else 'локальная быстрее в %.1f раза' % (b / a)}).")
        a, b = loc["quality"]["coverage_mean"], cl["quality"]["coverage_mean"]
        if a is not None and b is not None:
            d = round((a - b) * 100)
            notes.append(f"- Качество: покрытие фактов — локальная {a:.0%}, облачная {b:.0%} "
                         f"({'разница в пределах шума' if abs(d) < 8 else ('локальная ниже на %d п.п.' % -d if d < 0 else 'локальная выше на %d п.п.' % d)}).")
        a, b = loc["stability"]["first_try_ok"], cl["stability"]["first_try_ok"]
        if a is not None and b is not None:
            notes.append(f"- Стабильность формата: с первой попытки проходят проверку цитат {a:.0%} ответов локальной и {b:.0%} облачной.")
    if loc and loc["quality"]["stress_hallucinated"]:
        notes.append(f"- Стресс-тест: локальная модель выдала ответ там, где ответа в базе нет, "
                     f"{loc['quality']['stress_hallucinated']} раз.")
    if cl and cl["quality"]["stress_hallucinated"]:
        notes.append(f"- Стресс-тест: облачная модель выдала ответ там, где ответа в базе нет, "
                     f"{cl['quality']['stress_hallucinated']} раз.")
    return notes or ["- (недостаточно данных для автоматических выводов)"]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(args) -> int:
    with open(args.questions, encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    if args.only:
        questions = [q for q in questions if q["id"] in args.only]
    cfg = load_config()

    # --- локальная среда
    try:
        installed = ollama_models()
    except RuntimeError as e:
        print(f"❌ {e}")
        return 2
    index_model = os.environ.get("OLLAMA_EMBED_MODEL", "bge-m3")
    for need in (args.local_model, index_model):
        if not has_model(installed, need):
            print(f"❌ Модель {need} не скачана. Выполните: ollama pull {need}")
            return 2

    index = load_index(args.backend, args.strategy)
    print(f"Индекс: {index.meta['embedder']['label']}, чанков: {len(index.chunks)}")
    if args.backend not in cfg["thresholds"] or "answer_min" not in cfg["thresholds"][args.backend]:
        print("ℹ Пороги поиска не откалиброваны (python calibrate.py --backend ollama) — используются значения по умолчанию.")

    gens = {"local": TimedLLM(OllamaLLM(args.local_model), "local")}
    cloud_raw = None
    if not args.no_cloud and os.environ.get("AITUNNEL_API_KEY"):
        try:
            cloud_raw = LLM(args.cloud_model)
            gens["cloud"] = TimedLLM(cloud_raw, "cloud")
        except SystemExit as e:
            print(f"Облако пропущено: {e}")
    else:
        print("Облачная модель пропущена (нет AITUNNEL_API_KEY или указан --no-cloud) — сравнение только по локальной.")

    print("Прогреваю локальные модели ...")
    warm = warmup(gens["local"], index)
    print(f"  загрузка эмбеддингов {warm['embed_model_load_sec']} с, генератора {warm['llm_cold_start_sec']} с")

    records = run_all(args, questions, index, cfg, gens)
    records = [r for r in records if r["gen"] in gens and r["model"] == gens[r["gen"]].model]

    judge_llm = None
    if args.judge == "cloud" or (args.judge == "auto" and cloud_raw):
        judge_llm = cloud_raw
        args.judge = "cloud"
    elif args.judge == "local":
        judge_llm = gens["local"].llm
    if judge_llm is not None:
        add_judge(args, records, questions, judge_llm)

    summ = {g: summarize(records, g) for g in gens}
    meta = {"created": datetime.now().strftime("%Y-%m-%d %H:%M"), "embedder": index.meta["embedder"]["label"],
            "mode": args.mode, "chunks": len(index.chunks), "runs": args.runs, "questions": len(questions),
            "warmup": warm, "judge": args.judge if judge_llm else "none"}
    report = write_report(records, questions, gens, meta, summ)
    with open(os.path.join(OUT, "report_day28.md"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(OUT, "results_day28.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summ, "records": records}, f, ensure_ascii=False, indent=2)

    print("\n" + report.split("## 5.")[0])
    print(f"Отчёт: {os.path.relpath(os.path.join(OUT, 'report_day28.md'), BASE)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Сравнение локальной и облачной модели в RAG")
    ap.add_argument("--backend", default="ollama", choices=["ollama", "local", "aitunnel"],
                    help="эмбеддинги индекса (для полностью локальной работы — ollama)")
    ap.add_argument("--strategy", default="structure", choices=["structure", "fixed"])
    ap.add_argument("--mode", default="filter", help="режим поиска; filter не использует LLM (поиск полностью локальный)")
    ap.add_argument("--local-model", default=os.environ.get("OLLAMA_MODEL", "qwen2.5:7b"))
    ap.add_argument("--cloud-model", default=os.environ.get("AITUNNEL_MODEL", CLOUD_DEFAULT))
    ap.add_argument("--no-cloud", action="store_true", help="не использовать облачную модель")
    ap.add_argument("--runs", type=int, default=3, help="сколько раз прогнать каждый вопрос (стабильность)")
    ap.add_argument("--questions", default=os.path.join(BASE, "eval_questions.json"))
    ap.add_argument("--only", type=int, nargs="*", help="только вопросы с этими номерами")
    ap.add_argument("--no-stress", dest="stress", action="store_false", help="не делать стресс-тест")
    ap.add_argument("--judge", choices=["auto", "cloud", "local", "none"], default="auto",
                    help="LLM-судья смысла ответов: auto — облачная модель, если есть ключ")
    ap.add_argument("--resume", action="store_true", help="продолжить прерванный прогон (output/runs.jsonl)")
    return ap


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        sys.exit(main(build_parser().parse_args()))
    except RuntimeError as e:
        print(f"❌ {e}")
        sys.exit(2)
