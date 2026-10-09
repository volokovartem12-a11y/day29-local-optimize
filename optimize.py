"""
День 29. Оптимизация локальной модели под задачу «ответ по документации с дословными цитатами».

Профили (profiles.json) прогоняются на ОДНОМ И ТОМ ЖЕ найденном контексте:

    baseline  — настройки дня 28 (до оптимизации)
    params    — + параметры генерации: temperature 0, seed, top_k/top_p, окно 6144, лимит ответа 600
    prompt    — + промпт под задачу (v2) и JSON-схема ответа
    q8_0      — то же на квантовании Q8_0
    q3_K_M    — то же на квантовании Q3_K_M
    ctx4096   — проверка нижней границы окна контекста

Для каждого профиля меряется:
  КАЧЕСТВО   покрытие обязательных фактов, ключевые элементы ответа в цитатах, ответы «мимо»
             (answered, но ни одного нужного факта), проверка цитат с первой попытки,
             правильный отказ (в т.ч. стресс-тест), обрезанные ответы/контекст;
             отдельно — на НОВЫХ вопросах (heldout_questions.json), которые не участвовали в настройке.
  СКОРОСТЬ   время ответа (медиана, p90), токены/с, время чтения контекста, холодная загрузка.
  РЕСУРСЫ    размер файла модели, память под загруженную модель (из /api/ps), доля на видеокарте.

Лучший профиль выбирается по индексу качества (при равенстве — быстрее и легче), сохраняется
в optimized_profile.json и в Modelfile → `ollama create advent-rag -f Modelfile`.

    python optimize.py                    # все профили, 2 прогона
    python optimize.py --profiles baseline prompt --runs 1
    python optimize.py --final            # проверить собранную модель advent-rag
    python optimize.py --resume           # продолжить прерванный прогон
"""

import argparse
import copy
import json
import os
import statistics as st
import sys
import time
from datetime import datetime

from embeddings import ollama_post
from evaluate_local import key_of, make_record, summarize
from grounded import Grounded, generate, prepare
from local_llm import OllamaLLM, TimedLLM, has_model, ollama_get, ollama_models
from prompts import PROMPTS, response_format
from rag import BASE, load_index
from retrieval import load_config

OUT = os.path.join(BASE, "output")
FINAL_MODEL = "advent-rag"
GB = 1024 ** 3


# ---------------------------------------------------------------------------
# Ollama: модели и ресурсы
# ---------------------------------------------------------------------------

def pull(model: str) -> None:
    """Скачать модель через API Ollama (с прогрессом)."""
    import urllib.request
    from embeddings import ollama_url
    print(f"  Скачиваю {model} ...", flush=True)
    req = urllib.request.Request(ollama_url() + "/api/pull", data=json.dumps({"model": model}).encode(),
                                 headers={"Content-Type": "application/json"})
    last = -10
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for line in resp:
            ev = json.loads(line.decode("utf-8") or "{}")
            if ev.get("error"):
                raise RuntimeError(f"ollama pull {model}: {ev['error']}")
            if ev.get("total") and ev.get("completed"):
                pct = int(100 * ev["completed"] / ev["total"])
                if pct >= last + 10:
                    print(f"    {pct}%", flush=True)
                    last = pct
    print(f"  ✅ {model} скачана")


def unload_all() -> None:
    """Выгрузить все модели из памяти — чтобы замер памяти и холодной загрузки был честным."""
    try:
        for m in ollama_get("/api/ps").get("models", []):
            ollama_post("/api/generate", {"model": m["name"], "keep_alive": 0}, timeout=60)
    except Exception:  # noqa: BLE001
        pass
    time.sleep(1)


def model_info(model: str) -> dict:
    info = {}
    try:
        for m in ollama_get("/api/tags").get("models", []):
            if m["name"] in (model, f"{model}:latest"):
                d = m.get("details") or {}
                info = {"file_gb": round(m.get("size", 0) / GB, 2), "quant": d.get("quantization_level"),
                        "params": d.get("parameter_size")}
    except Exception:  # noqa: BLE001
        pass
    return info


def loaded_info(model: str) -> dict:
    try:
        for m in ollama_get("/api/ps").get("models", []):
            if m["name"] in (model, f"{model}:latest"):
                size, vram = m.get("size", 0), m.get("size_vram", 0)
                return {"mem_gb": round(size / GB, 2), "vram_gb": round(vram / GB, 2),
                        "gpu_share": round(vram / size, 2) if size else None,
                        "context_length": m.get("context_length")}
    except Exception:  # noqa: BLE001
        pass
    return {}


# ---------------------------------------------------------------------------
# Прогон профилей
# ---------------------------------------------------------------------------

def load_sets(args) -> dict:
    sets = {}
    for name, path in (("main", args.questions), ("heldout", args.heldout)):
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                sets[name] = json.load(f)["questions"]
    return sets


def make_llm(p: dict) -> OllamaLLM:
    return OllamaLLM(p["model"], options=p.get("options", {}), fmt=response_format(p.get("format", "auto")))


def run_profiles(args, profiles, sets, index, cfg) -> tuple:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "runs29.jsonl")
    records, done, meta = [], set(), {}
    if (args.resume or args.final) and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    records.append(r)
                    done.add(key_of(r) + (r.get("set"),))
        print(f"Загружено готовых записей: {len(records)}")
    elif not args.final:
        open(path, "w", encoding="utf-8").close()
    meta_path = os.path.join(OUT, "profiles_meta29.json")
    if os.path.exists(meta_path) and (args.resume or args.final):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

    # поиск один раз на вопрос: контекст одинаков для всех профилей
    cache = {}

    def retrieve_once(q, stress):
        k = (q["id"], stress)
        if k not in cache:
            t0 = time.perf_counter()
            try:
                res0, chunks = prepare(index, q["q"], cfg, "ollama", "filter", None, force=stress)
                err = None
            except Exception as e:  # noqa: BLE001
                res0, chunks, err = None, None, str(e)
            cache[k] = (res0, chunks, time.perf_counter() - t0, err)
        return cache[k]

    with open(path, "a", encoding="utf-8") as out:
        for p in profiles:
            pid = p["id"]
            todo = [(s, q, run, stress) for run in range(1, args.runs + 1) for s, qs in sets.items() for q in qs
                    for stress in ([False] + ([True] if args.stress and q["type"] in ("absent", "vague") else []))
                    if (run, q["id"], pid, p["model"], stress, s) not in done]
            print(f"\n=== Профиль {pid}: {p['title']}  ·  модель {p['model']}  ·  вызовов: {len(todo)} ===", flush=True)
            if not todo:
                continue
            system, prompt_fn = PROMPTS[p.get("prompt", "v1")]
            timed = TimedLLM(make_llm(p), pid)

            # холодный старт и память: выгружаем всё, грузим модель этого профиля с его окном
            unload_all()
            index.embedder.embed(["прогрев"])
            t0 = time.perf_counter()
            try:
                timed.llm.chat("Ответь одним словом.", "Привет")
            except Exception as e:  # noqa: BLE001
                print(f"  ❌ модель не запускается: {e}")
                meta[pid] = {"error": str(e)}
                continue
            cold = time.perf_counter() - t0
            m = {"cold_load_sec": round(cold, 2), **model_info(p["model"]), **loaded_info(p["model"])}
            meta[pid] = m
            print(f"  загрузка {m['cold_load_sec']} с · файл {m.get('file_gb', '?')} ГБ ({m.get('quant', '?')}) · "
                  f"в памяти {m.get('mem_gb', '?')} ГБ, на видеокарте {int((m.get('gpu_share') or 0) * 100)}%",
                  flush=True)

            for n, (s, q, run, stress) in enumerate(todo, 1):
                res0, chunks, retr_sec, err = retrieve_once(q, stress)
                timed.reset()
                res = copy.deepcopy(res0) if res0 is not None else Grounded(q["q"], "unknown", "")
                if chunks is not None and not err:
                    try:
                        generate(timed, index, q["q"], res, chunks, system=system, prompt_fn=prompt_fn)
                    except Exception as e:  # noqa: BLE001
                        err = str(e)
                rec = make_record(res, q, pid, p["model"], run, stress, retr_sec, timed, err)
                rec["set"] = s
                records.append(rec)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                cov = f"{rec['coverage']:.0%}" if rec["coverage"] is not None else "-"
                sec = rec.get("gen_wall_sec")
                print(f"  [{n}/{len(todo)}] {'новый ' if s == 'heldout' else ''}вопрос {q['id']:>3}"
                      f"{' стресс' if stress else ''} · прогон {run} · {rec['status']:<10} покрытие={cov} · "
                      f"{('%.1f с' % sec) if sec else 'без модели'}"
                      + (" · ОБРЕЗАН" if rec.get("hit_limit") else "")
                      + (f" · ОШИБКА: {err[:70]}" if err else ""), flush=True)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
    return records, meta


# ---------------------------------------------------------------------------
# Метрики, выбор лучшего профиля
# ---------------------------------------------------------------------------

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def quality(records: list, pid: str) -> dict:
    s = summarize(records, pid)
    rs = [r for r in records if r["gen"] == pid and not r["stress"]]
    ans_q = [r for r in rs if r["type"] in ("project", "general")]
    q = s["quality"]
    wrong = sum(r["status"] == "answered" and r["coverage"] == 0 for r in ans_q if r["type"] == "project")
    refusal = _mean([q["refusal_correct"], q["stress_refusal_correct"]])
    parts = {"coverage": q["coverage_mean"], "grounding": q["grounding_mean"],
             "first_try": s["stability"]["first_try_ok"], "refusal": refusal}
    weights = {"coverage": 0.4, "grounding": 0.25, "first_try": 0.15, "refusal": 0.2}
    avail = {k: v for k, v in parts.items() if v is not None}
    qi = round(sum(weights[k] * v for k, v in avail.items()) / sum(weights[k] for k in avail), 3) if avail else None
    allr = [r for r in records if r["gen"] == pid]
    return {**s, "qi": qi, "wrong_answers": wrong, "n_answerable": len(ans_q),
            "hit_limit": sum(bool(r.get("hit_limit")) for r in allr),
            "truncated": sum(bool(r.get("truncated_prompt")) for r in allr),
            "prompt_eval_sec": _mean(r.get("prompt_eval_sec") for r in allr if r.get("model_called"))}


def evaluate_all(records, profiles) -> dict:
    out = {}
    for p in profiles:
        pid = p["id"]
        if not any(r["gen"] == pid for r in records):
            continue
        out[pid] = {"main": quality([r for r in records if r.get("set") == "main"], pid),
                    "heldout": quality([r for r in records if r.get("set") == "heldout"], pid),
                    "all": quality(records, pid)}
    return out


def choose(results: dict, meta: dict, profiles: list) -> tuple:
    cands = []
    for p in profiles:
        r = results.get(p["id"])
        if not r or r["all"]["qi"] is None or p["id"] == "final" or p.get("selectable") is False:
            continue
        qi = r["all"]["qi"] - 0.05 * min(r["all"]["truncated"], 3)  # обрезанный контекст — штраф
        lat = r["all"]["speed"]["answer_sec_median"] or 1e9
        mem = (meta.get(p["id"]) or {}).get("mem_gb") or 1e9
        cands.append((qi, lat, mem, p))
    if not cands:
        return None, ""
    best_qi = max(c[0] for c in cands)
    close = [c for c in cands if c[0] >= best_qi - 0.02]  # разница меньше 0.02 — в пределах шума
    close.sort(key=lambda c: (c[1], c[2]))
    qi, lat, mem, p = close[0]
    why = (f"наивысший индекс качества по всем вопросам ({best_qi:.3f})" if qi == best_qi else
           f"индекс качества {qi:.3f} в пределах 0.02 от лучшего ({best_qi:.3f}), но быстрее/легче")
    if len(close) > 1:
        why += f"; среди близких по качеству ({', '.join(c[3]['id'] for c in close)}) — самый быстрый"
    return p, why


def write_modelfile(p: dict) -> str:
    opts = p.get("options", {})
    system = PROMPTS[p.get("prompt", "v1")][0]
    lines = [f"# Оптимизированная локальная модель для RAG по документации челленджа (день 29).",
             f"# Профиль: {p['id']} — {p['title']}",
             f"# Сборка:  ollama create {FINAL_MODEL} -f Modelfile",
             f"# Формат ответа (JSON-схема) задаётся в запросе к API, в Modelfile его указать нельзя.",
             f"FROM {p['model']}"]
    lines += [f"PARAMETER {k} {v}" for k, v in opts.items()]
    lines.append(f'SYSTEM """{system}"""')
    text = "\n".join(lines) + "\n"
    with open(os.path.join(BASE, "Modelfile"), "w", encoding="utf-8") as f:
        f.write(text)
    return text


# ---------------------------------------------------------------------------
# Отчёт
# ---------------------------------------------------------------------------

def fmt(v, pct=False, nd=None):
    if v is None:
        return "—"
    if pct:
        return f"{v * 100:.0f}%"
    if nd is not None and isinstance(v, (int, float)):
        return f"{v:.{nd}f}"
    return str(v)


def write_report(records, results, meta, profiles, best, why, args) -> str:
    ps = [p for p in profiles if p["id"] in results]
    L = ["# День 29. Оптимизация локальной LLM под задачу", "",
         f"Сформировано: {datetime.now():%Y-%m-%d %H:%M}  ",
         "Задача: ответы по документации проектов челленджа с обязательными дословными цитатами (RAG из дня 28).  ",
         f"Поиск один и тот же для всех профилей (bge-m3, локально). Прогонов на вопрос: {args.runs}. "
         f"Вопросы: основной набор (14) + новые вопросы (9), которые не участвовали в настройке.", ""]

    def table(title, cols, rows_fn):
        L.append(f"### {title}")
        L.append("| Профиль | " + " | ".join(cols) + " |")
        L.append("|---|" + "---|" * len(cols))
        for p in ps:
            L.append(f"| **{p['id']}** | " + " | ".join(rows_fn(p, results[p['id']], meta.get(p['id'], {}))) + " |")
        L.append("")

    L.append("## Профили")
    L.append("| Профиль | Что меняется | Модель | Промпт / формат | temperature | num_ctx | num_predict |")
    L.append("|---|---|---|---|---|---|---|")
    for p in ps:
        o = p.get("options", {})
        L.append(f"| **{p['id']}** | {p['title']} | `{p['model']}` | {p.get('prompt')} / {p.get('format')} | "
                 f"{o.get('temperature', 'из Modelfile')} | {o.get('num_ctx', 'из Modelfile')} | "
                 f"{o.get('num_predict', 'из Modelfile')} |")
    L.append("")

    L.append("## 1. Качество")
    L.append("Индекс качества = 0.4·покрытие фактов + 0.25·элементы ответа в цитатах + "
             "0.15·проверка цитат с первой попытки + 0.2·правильный отказ (обычный и стресс).")
    L.append("")
    table("Основной набор (14 вопросов)",
          ["Индекс", "Покрытие фактов", "Элементы ответа в цитатах", "С первой попытки", "Ответы «мимо»",
           "Правильный отказ", "Стресс: выдумал", "Не подтверждено", "Ответ обрезан", "Контекст обрезан"],
          lambda p, r, m: [fmt(r["main"]["qi"], nd=3), fmt(r["main"]["quality"]["coverage_mean"], True),
                           fmt(r["main"]["quality"]["grounding_mean"], True),
                           fmt(r["main"]["stability"]["first_try_ok"], True),
                           str(r["main"]["wrong_answers"]), fmt(r["main"]["quality"]["refusal_correct"], True),
                           str(r["main"]["quality"]["stress_hallucinated"]), str(r["main"]["stability"]["unverified"]),
                           str(r["main"]["hit_limit"]), str(r["main"]["truncated"])])
    table("Новые вопросы (9, не участвовали в настройке)",
          ["Индекс", "Покрытие фактов", "Элементы ответа в цитатах", "С первой попытки", "Ответы «мимо»",
           "Правильный отказ"],
          lambda p, r, m: [fmt(r["heldout"]["qi"], nd=3), fmt(r["heldout"]["quality"]["coverage_mean"], True),
                           fmt(r["heldout"]["quality"]["grounding_mean"], True),
                           fmt(r["heldout"]["stability"]["first_try_ok"], True), str(r["heldout"]["wrong_answers"]),
                           fmt(r["heldout"]["quality"]["refusal_correct"], True)])

    L.append("## 2. Скорость")
    table("Все вопросы",
          ["Ответ, медиана, с", "Ответ, p90, с", "Токенов/с", "Чтение контекста, с", "Токенов ответа",
           "Повторных попыток", "Холодная загрузка, с"],
          lambda p, r, m: [fmt(r["all"]["speed"]["answer_sec_median"]), fmt(r["all"]["speed"]["answer_sec_p90"]),
                           fmt(r["all"]["speed"]["tok_per_sec_mean"], nd=1), fmt(r["all"]["prompt_eval_sec"], nd=2),
                           fmt(r["all"]["speed"]["completion_tokens_mean"], nd=0),
                           fmt(r["all"]["stability"]["retry_used"], True), fmt(m.get("cold_load_sec"))])

    L.append("## 3. Ресурсы")
    table("Память (по данным Ollama /api/ps после загрузки с окном профиля)",
          ["Квантование", "Файл модели, ГБ", "В памяти, ГБ", "Из них видеопамять, ГБ", "На видеокарте", "Окно"],
          lambda p, r, m: [str(m.get("quant") or "—"), fmt(m.get("file_gb")), fmt(m.get("mem_gb")),
                           fmt(m.get("vram_gb")), fmt(m.get("gpu_share"), True),
                           str(m.get("context_length") or p.get("options", {}).get("num_ctx", "—"))])

    L.append("## 4. Стабильность")
    table("Между прогонами",
          ["Совпадение статуса", "Разброс покрытия (σ)", "Разброс времени (CV)", "Сбои"],
          lambda p, r, m: [fmt(r["all"]["stability"]["status_agreement"], True),
                           fmt(r["all"]["stability"]["coverage_sd"]), fmt(r["all"]["stability"]["latency_cv"]),
                           str(r["all"]["stability"]["errors"])])

    if best:
        b, base = results[best["id"]], results.get("baseline")
        L.append(f"## 5. Итог: выбран профиль `{best['id']}`")
        L.append(f"Почему: {why}.  ")
        L.append(f"Собран в модель Ollama `{FINAL_MODEL}` (файл `Modelfile`).")
        L.append("")
        if base and best["id"] != "baseline":
            L.append("| Метрика | До (baseline) | После (" + best["id"] + ") |")
            L.append("|---|---|---|")
            mb, ma = meta.get("baseline", {}), meta.get(best["id"], {})
            rows = [
                ("Индекс качества, основной набор", fmt(base["main"]["qi"], nd=3), fmt(b["main"]["qi"], nd=3)),
                ("Индекс качества, новые вопросы", fmt(base["heldout"]["qi"], nd=3), fmt(b["heldout"]["qi"], nd=3)),
                ("Покрытие фактов (все)", fmt(base["all"]["quality"]["coverage_mean"], True),
                 fmt(b["all"]["quality"]["coverage_mean"], True)),
                ("Элементы ответа в цитатах", fmt(base["all"]["quality"]["grounding_mean"], True),
                 fmt(b["all"]["quality"]["grounding_mean"], True)),
                ("Проверка цитат с первой попытки", fmt(base["all"]["stability"]["first_try_ok"], True),
                 fmt(b["all"]["stability"]["first_try_ok"], True)),
                ("Ответы «мимо»", str(base["all"]["wrong_answers"]), str(b["all"]["wrong_answers"])),
                ("Ответ, медиана, с", fmt(base["all"]["speed"]["answer_sec_median"]),
                 fmt(b["all"]["speed"]["answer_sec_median"])),
                ("Токенов ответа в среднем", fmt(base["all"]["speed"]["completion_tokens_mean"], nd=0),
                 fmt(b["all"]["speed"]["completion_tokens_mean"], nd=0)),
                ("Память модели, ГБ", fmt(mb.get("mem_gb")), fmt(ma.get("mem_gb"))),
                ("Совпадение статуса между прогонами", fmt(base["all"]["stability"]["status_agreement"], True),
                 fmt(b["all"]["stability"]["status_agreement"], True)),
            ]
            L += [f"| {a} | {x} | {y} |" for a, x, y in rows]
            L.append("")

        L.append("## 6. Ответы до и после (прогон 1)")
        for s, name in (("main", "основной набор"), ("heldout", "новые вопросы")):
            qids = sorted({r["qid"] for r in records if r.get("set") == s})
            L.append(f"### {name.capitalize()}")
            for qid in qids:
                for pid in ("baseline", best["id"]):
                    rs = [r for r in records if r["gen"] == pid and r["qid"] == qid and r["run"] == 1
                          and not r["stress"] and r.get("set") == s]
                    if not rs:
                        continue
                    r = rs[0]
                    cov = f", покрытие {r['coverage']:.0%}" if r["coverage"] is not None else ""
                    L.append(f"- вопрос {qid} · **{pid}** — {r['status']}{cov}: {(r['answer'] or '')[:300]}")
            L.append("")

    L += ["## Ограничения", "",
          "- Вопросов немного (14 + 9): разница индекса меньше ~0.02–0.03 — шум, поэтому при близком качестве "
          "выбирается более быстрый и лёгкий профиль.",
          "- Покрытие фактов — проверка по ключевым словам; смысловые ошибки она не видит.",
          "- Промпт v2 писался после анализа ошибок на основном наборе — поэтому отдельно показаны новые вопросы.",
          "- Память — по данным Ollama (/api/ps): модель + кэш контекста; загрузку видеокарты в процентах Ollama не отдаёт.",
          "- При temperature 0 и фиксированном seed ответы почти детерминированы, поэтому стабильность между прогонами "
          "у оптимизированных профилей ожидаемо выше."]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(args) -> int:
    with open(args.profiles_file, encoding="utf-8") as f:
        profiles = json.load(f)["profiles"]
    if args.final:
        with open(os.path.join(BASE, "optimized_profile.json"), encoding="utf-8") as f:
            best = json.load(f)["profile"]
        final = {**best, "id": "final", "title": f"Собранная модель {FINAL_MODEL} (параметры из Modelfile)",
                 "model": FINAL_MODEL, "options": {}}
        run_list = [final]
        profiles = profiles + [final]
    else:
        run_list = [p for p in profiles if not args.profiles or p["id"] in args.profiles]

    installed = ollama_models()
    for need in ["bge-m3"] + sorted({p["model"] for p in run_list}):
        if not has_model(installed, need):
            if need == FINAL_MODEL:
                print(f"❌ Модель {FINAL_MODEL} не собрана: ollama create {FINAL_MODEL} -f Modelfile")
                return 2
            if args.no_pull:
                print(f"❌ Модель {need} не скачана: ollama pull {need}")
                return 2
            try:
                pull(need)
            except Exception as e:  # noqa: BLE001
                print(f"⚠️  {need} скачать не удалось ({e}) — профили с ней пропущены")
                run_list = [p for p in run_list if p["model"] != need]

    sets = load_sets(args)
    index = load_index("ollama", "structure")
    cfg = load_config()
    if "answer_min" not in cfg["thresholds"].get("ollama", {}):
        print("ℹ Пороги поиска не откалиброваны (python calibrate.py --backend ollama) — значения по умолчанию.")

    records, meta = run_profiles(args, run_list, sets, index, cfg)
    known = {p["id"] for p in profiles}
    records = [r for r in records if r["gen"] in known]
    results = evaluate_all(records, profiles)

    best, why = None, ""
    if not args.final:
        best, why = choose(results, meta, profiles)
    if best and not args.final:
        with open(os.path.join(BASE, "optimized_profile.json"), "w", encoding="utf-8") as f:
            json.dump({"profile": best, "why": why, "model_name": FINAL_MODEL,
                       "created": datetime.now().isoformat(timespec="seconds")}, f, ensure_ascii=False, indent=2)
        write_modelfile(best)
    if args.final:
        with open(os.path.join(BASE, "optimized_profile.json"), encoding="utf-8") as f:
            saved = json.load(f)
        best, why = saved["profile"], saved["why"]

    report = write_report(records, results, meta, profiles, best, why, args)
    with open(os.path.join(OUT, "report_day29.md"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(OUT, "results_day29.json"), "w", encoding="utf-8") as f:
        json.dump({"results": results, "meta": meta, "best": best, "why": why, "records": records},
                  f, ensure_ascii=False, indent=2)
    print("\n" + report.split("## 6.")[0])
    if best and not args.final:
        print(f"Лучший профиль: {best['id']} — {why}")
        print(f"Сохранено: optimized_profile.json и Modelfile → ollama create {FINAL_MODEL} -f Modelfile")
    if args.final and "final" in results and best and best["id"] in results:
        a, b = results["final"]["all"], results[best["id"]]["all"]
        print(f"Проверка собранной модели {FINAL_MODEL}: индекс качества {a['qi']} "
              f"(профиль {best['id']} в замере: {b['qi']}), медиана ответа {a['speed']['answer_sec_median']} с.")
    print(f"Отчёт: {os.path.relpath(os.path.join(OUT, 'report_day29.md'), BASE)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Оптимизация локальной модели: сравнение профилей")
    ap.add_argument("--profiles-file", default=os.path.join(BASE, "profiles.json"))
    ap.add_argument("--profiles", nargs="*", help="только эти профили (по id)")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--questions", default=os.path.join(BASE, "eval_questions.json"))
    ap.add_argument("--heldout", default=os.path.join(BASE, "heldout_questions.json"))
    ap.add_argument("--no-stress", dest="stress", action="store_false")
    ap.add_argument("--no-pull", action="store_true", help="не скачивать недостающие модели")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--final", action="store_true", help=f"проверить собранную модель {FINAL_MODEL}")
    return ap


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        sys.exit(main(build_parser().parse_args()))
    except RuntimeError as e:
        print(f"❌ {e}")
        sys.exit(2)
