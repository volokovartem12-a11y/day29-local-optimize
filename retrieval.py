"""
День 23. Двухэтапный поиск: кандидаты → фильтр → реранкинг → финальный top-K.

    вопрос ─► [query rewrite] ─► поиск top-K_before кандидатов (по всем вариантам запроса)
           ─► порог отсечения:  similarity ≥ min_score  и  ≥ rel_ratio × лучший результат
           ─► реранкинг:        эвристика (близость + термины + раздел/файл + номер дня)
                                или LLM, которая сама оценивает релевантность
           ─► не больше max_per_source чанков из одного файла ─► top-K_after

Каждый вызов возвращает трассировку: сколько кандидатов было на каждом этапе
и почему отброшены остальные — чтобы работу фильтра можно было проверить.
"""

import json
import os
import re

import numpy as np
from dataclasses import dataclass, field

from embeddings import _tokens

BASE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------------

DEFAULTS = {
    "k_before": 12,          # сколько кандидатов достаём до фильтра
    "k_after": 4,            # сколько оставляем после
    "max_per_source": 2,     # не больше N чанков из одного файла — разнообразие контекста
    "weights": {"sim": 1.0, "lexical": 0.35, "meta": 0.25, "day_match": 0.4, "day_mismatch": -0.3},
    "thresholds": {          # подбираются calibrate.py; здесь значения по умолчанию
        "local": {"min_score": 0.10, "rel_ratio": 0.45},
        "aitunnel": {"min_score": 0.30, "rel_ratio": 0.70},
        "ollama": {"min_score": 0.35, "rel_ratio": 0.70},
    },
}


def load_config() -> dict:
    path = os.path.join(BASE, "config.json")
    cfg = json.loads(json.dumps(DEFAULTS))
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            user = json.load(f)
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                for kk, vv in v.items():
                    cfg[k][kk] = {**cfg[k].get(kk, {}), **vv} if isinstance(vv, dict) else vv
            else:
                cfg[k] = v
    return cfg


def save_thresholds(backend: str, thresholds: dict) -> None:
    path = os.path.join(BASE, "config.json")
    cfg = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
    cfg.setdefault("thresholds", {})[backend] = thresholds
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Query rewrite
# ---------------------------------------------------------------------------

_DAY = re.compile(r"(?:\bдень|\bдня|\bдне|\bднём|\bday)[\s\-]*(\d{1,2})\b", re.IGNORECASE)

GLOSSARY = {  # термины вопроса → как они пишутся в коде и README
    "слои": "слой слоя слоёв layers", "слой": "слоя слоёв layers", "памят": "memory",
    "сжати": "compression compress summary", "эконом": "экономия savings", "токен": "tokens",
    "инструмент": "tool tools", "стратеги": "strategy strategies", "поиск": "search",
    "раздел": "section", "чанк": "chunk chunking", "оркестр": "orchestrator orchestration",
    "маршрут": "route routing", "пайплайн": "pipeline", "провер": "check verify",
    "контекст": "context", "ветк": "branch branching", "факт": "facts",
    "эмбеддинг": "embeddings", "сервер": "server", "реранк": "rerank",
}


def mentioned_day(text: str):
    m = _DAY.search(text)
    return int(m.group(1)) if m else None


def rewrite_heuristic(question: str) -> str:
    """Без LLM: номер дня → префикс папки (day11), термины → варианты из кода и README."""
    extra = []
    day = mentioned_day(question)
    if day is not None:
        extra.append(f"day{day}")
    low = question.lower()
    extra += [v for k, v in GLOSSARY.items() if k in low]
    return f"{question} {' '.join(extra)}".strip()


REWRITE_PROMPT = (
    "Перепиши вопрос пользователя в поисковый запрос для поиска по базе документации проектов "
    "(README на русском и код на Python). Добавь ключевые термины, возможные названия функций, "
    "классов и файлов, английские эквиваленты. Если в вопросе есть номер дня — сохрани его и добавь "
    "форму dayN. Верни ТОЛЬКО запрос одной строкой, без пояснений.\n\nВопрос: {q}"
)


def rewrite_llm(llm, question: str) -> tuple:
    text, usage = llm.chat("Ты переписываешь вопросы в поисковые запросы.", REWRITE_PROMPT.format(q=question))
    text = text.strip().strip('"«»').splitlines()[0] if text.strip() else question
    return text, usage


# ---------------------------------------------------------------------------
# Реранкинг
# ---------------------------------------------------------------------------

_STOP = set("как что какой какая какие каким почему где зачем чем для при это этот эта или the and "
            "проект проекте проекта дня день между через после".split())


def _terms(text: str) -> set:
    return {t for t in _tokens(text) if len(t) > 2 and t not in _STOP}


def rerank_heuristic(question: str, cands: list, weights: dict) -> list:
    q_terms = _terms(question)
    day = mentioned_day(question)
    top_sim = max(c["score"] for c in cands) or 1.0
    for c in cands:
        body = _terms(c["text"])
        meta = _terms(f"{c['section']} {c['title']} {c['file']}")
        lexical = len(q_terms & body) / len(q_terms) if q_terms else 0.0
        meta_overlap = len(q_terms & meta) / len(q_terms) if q_terms else 0.0
        day_part = 0.0
        if day is not None:
            day_part = weights["day_match"] if c["source"].startswith(f"day{day}-") else weights["day_mismatch"]
        c["rerank"] = round(weights["sim"] * c["score"] / top_sim + weights["lexical"] * lexical
                            + weights["meta"] * meta_overlap + day_part, 4)
        c["why"] = f"sim {c['score']:.3f}, термины {lexical:.0%}, раздел/файл {meta_overlap:.0%}" + \
                   (f", день {'совпал' if day_part > 0 else 'другой'}" if day is not None else "")
    return sorted(cands, key=lambda c: -c["rerank"])


RERANK_PROMPT = (
    "Оцени, насколько каждый фрагмент помогает ответить на вопрос, по шкале 0–10 "
    "(0 — не относится, 10 — содержит прямой ответ). Верни ТОЛЬКО JSON-массив чисел "
    "в порядке фрагментов, например [7, 0, 3].\n\nВопрос: {q}\n\n{blocks}"
)


def rerank_llm(llm, question: str, cands: list) -> tuple:
    blocks = "\n\n".join(f"Фрагмент {i}: ({c['source']} — {c['section']})\n{c['text'][:700]}"
                         for i, c in enumerate(cands, 1))
    text, usage = llm.chat("Ты оцениваешь релевантность фрагментов.", RERANK_PROMPT.format(q=question, blocks=blocks))
    m = re.search(r"\[[\d\s.,]*\]", text)
    scores = json.loads(m.group(0)) if m else []
    if len(scores) != len(cands):  # модель ошиблась с форматом — не ломаем поиск, оставляем порядок
        for c in cands:
            c["rerank"], c["why"] = c["score"], "LLM-оценка не распознана, оставлена близость"
        return sorted(cands, key=lambda c: -c["rerank"]), usage
    for c, s in zip(cands, scores):
        c["rerank"] = float(s) / 10 + c["score"] * 0.01  # близость — только для развязки ничьих
        c["why"] = f"LLM: {s}/10, sim {c['score']:.3f}"
    return sorted(cands, key=lambda c: -c["rerank"]), usage


# ---------------------------------------------------------------------------
# Весь второй этап
# ---------------------------------------------------------------------------

@dataclass
class Mode:
    name: str
    rewrite: str = None        # None | "heuristic" | "llm"
    filter: bool = False       # порог отсечения
    rerank: str = None         # None | "heuristic" | "llm"
    k_before: int = 4
    k_after: int = 4


@dataclass
class Retrieved:
    chunks: list
    queries: list
    trace: dict = field(default_factory=dict)
    usage: dict = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0})


def _add_usage(total: dict, usage: dict) -> None:
    for k in total:
        total[k] += usage.get(k, 0)


def retrieve(index, question: str, mode: Mode, backend: str, cfg: dict, llm=None) -> Retrieved:
    usage = {"prompt_tokens": 0, "completion_tokens": 0}

    # 1. query rewrite
    queries = [question]
    if mode.rewrite == "heuristic":
        queries.append(rewrite_heuristic(question))
    elif mode.rewrite == "llm":
        rewritten, u = rewrite_llm(llm, question)
        _add_usage(usage, u)
        queries.append(rewritten)

    # 2. кандидаты: объединяем выдачу по всем вариантам запроса, берём лучшую близость.
    #    Если в вопросе указан день, который есть в базе, — дополнительно ищем внутри его файлов
    #    (фильтр по метаданным): иначе нужные чанки могут не попасть даже в кандидаты.
    day = mentioned_day(question) if mode.rerank else None
    prefix = f"day{day}-" if day is not None else None
    if prefix and not any(c["source"].startswith(prefix) for c in index.chunks):
        prefix = None
    in_day = [i for i, c in enumerate(index.chunks) if prefix and c["source"].startswith(prefix)]
    pool = {}
    for q in queries:
        scores = index.matrix @ index.embedder.embed([q])[0]
        picks = list(np.argsort(-scores)[:mode.k_before])
        if in_day:
            picks += sorted(in_day, key=lambda i: -scores[i])[:mode.k_before]
        for i in picks:
            c = {**index.chunks[i], "score": float(scores[i])}
            if c["chunk_id"] not in pool or c["score"] > pool[c["chunk_id"]]["score"]:
                pool[c["chunk_id"]] = c
    ranked = sorted(pool.values(), key=lambda c: -c["score"])
    day_hits = [c for c in ranked if prefix and c["source"].startswith(prefix)][:mode.k_before]
    others = [c for c in ranked if not (prefix and c["source"].startswith(prefix))]
    cands = sorted(day_hits + others[:max(0, mode.k_before - len(day_hits))] if prefix else ranked[:mode.k_before],
                   key=lambda c: -c["score"])
    trace = {"queries": queries, "candidates": len(cands), "dropped": [], "day_filter": prefix}

    # 3. порог отсечения
    if mode.filter and cands:
        th = cfg["thresholds"][backend]
        best = cands[0]["score"]
        kept = []
        for c in cands:
            if prefix and c["source"].startswith(prefix):
                kept.append(c)  # релевантность доказана метаданными — порог по близости не применяем
            elif c["score"] < th["min_score"]:
                trace["dropped"].append((c["chunk_id"], f"sim {c['score']:.4f} < порога {th['min_score']}"))
            elif c["score"] < th["rel_ratio"] * best:
                trace["dropped"].append((c["chunk_id"], f"sim {c['score']:.4f} < {th['rel_ratio']} × лучший {best:.4f}"))
            else:
                kept.append(c)
        cands = kept
    trace["after_filter"] = len(cands)

    # 4. реранкинг
    if cands and mode.rerank == "heuristic":
        cands = rerank_heuristic(question, cands, cfg["weights"])
    elif cands and mode.rerank == "llm":
        cands, u = rerank_llm(llm, question, cands)
        _add_usage(usage, u)

    # 5. финальный top-K с ограничением на число чанков из одного файла
    final, per_source = [], {}
    limit = cfg["max_per_source"] if mode.rerank else mode.k_after
    for c in cands:
        if per_source.get(c["source"], 0) >= limit:
            trace["dropped"].append((c["chunk_id"], f"уже {limit} чанка из {c['source']}"))
            continue
        final.append(c)
        per_source[c["source"]] = per_source.get(c["source"], 0) + 1
        if len(final) == mode.k_after:
            break
    trace["final"] = len(final)
    return Retrieved(final, queries, trace, usage)


MODES = {
    "baseline": Mode("baseline"),
    "rewrite": Mode("rewrite", rewrite="llm", k_before=4),
    "filter": Mode("filter", filter=True, rerank="heuristic", k_before=12),
    "rewrite+filter": Mode("rewrite+filter", rewrite="llm", filter=True, rerank="heuristic", k_before=12),
    "rewrite+filter+llm-rerank": Mode("rewrite+filter+llm-rerank", rewrite="llm", filter=True, rerank="llm", k_before=12),
}
