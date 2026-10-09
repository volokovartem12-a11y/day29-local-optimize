"""
День 24. Ответ с обязательными источниками и цитатами + режим «не знаю».

    вопрос ─► ДЕНЬ В ВОПРОСЕ: упомянут день, которого нет в базе? ─► «Не знаю» + список дней
           ─► ПОРОГ РЕЛЕВАНТНОСТИ (если день не упомянут): лучшая близость < answer_min?
           │      └─ да ─► «Не знаю» + уточняющий вопрос (модель не вызывается)
           │  (если упомянут существующий день — релевантность доказана метаданными,
           │   порог по близости не применяется)
           ─► поиск (rewrite + фильтр + реранкер из дня 23) ─► пусто? ─► «Не знаю» + уточнение
           ─► модель возвращает JSON: ответ, номера фрагментов, ДОСЛОВНЫЕ цитаты
           ─► проверка кодом: каждая цитата должна дословно найтись в своём фрагменте,
              источники собираются кодом по номерам (source + section + chunk_id)
           ─► есть проблемы? ─► одна повторная попытка с перечнем замечаний
           ─► всё ещё нет подтверждённых цитат? ─► «Не знаю: ответ не подтверждён»

Итог: ответ без источников и цитат пользователь не получает никогда.
"""

import json
import re
from dataclasses import dataclass, field

from rag import build_prompt
from retrieval import MODES, mentioned_day, retrieve

GROUNDED_SYSTEM = """Ты отвечаешь на вопросы ТОЛЬКО по фрагментам из раздела «Контекст».
Верни ТОЛЬКО JSON, без markdown и пояснений:
{"status": "answered" или "unknown",
 "answer": "краткий ответ на русском со ссылками на фрагменты [n]",
 "sources": [номера фрагментов, на которых основан ответ],
 "quotes": [{"chunk": номер фрагмента, "text": "дословная цитата из этого фрагмента"}],
 "clarification": "уточняющий вопрос пользователю — только если status = unknown"}

Правила:
1. Каждое утверждение ответа подтверждай хотя бы одной цитатой.
2. Цитату копируй из фрагмента символ в символ: без перефразирования, без перевода,
   20–300 символов, одно–два предложения или строка кода.
3. Числа, названия функций, файлов и терминов в ответе бери только из цитат.
4. Если во фрагментах нет ответа на вопрос — status = "unknown", answer = "Не знаю",
   quotes = [], и задай в clarification вопрос, который поможет уточнить запрос."""

UNKNOWN_TEXT = "Не знаю: в базе не нашлось достаточно релевантных фрагментов, чтобы ответить."
UNVERIFIED_TEXT = "Не знаю: не удалось подтвердить ответ дословными цитатами из базы."


@dataclass
class Grounded:
    question: str
    status: str                      # answered | unknown | unverified
    answer: str
    sources: list = field(default_factory=list)   # [{n, source, section, chunk_id, quoted}]
    quotes: list = field(default_factory=list)    # [{n, text, source, section}]
    clarification: str = ""
    reason: str = ""                 # почему «не знаю»: no_day | gate | empty | model | unverified
    rejected_quotes: list = field(default_factory=list)
    attempts: int = 0
    first_try_ok: bool = False       # ответ прошёл проверку с первой попытки (без повтора)
    raw_answer: str = ""             # исходный ответ модели, если его пришлось заменить
    top_score: float = 0.0
    gate: float = 0.0
    chunks: list = field(default_factory=list)
    trace: dict = field(default_factory=dict)
    usage: dict = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0})


# ---------------------------------------------------------------------------
# Проверка цитат
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    t = text.lower().replace("ё", "е")
    t = re.sub(r"[«»“”„\"'`]", '"', t)
    t = re.sub(r"[—–]", "-", t)
    return re.sub(r"\s+", " ", t).strip()


def quote_in_chunk(quote: str, chunk_text: str) -> tuple:
    """Цитата должна дословно (с точностью до пробелов, кавычек, регистра) встречаться во фрагменте.
    Пропуск внутри цитаты можно обозначить «...» — тогда части ищутся по порядку."""
    q = normalize(quote).strip(' ."…')
    if len(q) < 15:
        return False, "слишком короткая цитата (< 15 символов)"
    if len(re.findall(r"[a-zа-я0-9]{3,}", q)) < 2:
        return False, "в цитате нет содержательного текста"
    hay = normalize(chunk_text)
    pos = 0
    for part in [p.strip(' ."') for p in re.split(r"\s*(?:\.\.\.|…)\s*", q) if p.strip(' ."')]:
        found = hay.find(part, pos)
        if found < 0:
            return False, "нет во фрагменте дословно"
        pos = found + len(part)
    return True, ""


def parse_json(raw: str):
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def validate(obj, chunks: list) -> tuple:
    """→ (разобранный ответ, список проблем). Источники строятся кодом, а не берутся у модели."""
    if obj is None:
        return None, ["ответ не в формате JSON"]
    problems, valid, rejected = [], [], []
    status = obj.get("status")
    if status not in ("answered", "unknown"):
        problems.append('поле status должно быть "answered" или "unknown"')
        status = "answered"
    answer = str(obj.get("answer") or "").strip()
    if not answer:
        problems.append("пустое поле answer")

    for q in obj.get("quotes") or []:
        n, text = (q.get("chunk"), str(q.get("text") or "")) if isinstance(q, dict) else (None, str(q))
        if not isinstance(n, int) or not 1 <= n <= len(chunks):
            rejected.append({"n": n, "text": text, "why": "неверный номер фрагмента"})
            continue
        ok, why = quote_in_chunk(text, chunks[n - 1]["text"])
        if ok:
            valid.append({"n": n, "text": text.strip(), "source": chunks[n - 1]["source"],
                          "section": chunks[n - 1]["section"]})
        else:
            rejected.append({"n": n, "text": text, "why": why})

    for r in rejected:
        problems.append(f"цитата {r['text'][:60]!r} к фрагменту [{r['n']}]: {r['why']}")
    if status == "answered" and not valid:
        problems.append("нет ни одной дословной цитаты — ответ нечем подтвердить")

    declared = {n for n in obj.get("sources") or [] if isinstance(n, int) and 1 <= n <= len(chunks)}
    quoted = {q["n"] for q in valid}
    sources = [{"n": n, "source": chunks[n - 1]["source"], "section": chunks[n - 1]["section"],
                "chunk_id": chunks[n - 1]["chunk_id"], "quoted": n in quoted}
               for n in sorted(declared | quoted)]
    if status == "answered" and not sources:
        problems.append("пустой список источников")
    return {"status": status, "answer": answer, "quotes": valid, "sources": sources, "rejected": rejected,
            "clarification": str(obj.get("clarification") or "").strip()}, problems


# ---------------------------------------------------------------------------
# Уточняющий вопрос при «не знаю»
# ---------------------------------------------------------------------------

def suggest_clarification(index, question: str) -> str:
    """Подсказка без модели: ближайшие темы базы по заголовкам документов."""
    titles = []
    for c in index.search(question, 8):
        t = c["title"].strip()
        if t and t not in titles:
            titles.append(t)
        if len(titles) == 3:
            break
    hint = "; ".join(f"«{t[:70]}»" for t in titles)
    return (f"Уточните, пожалуйста, вопрос: о каком проекте или дне челленджа идёт речь и что именно "
            f"нужно узнать? Ближайшие темы в базе: {hint}.")


# ---------------------------------------------------------------------------
# Основная функция
# ---------------------------------------------------------------------------

def _add(total: dict, usage: dict) -> None:
    for k in total:
        total[k] += usage.get(k, 0)


def prepare(index, question: str, cfg: dict, backend: str, mode: str = "rewrite+filter", llm=None,
            force: bool = False) -> tuple:
    """Шаги 1–3: проверка дня, порог «не знаю», поиск. → (Grounded, chunks | None).
    chunks = None означает, что отказ уже принят и модель не нужна.
    force=True (стресс-тест) пропускает отказы по дню и порогу и, если фильтр отсёк всё,
    отдаёт модели «сырые» top-K — чтобы проверить, как модель ведёт себя на слабом контексте."""
    gate = cfg["thresholds"][backend].get("answer_min", 0.0)
    top = index.search(question, 1)
    top_score = top[0]["score"] if top else 0.0
    res = Grounded(question, "unknown", UNKNOWN_TEXT, top_score=round(top_score, 4), gate=gate)

    # 1. Метаданные: упомянут день, которого нет в базе → «не знаю» сразу
    day = mentioned_day(question)
    known = sorted({int(m.group(1)) for c in index.chunks for m in [re.match(r"day(\d+)-", c["source"])] if m})
    if day is not None and day not in known and not force:
        res.reason = "no_day"
        res.clarification = (f"В базе нет проекта дня {day}. Есть дни: {', '.join(map(str, known))}. "
                             f"Уточните, пожалуйста, какой проект вас интересует?")
        return res, None

    # 2. Порог релевантности (если день не указан): слабый контекст → «не знаю», модель не вызывается
    if day is None and top_score < gate and not force:
        res.reason = "gate"
        res.clarification = suggest_clarification(index, question)
        return res, None

    # 3. Поиск: rewrite + порог отсечения + реранкер (день 23)
    r = retrieve(index, question, MODES[mode], backend, cfg, llm)
    _add(res.usage, r.usage)
    res.chunks, res.trace = r.chunks, r.trace
    chunks = r.chunks
    if not chunks and force:
        chunks = index.search(question, MODES[mode].k_after)  # стресс: слабый контекст всё же отдаём модели
        res.chunks = chunks
        res.trace = {**res.trace, "forced_weak_context": True}
    if not chunks:
        res.reason = "empty"
        res.clarification = suggest_clarification(index, question)
        return res, None
    return res, chunks


def generate(llm, index, question: str, res: Grounded, chunks: list,
             system: str = None, prompt_fn=None) -> Grounded:
    """Шаги 4–5: ответ модели по ЗАДАННОМУ контексту + проверка цитат + одна повторная попытка.
    system / prompt_fn — шаблон промпта (день 29); по умолчанию — шаблон дня 24."""
    system = system or GROUNDED_SYSTEM
    prompt = (prompt_fn or build_prompt)(question, chunks)
    raw, usage = llm.chat(system, prompt)
    _add(res.usage, usage)
    parsed, problems = validate(parse_json(raw), chunks)
    res.attempts = 1
    res.first_try_ok = not problems
    if problems:
        feedback = ("Ответ не прошёл проверку:\n- " + "\n- ".join(problems) +
                    "\nИсправь и верни только JSON. Цитаты копируй из фрагментов дословно.")
        raw2, usage2 = llm.chat(system, prompt, history=[{"role": "assistant", "content": raw},
                                                                  {"role": "user", "content": feedback}])
        _add(res.usage, usage2)
        res.attempts = 2
        parsed2, problems2 = validate(parse_json(raw2), chunks)
        if parsed2 is not None and (parsed is None or len(problems2) <= len(problems)):
            parsed, problems, raw = parsed2, problems2, raw2

    if parsed is None:
        res.status, res.reason, res.answer = "unverified", "unverified", UNVERIFIED_TEXT
        res.raw_answer = raw
        res.clarification = suggest_clarification(index, question)
        return res

    res.rejected_quotes = parsed["rejected"]
    if parsed["status"] == "unknown":
        res.reason = "model"
        res.answer = parsed["answer"] or UNKNOWN_TEXT
        res.clarification = parsed["clarification"] or suggest_clarification(index, question)
        return res

    # 5. Обязательность: без подтверждённых цитат ответ не выдаётся
    if not parsed["quotes"] or not parsed["sources"]:
        res.status, res.reason, res.answer = "unverified", "unverified", UNVERIFIED_TEXT
        res.raw_answer = parsed["answer"]
        res.clarification = suggest_clarification(index, question)
        return res

    res.status, res.answer, res.reason = "answered", parsed["answer"], ""
    res.quotes, res.sources = parsed["quotes"], parsed["sources"]
    return res


def grounded_answer(llm, index, question: str, cfg: dict, backend: str, mode: str = "rewrite+filter") -> Grounded:
    res, chunks = prepare(index, question, cfg, backend, mode, llm)
    if chunks is None:
        return res
    return generate(llm, index, question, res, chunks)


def render(g: Grounded) -> str:
    """Ответ в виде, который видит пользователь."""
    lines = [f"Ответ: {g.answer}"]
    if g.status == "answered":
        lines.append("\nИсточники:")
        lines += [f"  [{s['n']}] {s['source']} — {s['section']}  ({s['chunk_id']})"
                  + ("" if s["quoted"] else "  · без цитаты") for s in g.sources]
        lines.append("\nЦитаты:")
        lines += [f"  [{q['n']}] «{q['text']}»" for q in g.quotes]
        note = f"\nПроверка: все цитаты ({len(g.quotes)}) найдены во фрагментах дословно ✅"
        if g.rejected_quotes:
            note += f"; отброшено неточных цитат: {len(g.rejected_quotes)}"
        lines.append(note)
    else:
        if g.clarification:
            lines.append(f"\n{g.clarification}")
        why = {"gate": f"лучшая близость {g.top_score:.3f} ниже порога {g.gate}",
               "no_day": "такого дня нет в базе",
               "empty": "фильтр отсёк все фрагменты",
               "model": "модель не нашла ответа во фрагментах",
               "unverified": "ответ модели не подтверждён цитатами"}.get(g.reason, "")
        if why:
            lines.append(f"(причина: {why})")
    return "\n".join(lines)
