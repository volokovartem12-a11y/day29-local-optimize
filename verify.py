"""
День 24. Совпадает ли смысл ответа с цитатами — две независимые проверки.

1. Детерминированная (без модели): ключевые элементы ответа — числа, имена
   функций и файлов, идентификаторы, латинские термины — должны встречаться
   в цитатах. Если ответ называет число или функцию, которой нет в цитатах,
   это утверждение не подтверждено.

2. LLM-судья: отдельный запрос «подтверждают ли цитаты каждое утверждение
   ответа?» → supported / partial / unsupported + список неподтверждённых
   утверждений. Ловит смысловые расхождения, которые не видны по словам.
"""

import json
import re

from embeddings import _tokens
from grounded import normalize

_CITE = re.compile(r"\[[\d,\s]+\]")
_NUM = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?%?")
_IDENT = re.compile(r"`([^`]+)`|\b([A-Za-z_][\w./-]*(?:_|\.|/)[\w./-]+|[a-z]+[A-Z]\w*|[A-Za-z]{4,}\d*)\b")
_STOP_LATIN = {"json", "true", "false", "none", "null"}
# служебная обвязка ответа — не утверждения о фактах («согласно базе», «в источнике указано»)
_FILLER = {"соглас", "базе", "данным", "фрагме", "источн", "ответ", "указан", "указыв", "говори", "контек",
           "цитат", "приве", "привод", "описан", "описыв", "следуе", "являет", "также", "именно", "поэтому",
           "итоге", "образо", "сказан", "отмече", "упомин", "проект", "проекте", "проекта"}


def key_terms(answer: str) -> list:
    text = _CITE.sub(" ", answer)
    terms = []
    for n in _NUM.findall(text):
        if len(n.rstrip("%")) > 1 or "%" in n:  # одиночные цифры («3 стратегии») слишком шумные
            terms.append(n)
    for a, b in _IDENT.findall(text):
        t = (a or b).strip(".,;:")
        if t and t.lower() not in _STOP_LATIN:
            terms.append(t)
    seen, out = set(), []
    for t in terms:
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


def _num_variants(n: str) -> set:
    base = n.rstrip("%")
    return {base, base.replace(",", "."), base.replace(".", ",")}


def lexical_grounding(answer: str, quotes: list) -> dict:
    """Доля ключевых элементов ответа, найденных в цитатах."""
    hay = normalize(" ".join(q["text"] for q in quotes))
    hay_digits = hay.replace(" ", "")
    terms = key_terms(answer)
    if terms:
        unsupported = []
        for t in terms:
            if _NUM.fullmatch(t):
                ok = any(v in hay_digits for v in _num_variants(t))
            else:
                ok = normalize(t) in hay
            if not ok:
                unsupported.append(t)
        score = 1 - len(unsupported) / len(terms)
        return {"method": "ключевые элементы", "score": round(score, 2), "checked": terms, "unsupported": unsupported}
    # в ответе нет чисел и имён — сравниваем значимые слова
    a = {t for t in _tokens(_CITE.sub(" ", answer)) if len(t) > 3 and t not in _FILLER}
    if not a:
        return {"method": "нечего проверять", "score": None, "checked": [], "unsupported": []}
    q = set(_tokens(hay))
    score = len(a & q) / len(a)
    return {"method": "значимые слова", "score": round(score, 2), "checked": sorted(a), "unsupported": sorted(a - q)}


JUDGE_PROMPT = """Проверь, подтверждают ли цитаты ответ. Оценивай ТОЛЬКО по цитатам, без своих знаний.

Вопрос: {question}

Ответ:
{answer}

Цитаты:
{quotes}

Верни ТОЛЬКО JSON:
{{"verdict": "supported" | "partial" | "unsupported",
 "unsupported_claims": ["утверждения ответа, которых нет в цитатах"],
 "comment": "одно предложение"}}
supported — каждое утверждение ответа следует из цитат; partial — часть утверждений не подтверждена;
unsupported — ответ противоречит цитатам или не следует из них."""


def judge(llm, question: str, answer: str, quotes: list) -> dict:
    qtext = "\n".join(f"[{q['n']}] «{q['text']}»" for q in quotes) or "(цитат нет)"
    raw, usage = llm.chat("Ты строгий проверяющий фактов.", JUDGE_PROMPT.format(
        question=question, answer=answer, quotes=qtext))
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        obj = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        obj = {}
    verdict = obj.get("verdict")
    if verdict not in ("supported", "partial", "unsupported"):
        verdict = "error"
    return {"verdict": verdict, "unsupported_claims": obj.get("unsupported_claims") or [],
            "comment": obj.get("comment") or ("ответ судьи не распознан" if verdict == "error" else ""),
            "usage": usage}
