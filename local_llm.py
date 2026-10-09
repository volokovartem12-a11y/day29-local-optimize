"""
Дни 28–29. Локальная модель-генератор через Ollama + замер скорости любой модели.

OllamaLLM — тот же интерфейс, что у rag.LLM:  chat(system, user, history) → (текст, usage).
Всё идёт на http://localhost:11434, наружу ничего не уходит.

TimedLLM — обёртка над ЛЮБОЙ моделью (локальной или облачной): запоминает время и токены
каждого вызова, чтобы сравнивать скорость одинаково.
"""

import os
import re
import time

from embeddings import ollama_post, ollama_url

import json
import urllib.request

DEFAULT_LOCAL_MODEL = "qwen2.5:7b"
_WANT_JSON = re.compile(r"ТОЛЬКО JSON(?!-)")  # «JSON-массив» (реранк) под формат json не подходит


DEFAULT_OPTIONS = {"temperature": 0.2, "num_ctx": 8192, "num_predict": 900}  # настройки дня 28 = «до оптимизации»


class OllamaLLM:
    """options — параметры Ollama (temperature, top_k, top_p, repeat_penalty, seed, num_ctx, num_predict ...).
    fmt — формат ответа: "auto" (json, если промпт просит «ТОЛЬКО JSON»), "json", JSON-схема (dict) или None.
    options = {} — не передавать ничего: работают параметры, зашитые в Modelfile модели."""
    name = "local"

    def __init__(self, model: str = None, options: dict = None, fmt="auto", **legacy):
        self.model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_LOCAL_MODEL)
        self.options = dict(DEFAULT_OPTIONS if options is None else options)
        for k, v in legacy.items():  # совместимость: OllamaLLM(model, temperature=..., num_ctx=...)
            self.options[k] = v
        self.fmt = fmt
        self.timeout = float(os.environ.get("OLLAMA_TIMEOUT", "900"))
        self.last = {}

    @property
    def label(self) -> str:
        return f"ollama:{self.model}"

    def chat(self, system: str, user: str, history: list = None) -> tuple:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}] + (history or [])
        payload = {"model": self.model, "messages": messages, "stream": False, "keep_alive": "15m",
                   "options": self.options}
        if isinstance(self.fmt, dict):
            payload["format"] = self.fmt               # JSON-схема: Ollama генерирует строго по ней
        elif self.fmt == "json" or (self.fmt == "auto" and _WANT_JSON.search(system + user)):
            payload["format"] = "json"                 # Ollama принудительно отдаёт валидный JSON
        t0 = time.perf_counter()
        try:
            data = ollama_post("/api/chat", payload, timeout=self.timeout)
        except RuntimeError as e:
            if "HTTP 404" in str(e):
                raise RuntimeError(f"Модель {self.model} не скачана. Выполните: ollama pull {self.model}") from e
            raise
        wall = time.perf_counter() - t0
        ns = 1e9
        gen_tok = data.get("eval_count", 0) or 0
        gen_sec = (data.get("eval_duration", 0) or 0) / ns
        num_ctx = self.options.get("num_ctx")
        p_tok = data.get("prompt_eval_count", 0) or 0
        self.last = {
            "wall_sec": wall,
            "load_sec": (data.get("load_duration", 0) or 0) / ns,
            "prompt_eval_sec": (data.get("prompt_eval_duration", 0) or 0) / ns,
            "gen_sec": gen_sec,
            "tok_per_sec": (gen_tok / gen_sec) if gen_sec > 0 else None,
            "truncated_prompt": bool(num_ctx and p_tok >= num_ctx - 8),
            "hit_limit": data.get("done_reason") == "length",   # ответ оборван лимитом num_predict
        }
        text = ((data.get("message") or {}).get("content") or "").strip()
        return text, {"prompt_tokens": p_tok, "completion_tokens": gen_tok}


class TimedLLM:
    """Записывает каждый вызов: время, токены, ошибки. Ошибка вызова не глотается — она летит дальше."""

    def __init__(self, llm, name: str):
        self.llm, self.name = llm, name
        self.model = getattr(llm, "model", name)
        self.calls = []

    def reset(self) -> None:
        self.calls = []

    def chat(self, system: str, user: str, history: list = None) -> tuple:
        t0 = time.perf_counter()
        rec = {}
        try:
            text, usage = self.llm.chat(system, user, history)
        except Exception as e:  # noqa: BLE001
            self.calls.append({"wall_sec": time.perf_counter() - t0, "error": str(e)[:300]})
            raise
        rec.update({"wall_sec": time.perf_counter() - t0, **usage})
        extra = getattr(self.llm, "last", None)
        if extra:
            rec.update({k: v for k, v in extra.items() if k != "wall_sec"})
        self.calls.append(rec)
        return text, usage

    def summary(self) -> dict:
        ok = [c for c in self.calls if "error" not in c]
        out_tok = sum(c.get("completion_tokens", 0) for c in ok)
        gen = sum(c.get("gen_sec", 0) for c in ok)
        wall = sum(c["wall_sec"] for c in self.calls)
        return {
            "calls": len(self.calls),
            "gen_wall_sec": round(wall, 3),
            "load_sec": round(sum(c.get("load_sec", 0) for c in ok), 3),
            "prompt_tokens": sum(c.get("prompt_tokens", 0) for c in ok),
            "completion_tokens": out_tok,
            # у локальной модели — чистая скорость генерации (из счётчиков Ollama);
            # у облачной таких счётчиков нет — считаем «токены / время запроса» вместе с сетью
            "tok_per_sec": round(out_tok / gen, 2) if gen > 0 else (round(out_tok / wall, 2) if wall > 0 and out_tok else None),
            "tok_per_sec_kind": "generation" if gen > 0 else "wall(incl. network)",
            "truncated_prompt": any(c.get("truncated_prompt") for c in ok),
            "hit_limit": any(c.get("hit_limit") for c in ok),
            "prompt_eval_sec": round(sum(c.get("prompt_eval_sec", 0) for c in ok), 3),
        }


# ---------------------------------------------------------------------------
# Проверка среды
# ---------------------------------------------------------------------------

def ollama_get(path: str, timeout: float = 5) -> dict:
    with urllib.request.urlopen(ollama_url() + path, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ollama_models() -> list:
    """Имена скачанных моделей. Бросает RuntimeError, если сервер не отвечает."""
    try:
        return [m["name"] for m in ollama_get("/api/tags").get("models", [])]
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Ollama не отвечает на {ollama_url()}: {e}") from e


def has_model(installed: list, wanted: str) -> bool:
    base = wanted.split(":")[0]
    return any(n == wanted or n == f"{wanted}:latest" or (":" not in wanted and n.split(":")[0] == base)
               for n in installed)
