"""
День 9. Сравнение: без сжатия vs со сжатием — через реальный AITunnel API.

Прогоняет ОДИН и тот же сценарий диалога (планирование поездки, с деталями,
которые всплывают повторно — бюджет, аллергия и т.д.) дважды:
  1) без сжатия — каждый запрос содержит всю накопленную историю;
  2) со сжатием — каждый запрос содержит summary + последние N сообщений.

Считает реальный расход токенов (usage.prompt_tokens / completion_tokens из
ответов API) и сохраняет оба транскрипта в файлы для ручного сравнения
качества ответов.

Если AITUNNEL_API_KEY не задан — работает в MOCK-режиме (без реальных вызовов),
чтобы можно было проверить механику без расходов.

Запуск (PowerShell):

    cd day9-context-compression
    $env:AITUNNEL_API_KEY = "sk-aitunnel-xxx"
    python .\\compare.py

Внимание: скрипт делает реальные запросы к API и тратит токены на обоих
прогонах — по умолчанию сценарий укорочен до 20 реплик, чтобы ограничить
стоимость. При необходимости отредактируйте SIMULATED_TURNS.
"""

import os
import json

from context_manager import ContextManager, default_summarizer_factory, approx_tokens

USE_REAL_API = bool(os.environ.get("AITUNNEL_API_KEY"))

MODEL = os.environ.get("AITUNNEL_MODEL", "gpt-4o-mini")
SUMMARY_MODEL = os.environ.get("AITUNNEL_SUMMARY_MODEL", MODEL)
KEEP_LAST_N = int(os.environ.get("KEEP_LAST_N", "4"))
SUMMARIZE_EVERY = int(os.environ.get("SUMMARIZE_EVERY", "6"))

SYSTEM_PROMPT = "Ты — ассистент, который помогает планировать поездку и отвечать по деталям, упомянутым ранее."

SIMULATED_TURNS = [
    "Привет! Я хочу спланировать поездку в Японию на 10 дней в апреле.",
    "Бюджет — примерно 3000 долларов на человека, нас двое.",
    "Мы хотим увидеть Токио, Киото и Осаку.",
    "Я не ем морепродукты, учти это при советах по еде.",
    "У жены аллергия на арахис, важно для ресторанов.",
    "Хотим попробовать что-то необычное — например, остаться в рёкане.",
    "Сколько дней ты бы выделил на Осаку?",
    "Стоит ли покупать JR Pass при таком маршруте?",
    "Хотим один день выделить на дейтрип, например в Нару.",
    "Напомни, какой у нас бюджет на человека?",
    "Помнишь про аллергию на арахис? Учитывай в ресторанных рекомендациях.",
    "Стоит ли бронировать рестораны заранее в Киото?",
    "Напомни, какие города у нас в маршруте?",
    "Какие вещи взять с собой в апреле по погоде?",
    "Напомни про морепродукты — я вроде говорил, что не ем.",
    "Стоит ли брать с собой наличные или карты работают везде?",
    "Хотим взять один день на шопинг в Токио перед вылетом.",
    "Напомни весь бюджет — сколько всего на двоих получается?",
    "Мы должны что-то забронировать заранее уже сейчас?",
    "Подведи итог всего плана поездки одним сообщением.",
]

if USE_REAL_API:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["AITUNNEL_API_KEY"], base_url="https://api.aitunnel.ru/v1/")


def real_answer(messages, system):
    full_messages = [{"role": "system", "content": system}] + list(messages)
    resp = client.chat.completions.create(model=MODEL, max_tokens=500, messages=full_messages)
    return resp.choices[0].message.content, resp.usage.prompt_tokens, resp.usage.completion_tokens


def real_summarize(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=SUMMARY_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def mock_answer(messages, system):
    blob = system + " " + " ".join(m["content"] for m in messages)
    return f"[mock-ответ, контекст ~{approx_tokens(blob)} ток.] {messages[-1]['content'][:60]}", approx_tokens(blob), 20


def mock_summarize(prompt: str) -> str:
    return "[MOCK SUMMARY] " + prompt[:200].replace("\n", " ") + " ..."


def run_without_compression():
    history = []
    total_in, total_out = 0, 0
    transcript = []
    for turn in SIMULATED_TURNS:
        history.append({"role": "user", "content": turn})
        if USE_REAL_API:
            answer, in_tok, out_tok = real_answer(history, SYSTEM_PROMPT)
        else:
            answer, in_tok, out_tok = mock_answer(history, SYSTEM_PROMPT)
        total_in += in_tok
        total_out += out_tok
        history.append({"role": "assistant", "content": answer})
        transcript.append({"user": turn, "assistant": answer})
    return total_in, total_out, transcript


def run_with_compression():
    cm = ContextManager(keep_last_n=KEEP_LAST_N, summarize_every=SUMMARIZE_EVERY)
    cm.summarizer = default_summarizer_factory(real_summarize if USE_REAL_API else mock_summarize)

    total_in, total_out = 0, 0
    transcript = []
    for turn in SIMULATED_TURNS:
        cm.add_message("user", turn)
        ctx = cm.get_context_compressed(SYSTEM_PROMPT)
        if USE_REAL_API:
            answer, in_tok, out_tok = real_answer(ctx["messages"], ctx["system"])
        else:
            answer, in_tok, out_tok = mock_answer(ctx["messages"], ctx["system"])
        total_in += in_tok
        total_out += out_tok
        cm.add_message("assistant", answer)
        transcript.append({"user": turn, "assistant": answer})
    return total_in, total_out, transcript, cm.summary_calls, cm.summary


def main():
    print(f"Режим: {'REAL API (AITunnel, модель ' + MODEL + ')' if USE_REAL_API else 'MOCK (без AITUNNEL_API_KEY)'}")
    print(f"Реплик в сценарии: {len(SIMULATED_TURNS)}\n")

    in_no, out_no, transcript_no = run_without_compression()
    in_c, out_c, transcript_c, n_summaries, final_summary = run_with_compression()

    saved = in_no - in_c
    saved_pct = 100 * saved / in_no if in_no else 0

    print("=" * 60)
    print("СРАВНЕНИЕ РАСХОДА ТОКЕНОВ (реальные usage.prompt_tokens из API)")
    print("=" * 60)
    print(f"Без сжатия — input:   {in_no:>8}   output: {out_no:>6}")
    print(f"Со сжатием — input:   {in_c:>8}   output: {out_c:>6}")
    print(f"Экономия input:       {saved:>8} ток.  ({saved_pct:.1f}%)")
    print(f"Вызовов суммаризации: {n_summaries}")
    print()

    with open("transcript_no_compression.json", "w", encoding="utf-8") as f:
        json.dump(transcript_no, f, ensure_ascii=False, indent=2)
    with open("transcript_with_compression.json", "w", encoding="utf-8") as f:
        json.dump({"summary": final_summary, "turns": transcript_c}, f, ensure_ascii=False, indent=2)

    print("Транскрипты сохранены в transcript_no_compression.json и transcript_with_compression.json")
    print("Сравните их вручную — не потерялись ли детали (бюджет, аллергия и т.д.) в версии со сжатием.")


if __name__ == "__main__":
    main()
