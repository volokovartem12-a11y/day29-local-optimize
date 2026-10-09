"""
День 9. Агент с компрессией истории — интерактивный чат через AITunnel.

Использует тот же принцип, что и в day7-context-memory, но добавляет
управление контекстом: последние N сообщений хранятся как есть, более
старые каждые SUMMARIZE_EVERY сообщений сворачиваются в summary.

Запуск (PowerShell), аналогично day7:

    cd day9-context-compression
    $env:AITUNNEL_API_KEY = "sk-aitunnel-xxx"
    python .\\chat.py

Команды внутри чата:
    /stats   — показать расход токенов и оценку экономии от сжатия
    /summary — показать текущее накопленное summary
    /exit    — выход
"""

import os
import sys

from openai import OpenAI

from context_manager import ContextManager, default_summarizer_factory, approx_tokens

API_KEY = os.environ.get("AITUNNEL_API_KEY")
if not API_KEY:
    print("Ошибка: переменная окружения AITUNNEL_API_KEY не задана.")
    print('Установите её перед запуском: $env:AITUNNEL_API_KEY = "sk-aitunnel-xxx"')
    sys.exit(1)

BASE_URL = "https://api.aitunnel.ru/v1/"

# Модель для основных ответов и модель для суммаризации можно задать раздельно —
# например, для сжатия истории использовать более дешёвую модель.
MODEL = os.environ.get("AITUNNEL_MODEL", "gpt-4o-mini")
SUMMARY_MODEL = os.environ.get("AITUNNEL_SUMMARY_MODEL", MODEL)

KEEP_LAST_N = int(os.environ.get("KEEP_LAST_N", "6"))
SUMMARIZE_EVERY = int(os.environ.get("SUMMARIZE_EVERY", "10"))

SYSTEM_PROMPT = "Ты — полезный ассистент. Отвечай кратко и по делу, на русском языке."

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def llm_summarize(prompt: str) -> str:
    """Вызов модели для суммаризации старой части истории."""
    resp = client.chat.completions.create(
        model=SUMMARY_MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def ask_model(messages, system_prompt):
    """Основной вызов модели с уже собранным (сжатым) контекстом."""
    full_messages = [{"role": "system", "content": system_prompt}] + list(messages)
    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=800,
        messages=full_messages,
    )
    return resp.choices[0].message.content, resp.usage


def print_stats(cm, total_in, total_out, total_estimate_no_compression):
    print(f"— Реальных input-токенов потрачено (со сжатием):  {total_in}")
    print(f"— Реальных output-токенов потрачено:               {total_out}")
    print(f"— Оценка input-токенов БЕЗ сжатия (что было бы):   {total_estimate_no_compression}")
    if total_estimate_no_compression > 0:
        saved = total_estimate_no_compression - total_in
        pct = 100 * saved / total_estimate_no_compression
        print(f"— Экономия:                                        {saved} ток. (~{pct:.1f}%)")
    print(f"— Вызовов суммаризации:                            {cm.summary_calls}")


def main():
    cm = ContextManager(keep_last_n=KEEP_LAST_N, summarize_every=SUMMARIZE_EVERY)
    cm.summarizer = default_summarizer_factory(llm_summarize)

    total_real_input_tokens = 0
    total_real_output_tokens = 0
    total_estimate_no_compression = 0

    print("Агент с компрессией истории (AITunnel).")
    print(f"Модель: {MODEL} | keep_last_n={KEEP_LAST_N} | summarize_every={SUMMARIZE_EVERY}")
    print("Команды: /stats, /summary, /exit\n")

    while True:
        try:
            user_input = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nЗавершение.")
            break

        if not user_input:
            continue
        if user_input == "/exit":
            break
        if user_input == "/stats":
            print_stats(cm, total_real_input_tokens, total_real_output_tokens, total_estimate_no_compression)
            continue
        if user_input == "/summary":
            print("Текущее summary:")
            print(cm.summary or "(пока пусто)")
            continue

        cm.add_message("user", user_input)

        # Локальная оценка "что было бы без сжатия" — без реального вызова API,
        # чтобы не тратить деньги на параллельный запрос.
        total_estimate_no_compression += approx_tokens(SYSTEM_PROMPT) + \
            sum(approx_tokens(m["content"]) for m in cm.full_history)

        ctx = cm.get_context_compressed(SYSTEM_PROMPT)
        answer, usage = ask_model(ctx["messages"], ctx["system"])

        total_real_input_tokens += usage.prompt_tokens
        total_real_output_tokens += usage.completion_tokens

        cm.add_message("assistant", answer)
        print(f"Агент: {answer}\n")


if __name__ == "__main__":
    main()
