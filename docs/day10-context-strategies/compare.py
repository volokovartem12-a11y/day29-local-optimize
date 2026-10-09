"""
День 10. Сравнение 3 стратегий управления контекстом на одном сценарии.

Сценарий: сбор ТЗ на сайт бронирования антикафе, 9 базовых реплик + продолжение.

- Sliding Window и Facts прогоняются на полном линейном сценарии (14 реплик,
  включая контрольные вопросы на память: стек, бюджет, дедлайн, регистрация).
- Branching прогоняет те же 9 базовых реплик, затем ставит checkpoint и
  расходится в 2 независимые ветки с разным продолжением (добавить мобильное
  приложение / урезать фичи из-за сроков) — обе ветки заканчиваются вопросом
  "что там было в начале", чтобы проверить, не теряются ли факты до чекпоинта.

Реальный расход токенов берётся из usage.prompt_tokens / completion_tokens
ответов API. Без AITUNNEL_API_KEY скрипт уходит в MOCK-режим (без реальных
вызовов) — чтобы проверить механику бесплатно.

Запуск (PowerShell):

    cd day10-context-strategies
    $env:AITUNNEL_API_KEY = "sk-aitunnel-xxx"
    python .\\compare.py
"""

import os
import json

from context_strategies import (
    SlidingWindowStrategy,
    FactsStrategy,
    BranchingStrategy,
    default_facts_updater_factory,
    approx_tokens,
)

USE_REAL_API = bool(os.environ.get("AITUNNEL_API_KEY"))
MODEL = os.environ.get("AITUNNEL_MODEL", "gpt-4o-mini")
FACTS_MODEL = os.environ.get("AITUNNEL_FACTS_MODEL", MODEL)
WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "6"))
FACTS_WINDOW_SIZE = int(os.environ.get("FACTS_WINDOW_SIZE", "4"))

SYSTEM_PROMPT = "Ты — ассистент, который помогает собрать ТЗ на проект. Отвечай по делу, на русском."

BASE_TURNS = [
    "Привет! Хочу обсудить ТЗ для нового проекта — сайт для бронирования настольных игр в антикафе.",
    "Цель — чтобы посетители сами бронировали столик и игру онлайн, без звонков администратору.",
    "Бюджет на разработку — около 400000 рублей, дедлайн — 2 месяца.",
    "Аудитория — в основном молодёжь 18-30 лет, много мобильных пользователей.",
    "Стек предпочитаем: бэкенд на Python (FastAPI), фронт — React.",
    "Обязательно нужна интеграция с онлайн-оплатой, чтобы бронь можно было подтвердить предоплатой.",
    "Решили, что регистрация не обязательна — можно бронировать по номеру телефона с SMS-подтверждением.",
    "Админка нужна отдельная — управлять расписанием столов и играми.",
    "Из ограничений: хостинг уже есть, это VPS на 4 ГБ RAM, сильно перегружать не хотим.",
]

LINEAR_CONTINUATION = [
    "Напомни, какой у нас стек и бюджет?",
    "Ещё хотим базовую аналитику — сколько броней в день, какие игры популярнее.",
    "Помнишь, что мы говорили про регистрацию пользователей?",
    "Какой дедлайн мы обсуждали в самом начале?",
    "Подведи итог всего ТЗ одним сообщением — цель, стек, бюджет, сроки, ключевые фичи.",
]

BRANCH_A_TURNS = [  # добавляем мобильное приложение
    "Кстати, обязательно нужно ещё и мобильное приложение, а не только сайт.",
    "Для приложения — Flutter, чтобы сразу под iOS и Android.",
    "Как это меняет бюджет и сроки, которые мы обсуждали в начале?",
]

BRANCH_B_TURNS = [  # урезаем фичи, сроки поджимают
    "На самом деле сроки поджимают, давай уберём часть фич для первой версии.",
    "Уберём онлайн-оплату — на старте бронь будет без предоплаты, оплата на месте.",
    "Как тогда меняется дедлайн, который мы обсуждали в начале?",
]

if USE_REAL_API:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["AITUNNEL_API_KEY"], base_url="https://api.aitunnel.ru/v1/")


def real_answer(messages, system):
    full_messages = [{"role": "system", "content": system}] + list(messages)
    resp = client.chat.completions.create(model=MODEL, max_tokens=600, messages=full_messages)
    return resp.choices[0].message.content, resp.usage.prompt_tokens, resp.usage.completion_tokens


def real_facts_llm(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=FACTS_MODEL, max_tokens=250, messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content


def mock_answer(messages, system):
    blob = system + " " + " ".join(m["content"] for m in messages)
    return f"[mock-ответ, контекст ~{approx_tokens(blob)} ток.] {messages[-1]['content'][:60]}", approx_tokens(blob), 25


def mock_facts_llm(prompt: str) -> str:
    return "{}"  # в mock-режиме фактов не извлекаем, только проверяем механику вызовов


def run_sliding_window():
    strat = SlidingWindowStrategy(window_size=WINDOW_SIZE)
    total_in, total_out = 0, 0
    transcript = []
    for turn in BASE_TURNS + LINEAR_CONTINUATION:
        strat.add_message("user", turn)
        ctx = strat.get_context(SYSTEM_PROMPT)
        if USE_REAL_API:
            answer, in_tok, out_tok = real_answer(ctx["messages"], ctx["system"])
        else:
            answer, in_tok, out_tok = mock_answer(ctx["messages"], ctx["system"])
        total_in += in_tok
        total_out += out_tok
        strat.add_message("assistant", answer)
        transcript.append({"user": turn, "assistant": answer})
    return {"input_tokens": total_in, "output_tokens": total_out, "transcript": transcript}


def run_facts():
    strat = FactsStrategy(window_size=FACTS_WINDOW_SIZE)
    strat.facts_updater = default_facts_updater_factory(real_facts_llm if USE_REAL_API else mock_facts_llm)
    total_in, total_out = 0, 0
    transcript = []
    for turn in BASE_TURNS + LINEAR_CONTINUATION:
        strat.add_message("user", turn)
        ctx = strat.get_context(SYSTEM_PROMPT)
        if USE_REAL_API:
            answer, in_tok, out_tok = real_answer(ctx["messages"], ctx["system"])
        else:
            answer, in_tok, out_tok = mock_answer(ctx["messages"], ctx["system"])
        total_in += in_tok
        total_out += out_tok
        strat.add_message("assistant", answer)
        transcript.append({"user": turn, "assistant": answer, "facts_so_far": dict(strat.facts)})
    return {
        "input_tokens": total_in,
        "output_tokens": total_out,
        "facts_update_calls": strat.facts_update_calls,
        "final_facts": strat.facts,
        "transcript": transcript,
    }


def run_branching():
    strat = BranchingStrategy()
    total_in, total_out = 0, 0
    main_transcript = []

    for turn in BASE_TURNS:
        strat.add_message("user", turn)
        ctx = strat.get_context(SYSTEM_PROMPT)
        if USE_REAL_API:
            answer, in_tok, out_tok = real_answer(ctx["messages"], ctx["system"])
        else:
            answer, in_tok, out_tok = mock_answer(ctx["messages"], ctx["system"])
        total_in += in_tok
        total_out += out_tok
        strat.add_message("assistant", answer)
        main_transcript.append({"user": turn, "assistant": answer})

    strat.checkpoint("after-core-tz")
    strat.create_branch("mobile-app", from_checkpoint="after-core-tz")
    strat.create_branch("scope-cut", from_checkpoint="after-core-tz")

    def run_branch(name, turns):
        nonlocal total_in, total_out
        strat.switch(name)
        transcript = []
        for turn in turns:
            strat.add_message("user", turn, branch=name)
            ctx = strat.get_context(SYSTEM_PROMPT, branch=name)
            if USE_REAL_API:
                answer, in_tok, out_tok = real_answer(ctx["messages"], ctx["system"])
            else:
                answer, in_tok, out_tok = mock_answer(ctx["messages"], ctx["system"])
            total_in += in_tok
            total_out += out_tok
            strat.add_message("assistant", answer, branch=name)
            transcript.append({"user": turn, "assistant": answer})
        return transcript

    branch_a_transcript = run_branch("mobile-app", BRANCH_A_TURNS)
    branch_b_transcript = run_branch("scope-cut", BRANCH_B_TURNS)

    return {
        "input_tokens": total_in,
        "output_tokens": total_out,
        "main_transcript": main_transcript,
        "branch_mobile_app": branch_a_transcript,
        "branch_scope_cut": branch_b_transcript,
    }


def main():
    print(f"Режим: {'REAL API (AITunnel, модель ' + MODEL + ')' if USE_REAL_API else 'MOCK (без AITUNNEL_API_KEY)'}")
    print(f"Базовых реплик: {len(BASE_TURNS)}, линейное продолжение: {len(LINEAR_CONTINUATION)}, "
          f"по 3 реплики в каждой из 2 веток\n")

    res_window = run_sliding_window()
    res_facts = run_facts()
    res_branch = run_branching()

    print("=" * 70)
    print("СРАВНЕНИЕ РАСХОДА ТОКЕНОВ (реальные usage.* из API)")
    print("=" * 70)
    print(f"{'Стратегия':<16}{'input':>10}{'output':>10}{'доп. вызовы':>16}")
    print(f"{'Sliding Window':<16}{res_window['input_tokens']:>10}{res_window['output_tokens']:>10}{'—':>16}")
    print(f"{'Facts':<16}{res_facts['input_tokens']:>10}{res_facts['output_tokens']:>10}"
          f"{res_facts['facts_update_calls']:>16}")
    print(f"{'Branching':<16}{res_branch['input_tokens']:>10}{res_branch['output_tokens']:>10}{'—':>16}")
    print()
    print("Facts, извлечённые к концу диалога:")
    for k, v in res_facts["final_facts"].items():
        print(f"  {k}: {v}")
    print()

    with open("transcript_window.json", "w", encoding="utf-8") as f:
        json.dump(res_window["transcript"], f, ensure_ascii=False, indent=2)
    with open("transcript_facts.json", "w", encoding="utf-8") as f:
        json.dump({"final_facts": res_facts["final_facts"], "turns": res_facts["transcript"]}, f,
                   ensure_ascii=False, indent=2)
    with open("transcript_branching.json", "w", encoding="utf-8") as f:
        json.dump({
            "main": res_branch["main_transcript"],
            "branch_mobile_app": res_branch["branch_mobile_app"],
            "branch_scope_cut": res_branch["branch_scope_cut"],
        }, f, ensure_ascii=False, indent=2)

    print("Транскрипты сохранены: transcript_window.json, transcript_facts.json, transcript_branching.json")
    print("Сравните вручную: не потерялись ли стек/бюджет/дедлайн/решение про регистрацию "
          "в ответах на контрольные вопросы каждой стратегии.")


if __name__ == "__main__":
    main()
