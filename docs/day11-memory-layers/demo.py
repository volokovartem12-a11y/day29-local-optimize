"""
День 11. Демонстрация модели памяти: что попадает в каждый слой
и как это влияет на ответы агента.

Сценарий (сбор ТЗ на сайт бронирования антикафе) прогоняется ТРИЖДЫ:

  A) Только краткосрочная память (окно 4 реплики)
     — ожидание: всё, что вышло за окно, агент забывает.
  B) Все три слоя, авто-маршрутизация выключена (данные разложены вручную)
     — ожидание: агент помнит задачу и профиль, даже если реплики ушли из окна.
  C) Все три слоя, авто-маршрутизация включена (LLM-роутер сам раскладывает)
     — ожидание: то же, что B, но раскладка сделана моделью; её решения
       печатаются, чтобы было видно, что и куда ушло и почему.

В конце — контрольные вопросы на детали из начала диалога и сравнение
ответов трёх режимов + разбивка токенов по слоям.

Без AITUNNEL_API_KEY работает в MOCK-режиме (без реальных вызовов).

Запуск (PowerShell):

    cd day11-memory-layers
    $env:AITUNNEL_API_KEY = "sk-aitunnel-xxx"
    python .\\demo.py
"""

import os
import json

from memory_layers import MemoryManager, default_router_factory, approx_tokens

USE_REAL_API = bool(os.environ.get("AITUNNEL_API_KEY"))
MODEL = os.environ.get("AITUNNEL_MODEL", "gpt-4o-mini")
ROUTER_MODEL = os.environ.get("AITUNNEL_ROUTER_MODEL", MODEL)
WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "4"))

SYSTEM_PROMPT = ("Ты — ассистент с явной моделью памяти. Используй данные из блоков памяти ниже. "
                 "Если данных не хватает — так и скажи. Отвечай кратко, на русском.")

# Реплики сценария. Для режима B рядом указано, куда кладём данные вручную.
SCENARIO = [
    {
        "text": "Привет! Меня зовут Игорь, я продакт-менеджер, работаю по Москве.",
        "manual": [("profile", "имя", "Игорь"), ("profile", "роль", "продакт-менеджер"),
                    ("profile", "часовой пояс", "Москва")],
    },
    {
        "text": "Давай соберём ТЗ на сайт бронирования настольных игр в антикафе.",
        "manual": [("task_start", "ТЗ сайта бронирования антикафе", "бюджет,сроки,стек,аудитория")],
    },
    {
        "text": "Бюджет — 400000 рублей, дедлайн — 2 месяца.",
        "manual": [("slot", "бюджет", "400000 рублей"), ("slot", "сроки", "2 месяца")],
    },
    {
        "text": "Стек: бэкенд FastAPI, фронт React.",
        "manual": [("slot", "стек", "FastAPI + React")],
    },
    {
        "text": "Аудитория — молодёжь 18-30, в основном с телефонов.",
        "manual": [("slot", "аудитория", "молодёжь 18-30, мобильные")],
    },
    {
        "text": "Решили: регистрация не обязательна, бронь по номеру телефона с SMS.",
        "manual": [("decision", "", "Регистрация не обязательна, бронь по телефону с SMS-подтверждением")],
    },
    {
        "text": "Из ограничений: хостинг уже есть — VPS на 4 ГБ RAM.",
        "manual": [("knowledge", "хостинг", "VPS 4 ГБ RAM, уже есть")],
    },
    {
        "text": "Ещё нужна админка для управления расписанием столов.",
        "manual": [("slot", "админка", "управление расписанием столов")],
    },
]

CONTROL_QUESTIONS = [
    "Как меня зовут и кем я работаю?",
    "Напомни бюджет и сроки проекта.",
    "Что мы решили насчёт регистрации пользователей?",
    "Какие ограничения по хостингу?",
]

if USE_REAL_API:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["AITUNNEL_API_KEY"], base_url="https://api.aitunnel.ru/v1/")


def real_answer(messages, system):
    full = [{"role": "system", "content": system}] + list(messages)
    resp = client.chat.completions.create(model=MODEL, max_tokens=400, messages=full)
    return resp.choices[0].message.content, resp.usage.prompt_tokens, resp.usage.completion_tokens


def real_router_llm(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=ROUTER_MODEL, max_tokens=400, messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content


def mock_answer(messages, system):
    blob = system + " " + " ".join(m["content"] for m in messages)
    return f"[mock-ответ, контекст ~{approx_tokens(blob)} ток.] {messages[-1]['content'][:50]}", approx_tokens(blob), 20


def mock_router_llm(prompt: str) -> str:
    return '{"items": []}'


def answer(messages, system):
    return real_answer(messages, system) if USE_REAL_API else mock_answer(messages, system)


def apply_manual(mm, manual_ops):
    for op in manual_ops:
        kind = op[0]
        if kind == "task_start":
            mm.working.start_task(op[1], op[2].split(","))
        elif kind == "slot":
            mm.working.set_slot(op[1], op[2])
        elif kind == "profile":
            mm.long.set_profile(op[1], op[2])
        elif kind == "decision":
            mm.long.add_decision(op[2])
        elif kind == "knowledge":
            mm.long.set_knowledge(op[1], op[2])


def run_mode(mode: str, store_path: str):
    """mode: 'short_only' | 'manual' | 'auto'"""
    mm = MemoryManager(window_size=WINDOW_SIZE, storage_path=store_path)
    mm.long.forget("all")  # чистый старт для честного сравнения
    if mode == "auto":
        mm.router = default_router_factory(real_router_llm if USE_REAL_API else mock_router_llm)

    total_in, total_out = 0, 0

    for step in SCENARIO:
        text = step["text"]
        if mode == "manual":
            apply_manual(mm, step["manual"])
        elif mode == "auto":
            mm.route_message(text)

        mm.add_message("user", text)
        ctx = mm.build_context(SYSTEM_PROMPT)
        ans, tin, tout = answer(ctx["messages"], ctx["system"])
        total_in += tin
        total_out += tout
        mm.add_message("assistant", ans)

    control_answers = []
    for q in CONTROL_QUESTIONS:
        mm.add_message("user", q)
        ctx = mm.build_context(SYSTEM_PROMPT)
        ans, tin, tout = answer(ctx["messages"], ctx["system"])
        total_in += tin
        total_out += tout
        mm.add_message("assistant", ans)
        control_answers.append({"вопрос": q, "ответ": ans})

    breakdown = mm.context_breakdown(SYSTEM_PROMPT)
    snapshot = {
        "краткосрочная (в окне)": [m["content"][:60] for m in mm.short.get_window()],
        "рабочая": mm.working.render_for_prompt(),
        "долговременная": mm.long.render_for_prompt(),
    }
    return {
        "input_tokens": total_in,
        "output_tokens": total_out,
        "breakdown": breakdown,
        "snapshot": snapshot,
        "control": control_answers,
        "route_log": mm.route_log,
    }


def main():
    print(f"Режим: {'REAL API (AITunnel, ' + MODEL + ')' if USE_REAL_API else 'MOCK (без AITUNNEL_API_KEY)'}")
    print(f"Окно краткосрочной памяти: {WINDOW_SIZE} реплик")
    print(f"Реплик в сценарии: {len(SCENARIO)}, контрольных вопросов: {len(CONTROL_QUESTIONS)}\n")

    results = {}
    for mode, label, store in [
        ("short_only", "A. Только краткосрочная", "_demo_a.json"),
        ("manual", "B. Три слоя, ручная раскладка", "_demo_b.json"),
        ("auto", "C. Три слоя, авто-роутер", "_demo_c.json"),
    ]:
        print(f"--- прогон: {label} ---")
        results[mode] = run_mode(mode, store)
        results[mode]["label"] = label
        print("  готово\n")

    print("=" * 72)
    print("ЧТО ПОПАЛО В КАЖДЫЙ СЛОЙ (к концу диалога)")
    print("=" * 72)
    for mode in results:
        r = results[mode]
        print(f"\n### {r['label']}")
        snap = r["snapshot"]
        print(f"  Краткосрочная (последние {WINDOW_SIZE}): {len(snap['краткосрочная (в окне)'])} реплик")
        print(f"  Рабочая:        {snap['рабочая'] or '(пусто)'}")
        print(f"  Долговременная: {snap['долговременная'] or '(пусто)'}")

    print("\n" + "=" * 72)
    print("ВКЛАД СЛОЁВ В ПРОМПТ (оценка токенов) И РЕАЛЬНЫЙ РАСХОД")
    print("=" * 72)
    print(f"{'Режим':<32}{'кратк.':>9}{'раб.':>8}{'долгов.':>10}{'input':>10}{'output':>9}")
    for mode in results:
        r = results[mode]
        b = r["breakdown"]
        print(f"{r['label']:<32}{b['краткосрочная']:>9}{b['рабочая']:>8}"
              f"{b['долговременная']:>10}{r['input_tokens']:>10}{r['output_tokens']:>9}")

    if results["auto"]["route_log"]:
        print("\n" + "=" * 72)
        print("РЕШЕНИЯ АВТО-РОУТЕРА (что, куда и почему)")
        print("=" * 72)
        for rec in results["auto"]["route_log"]:
            key_part = f"/{rec['key']}" if rec["key"] else ""
            print(f"  → {rec['layer']}{key_part}: {rec['value'][:55]}")
            print(f"     причина: {rec['reason'][:70]}")

    print("\n" + "=" * 72)
    print("ОТВЕТЫ НА КОНТРОЛЬНЫЕ ВОПРОСЫ (влияние памяти на ответы)")
    print("=" * 72)
    for i, q in enumerate(CONTROL_QUESTIONS):
        print(f"\nВопрос: {q}")
        for mode in results:
            ans = results[mode]["control"][i]["ответ"]
            print(f"  [{results[mode]['label'][:2]}] {ans[:150]}")

    with open("demo_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nПолные результаты сохранены в demo_results.json")

    for tmp in ("_demo_a.json", "_demo_b.json", "_demo_c.json"):
        if os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    main()
