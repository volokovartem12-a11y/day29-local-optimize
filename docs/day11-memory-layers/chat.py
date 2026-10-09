"""
День 11. Агент с явной моделью памяти (memory layers) — через AITunnel.

Три слоя памяти:
  краткосрочная  — последние N реплик диалога
  рабочая        — слоты текущей задачи
  долговременная — профиль / решения / знания (файл memory_store.json,
                   переживает перезапуск)

Запуск (PowerShell):

    cd day11-memory-layers
    $env:AITUNNEL_API_KEY = "sk-aitunnel-xxx"
    python .\\chat.py

Или одной командой:  .\\run.ps1

КОМАНДЫ
-------
Ручная маршрутизация (вы сами решаете, что и куда):
    /task start <название> [слот1,слот2,...]  — начать задачу, задать нужные слоты
    /task set <ключ>=<значение>                — записать слот в РАБОЧУЮ память
    /task note <текст>                         — заметка в рабочую память
    /task show                                 — показать рабочую память
    /task done                                 — закрыть задачу (итог можно сохранить в долговременную)
    /remember profile <ключ>=<значение>        — в ДОЛГОВРЕМЕННУЮ (профиль)
    /remember decision <текст>                 — в ДОЛГОВРЕМЕННУЮ (решения)
    /remember knowledge <ключ>=<значение>      — в ДОЛГОВРЕМЕННУЮ (знания)

Авто-маршрутизация:
    /auto on | off      — включить/выключить LLM-роутер, который сам раскладывает
                          данные по слоям (решения всегда видны в /routing)
    /routing            — журнал: что, куда и почему было сохранено

Просмотр и очистка:
    /memory             — что сейчас лежит в каждом слое
    /context            — какой промпт реально уходит в модель (с разбивкой по слоям)
    /stats              — размеры слоёв и расход токенов
    /forget [profile|decisions|knowledge|all]  — очистить долговременную память
    /reset              — новая сессия: очистить краткосрочную и рабочую
    /exit
"""

import os
import sys

from openai import OpenAI

from memory_layers import MemoryManager, default_router_factory, approx_tokens

API_KEY = os.environ.get("AITUNNEL_API_KEY")
if not API_KEY:
    print("Ошибка: переменная окружения AITUNNEL_API_KEY не задана.")
    print('Установите её перед запуском: $env:AITUNNEL_API_KEY = "sk-aitunnel-xxx"')
    sys.exit(1)

BASE_URL = "https://api.aitunnel.ru/v1/"
MODEL = os.environ.get("AITUNNEL_MODEL", "gpt-4o-mini")
ROUTER_MODEL = os.environ.get("AITUNNEL_ROUTER_MODEL", MODEL)
WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "6"))
STORAGE_PATH = os.environ.get("MEMORY_STORE", "memory_store.json")

SYSTEM_PROMPT = ("Ты — ассистент с явной моделью памяти. Используй данные из блоков памяти ниже. "
                 "Если данных не хватает — спроси. Отвечай по делу, на русском.")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def llm_router_call(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=ROUTER_MODEL, max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def ask_model(messages, system_prompt):
    full = [{"role": "system", "content": system_prompt}] + list(messages)
    resp = client.chat.completions.create(model=MODEL, max_tokens=700, messages=full)
    return resp.choices[0].message.content, resp.usage


def print_memory(mm):
    print("\n--- КРАТКОСРОЧНАЯ (текущий диалог) ---")
    window = mm.short.get_window()
    if window:
        for m in window:
            print(f"  {m['role']}: {m['content'][:90]}")
    else:
        print("  (пусто)")

    print("\n--- РАБОЧАЯ (текущая задача) ---")
    wb = mm.working.render_for_prompt()
    print("  " + wb.replace("\n", "\n  ") if wb else "  (пусто)")

    print("\n--- ДОЛГОВРЕМЕННАЯ (профиль / решения / знания) ---")
    lb = mm.long.render_for_prompt()
    print("  " + lb.replace("\n", "\n  ") if lb else "  (пусто)")
    print()


def main():
    mm = MemoryManager(window_size=WINDOW_SIZE, storage_path=STORAGE_PATH)
    mm.router = default_router_factory(llm_router_call)
    auto_routing = os.environ.get("AUTO_ROUTING", "on").lower() == "on"

    total_in, total_out = 0, 0

    print("Агент с явной моделью памяти (AITunnel).")
    print(f"Модель: {MODEL} | окно краткосрочной: {WINDOW_SIZE} | хранилище: {STORAGE_PATH}")
    print(f"Авто-маршрутизация: {'вкл' if auto_routing else 'выкл'}")
    if mm.long.profile or mm.long.decisions or mm.long.knowledge:
        print("Долговременная память загружена из файла (агент помнит вас с прошлого раза).")
    print("Команды: /task, /remember, /auto, /routing, /memory, /context, /stats, /forget, /reset, /exit\n")

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

        # ---------- рабочая память ----------
        if user_input.startswith("/task"):
            parts = user_input.split(maxsplit=2)
            sub = parts[1] if len(parts) > 1 else ""
            arg = parts[2] if len(parts) > 2 else ""

            if sub == "start":
                bits = arg.split(maxsplit=1)
                name = bits[0] if bits else "задача"
                required = [s.strip() for s in bits[1].split(",")] if len(bits) > 1 else []
                mm.working.start_task(name, required)
                print(f"Задача '{name}' начата. Нужные слоты: {', '.join(required) or '(не заданы)'}")
            elif sub == "set":
                if "=" not in arg:
                    print("Использование: /task set ключ=значение")
                else:
                    k, v = arg.split("=", 1)
                    mm.working.set_slot(k.strip(), v.strip())
                    print(f"[РАБОЧАЯ] {k.strip()} = {v.strip()}")
            elif sub == "note":
                mm.working.add_scratch(arg)
                print(f"[РАБОЧАЯ] заметка добавлена")
            elif sub == "show":
                wb = mm.working.render_for_prompt()
                print(wb if wb else "(рабочая память пуста)")
            elif sub == "done":
                result = mm.working.finish_task()
                if result["task"]:
                    summary = f"Задача '{result['task']}' завершена: " + \
                        ", ".join(f"{k}={v}" for k, v in result["slots"].items())
                    mm.long.add_decision(summary)
                    print(f"[ДОЛГОВРЕМЕННАЯ] сохранён итог: {summary}")
                else:
                    print("Активной задачи не было.")
            else:
                print("Использование: /task start|set|note|show|done")
            continue

        # ---------- долговременная память ----------
        if user_input.startswith("/remember"):
            parts = user_input.split(maxsplit=2)
            if len(parts) < 3:
                print("Использование: /remember profile|decision|knowledge <данные>")
                continue
            section, arg = parts[1], parts[2]
            if section == "profile":
                if "=" not in arg:
                    print("Использование: /remember profile ключ=значение")
                    continue
                k, v = arg.split("=", 1)
                mm.long.set_profile(k.strip(), v.strip())
                print(f"[ДОЛГОВРЕМЕННАЯ/профиль] {k.strip()} = {v.strip()}")
            elif section == "decision":
                mm.long.add_decision(arg)
                print(f"[ДОЛГОВРЕМЕННАЯ/решения] {arg}")
            elif section == "knowledge":
                if "=" not in arg:
                    print("Использование: /remember knowledge ключ=значение")
                    continue
                k, v = arg.split("=", 1)
                mm.long.set_knowledge(k.strip(), v.strip())
                print(f"[ДОЛГОВРЕМЕННАЯ/знания] {k.strip()} = {v.strip()}")
            else:
                print("Секция должна быть: profile, decision или knowledge")
            continue

        if user_input.startswith("/forget"):
            parts = user_input.split()
            section = parts[1] if len(parts) > 1 else "all"
            mm.long.forget(section)
            print(f"Долговременная память очищена: {section}")
            continue

        # ---------- маршрутизация ----------
        if user_input.startswith("/auto"):
            parts = user_input.split()
            if len(parts) == 2 and parts[1] in ("on", "off"):
                auto_routing = parts[1] == "on"
                print(f"Авто-маршрутизация: {'вкл' if auto_routing else 'выкл'}")
            else:
                print("Использование: /auto on|off")
            continue

        if user_input == "/routing":
            if not mm.route_log:
                print("(журнал маршрутизации пуст)")
            else:
                for r in mm.route_log:
                    key_part = f" [{r['key']}]" if r["key"] else ""
                    print(f"  → {r['layer']}{key_part}: {r['value'][:60]}")
                    print(f"     причина: {r['reason'][:80]}")
            continue

        # ---------- просмотр ----------
        if user_input == "/memory":
            print_memory(mm)
            continue

        if user_input == "/context":
            ctx = mm.build_context(SYSTEM_PROMPT)
            print("\n=== SYSTEM-ПРОМПТ, который уходит в модель ===")
            print(ctx["system"])
            print("\n=== СООБЩЕНИЯ (краткосрочная память) ===")
            for m in ctx["messages"]:
                print(f"  {m['role']}: {m['content'][:90]}")
            print("\n=== ВКЛАД СЛОЁВ (оценка токенов) ===")
            for layer, tok in mm.context_breakdown(SYSTEM_PROMPT).items():
                print(f"  {layer}: {tok}")
            print()
            continue

        if user_input == "/stats":
            for layer, data in mm.stats().items():
                print(f"{layer}: {data}")
            print(f"Реальных токенов: input={total_in}, output={total_out}")
            continue

        if user_input == "/reset":
            mm.reset_session()
            print("Краткосрочная и рабочая память очищены. Долговременная сохранена.")
            continue

        # ---------- обычное сообщение ----------
        if auto_routing:
            applied = mm.route_message(user_input)
            for r in applied:
                key_part = f"/{r['key']}" if r["key"] else ""
                print(f"  [роутер] → {r['layer']}{key_part}: {r['value'][:50]}")

        mm.add_message("user", user_input)
        ctx = mm.build_context(SYSTEM_PROMPT)
        answer, usage = ask_model(ctx["messages"], ctx["system"])
        total_in += usage.prompt_tokens
        total_out += usage.completion_tokens
        mm.add_message("assistant", answer)
        print(f"Агент: {answer}\n")


if __name__ == "__main__":
    main()
