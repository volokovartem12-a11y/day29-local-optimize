"""
День 10. Агент с переключателем стратегий управления контекстом — через AITunnel.

Три стратегии, переключаются командой /strategy:
  window — Sliding Window: только последние N сообщений
  facts  — Sticky Facts: facts (ключ-значение) + последние N сообщений
  branch — Branching: дерево сообщений, можно ставить checkpoint,
           создавать ветки и переключаться между ними

Запуск (PowerShell):

    cd day10-context-strategies
    $env:AITUNNEL_API_KEY = "sk-aitunnel-xxx"
    python .\\chat.py

Если команда `python` у вас не находит настоящий интерпретатор (ставит
Windows-заглушку вместо запуска) — используйте `py .\\chat.py`, либо полный
путь к python.exe, который показывает `pip` после установки пакетов.

Команды внутри чата:
    /strategy window|facts|branch   — переключить активную стратегию
                                       (история переносится в новую стратегию как есть)
    /window N                       — задать размер окна (для window и facts)
    /facts                          — показать текущий блок facts
    /checkpoint <имя>                — (branch) поставить чекпоинт на текущем месте
    /branch <имя> [from <чекпоинт>] — (branch) создать новую ветку
    /switch <имя_ветки>              — (branch) переключиться на ветку
    /branches                        — (branch) показать список веток
    /history                         — показать сообщения в активном контексте
    /stats                           — расход токенов текущей стратегии (реальный, из API)
    /exit
"""

import os
import sys

from openai import OpenAI

from context_strategies import (
    SlidingWindowStrategy,
    FactsStrategy,
    BranchingStrategy,
    default_facts_updater_factory,
    approx_tokens,
)

API_KEY = os.environ.get("AITUNNEL_API_KEY")
if not API_KEY:
    print("Ошибка: переменная окружения AITUNNEL_API_KEY не задана.")
    print('Установите её перед запуском: $env:AITUNNEL_API_KEY = "sk-aitunnel-xxx"')
    sys.exit(1)

BASE_URL = "https://api.aitunnel.ru/v1/"
MODEL = os.environ.get("AITUNNEL_MODEL", "gpt-4o-mini")
FACTS_MODEL = os.environ.get("AITUNNEL_FACTS_MODEL", MODEL)

WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "6"))
FACTS_WINDOW_SIZE = int(os.environ.get("FACTS_WINDOW_SIZE", "4"))

SYSTEM_PROMPT = "Ты — ассистент, который помогает собрать ТЗ на проект. Отвечай по делу, на русском."

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def llm_extract_facts(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=FACTS_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def ask_model(messages, system_prompt):
    full_messages = [{"role": "system", "content": system_prompt}] + list(messages)
    resp = client.chat.completions.create(model=MODEL, max_tokens=700, messages=full_messages)
    return resp.choices[0].message.content, resp.usage


def make_strategy(name: str):
    if name == "window":
        return SlidingWindowStrategy(window_size=WINDOW_SIZE)
    if name == "facts":
        s = FactsStrategy(window_size=FACTS_WINDOW_SIZE)
        s.facts_updater = default_facts_updater_factory(llm_extract_facts)
        return s
    if name == "branch":
        return BranchingStrategy()
    raise ValueError(f"Неизвестная стратегия: {name}")


def reseed(new_strategy, old_history):
    """Переносит уже состоявшийся диалог в свежесозданную стратегию."""
    for m in old_history:
        new_strategy.add_message(m["role"], m["content"])


def get_flat_history(strategy):
    if isinstance(strategy, BranchingStrategy):
        return [{"role": m["role"], "content": m["content"]} for m in strategy.get_history()]
    return list(strategy.full_history)


def main():
    active_name = os.environ.get("DEFAULT_STRATEGY", "window")
    strategy = make_strategy(active_name)

    total_in, total_out = 0, 0

    print("Агент с переключателем стратегий контекста (AITunnel).")
    print(f"Модель: {MODEL} | активная стратегия: {active_name} | window={WINDOW_SIZE}")
    print("Команды: /strategy window|facts|branch, /window N, /facts, /checkpoint, /branch, "
          "/switch, /branches, /history, /stats, /exit\n")

    while True:
        try:
            user_input = input(f"[{active_name}] Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nЗавершение.")
            break

        if not user_input:
            continue

        if user_input == "/exit":
            break

        if user_input.startswith("/strategy"):
            parts = user_input.split()
            if len(parts) != 2 or parts[1] not in ("window", "facts", "branch"):
                print("Использование: /strategy window|facts|branch")
                continue
            old_history = get_flat_history(strategy)
            active_name = parts[1]
            strategy = make_strategy(active_name)
            reseed(strategy, old_history)
            print(f"Переключено на стратегию: {active_name} (история перенесена, {len(old_history)} сообщ.)")
            continue

        if user_input.startswith("/window"):
            parts = user_input.split()
            if len(parts) != 2 or not parts[1].isdigit():
                print("Использование: /window N")
                continue
            if hasattr(strategy, "window_size"):
                strategy.window_size = int(parts[1])
                print(f"window_size = {strategy.window_size}")
            else:
                print("У активной стратегии нет window_size.")
            continue

        if user_input == "/facts":
            if isinstance(strategy, FactsStrategy):
                if strategy.facts:
                    for k, v in strategy.facts.items():
                        print(f"  {k}: {v}")
                else:
                    print("  (пока пусто)")
            else:
                print("Facts доступны только в стратегии facts.")
            continue

        if user_input.startswith("/checkpoint"):
            if not isinstance(strategy, BranchingStrategy):
                print("Checkpoint доступен только в стратегии branch.")
                continue
            parts = user_input.split(maxsplit=1)
            if len(parts) != 2:
                print("Использование: /checkpoint <имя>")
                continue
            strategy.checkpoint(parts[1])
            print(f"Checkpoint '{parts[1]}' поставлен на ветке '{strategy.active_branch}'.")
            continue

        if user_input == "/branches":
            if not isinstance(strategy, BranchingStrategy):
                print("Ветки доступны только в стратегии branch.")
                continue
            for b in strategy.list_branches():
                marker = " <- активная" if b == strategy.active_branch else ""
                print(f"  {b}{marker}")
            continue

        if user_input.startswith("/branch "):
            if not isinstance(strategy, BranchingStrategy):
                print("Ветки доступны только в стратегии branch.")
                continue
            parts = user_input.split()
            if len(parts) == 2:
                strategy.create_branch(parts[1])
            elif len(parts) == 4 and parts[2] == "from":
                strategy.create_branch(parts[1], from_checkpoint=parts[3])
            else:
                print("Использование: /branch <имя> [from <чекпоинт>]")
                continue
            strategy.switch(parts[1])
            print(f"Создана и активирована ветка '{parts[1]}'.")
            continue

        if user_input.startswith("/switch"):
            if not isinstance(strategy, BranchingStrategy):
                print("Переключение веток доступно только в стратегии branch.")
                continue
            parts = user_input.split()
            if len(parts) != 2:
                print("Использование: /switch <имя_ветки>")
                continue
            try:
                strategy.switch(parts[1])
                print(f"Активная ветка: {parts[1]}")
            except KeyError as e:
                print(str(e))
            continue

        if user_input == "/history":
            hist = get_flat_history(strategy)
            for m in hist:
                print(f"  {m['role']}: {m['content'][:100]}")
            continue

        if user_input == "/stats":
            print(f"— Реальных input-токенов потрачено: {total_in}")
            print(f"— Реальных output-токенов потрачено: {total_out}")
            if isinstance(strategy, FactsStrategy):
                print(f"— Вызовов извлечения фактов: {strategy.facts_update_calls}")
            continue

        # обычное сообщение пользователя
        strategy.add_message("user", user_input)
        ctx = strategy.get_context(SYSTEM_PROMPT)
        answer, usage = ask_model(ctx["messages"], ctx["system"])
        total_in += usage.prompt_tokens
        total_out += usage.completion_tokens
        strategy.add_message("assistant", answer)
        print(f"Агент: {answer}\n")


if __name__ == "__main__":
    main()
