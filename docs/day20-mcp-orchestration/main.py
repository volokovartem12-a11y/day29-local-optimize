"""
День 20. Точка входа: оркестрация нескольких MCP-серверов.

    python main.py --list                      # реестр серверов и каталог инструментов
    python main.py --planner scripted          # длинный флоу без LLM (ключ не нужен)
    python main.py --planner llm               # длинный флоу, инструменты выбирает модель
    python main.py --planner llm --model gpt-5.6-luna

После прогона печатается журнал вызовов и результат проверок, а в output/
сохраняются orchestration_trace.json и orchestration_report.md.
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict

from agent import LlmPlanner, ScriptedPlanner, run_turn, system_prompt
from orchestrator import BASE_DIR, Orchestrator, load_registry
from scenario import SCENARIO, verify

OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def print_catalog(orch) -> None:
    print(f"\nЗарегистрировано серверов: {len(orch.specs)}, инструментов всего: {len(orch.routes)}")
    for spec in orch.specs:
        tools = orch.catalog()[spec.name]
        print(f"  ▸ {spec.name:<6} — {spec.description} ({len(tools)} инстр.)")
        for full, r in tools:
            params = ", ".join(r.schema.get("properties", {})) or "—"
            print(f"      {full:<26} ({params})")
    dupes = {}
    for r in orch.routes.values():
        dupes.setdefault(r.tool, []).append(r.server)
    for tool, servers in dupes.items():
        if len(servers) > 1:
            print(f"  ⚠ инструмент «{tool}» есть на серверах {', '.join(servers)} — "
                  f"различаются по префиксу: {', '.join(s + '__' + tool for s in servers)}")


def save_artifacts(orch, answers, checks, planner_name) -> tuple:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trace_path = os.path.join(OUTPUT_DIR, "orchestration_trace.json")
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump({"planner": planner_name, "scenario": SCENARIO, "answers": answers,
                   "calls": [{k: v for k, v in asdict(e).items() if k != "data"} for e in orch.trace],
                   "checks": checks}, f, ensure_ascii=False, indent=2)

    rows = "\n".join(f"| {e.n} | {e.turn} | {e.server} | {e.tool} | {'✅' if e.ok else '❌'} | {e.ms} |"
                     for e in orch.trace)
    chk = "\n".join(f"| {'✅' if c['ok'] else '❌'} | {c['check']} | {c['detail']} |" for c in checks)
    passed = sum(c["ok"] for c in checks)
    report = (f"# Отчёт: оркестрация MCP (планировщик: {planner_name})\n\n"
              f"Серверов: {len(orch.specs)}, вызовов: {len(orch.trace)}, "
              f"проверок пройдено: {passed}/{len(checks)}\n\n"
              f"## Вызовы\n\n| # | Ход | Сервер | Инструмент | Статус | мс |\n|---|---|---|---|---|---|\n{rows}\n\n"
              f"## Проверки\n\n| | Проверка | Детали |\n|---|---|---|\n{chk}\n")
    report_path = os.path.join(OUTPUT_DIR, "orchestration_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    return trace_path, report_path


async def main(args) -> int:
    specs = load_registry()
    planner = None if args.list else (ScriptedPlanner() if args.planner == "scripted" else LlmPlanner(args.model))

    print(f"Запускаю {len(specs)} MCP-сервера из servers.json ...")
    async with Orchestrator(specs) as orch:
        print("Все серверы подключены ✅")
        print_catalog(orch)
        if args.list:
            return 0

        name = "scripted" if args.planner == "scripted" else f"llm ({planner.model})"
        print(f"\nПланировщик: {name}")
        messages = [{"role": "system", "content": system_prompt()}]
        answers = []
        for turn, text in enumerate(SCENARIO, 1):
            orch.turn = turn
            print(f"\n{'─' * 70}\nХод {turn}. Пользователь: {text}\n")
            messages.append({"role": "user", "content": text})
            answer = await run_turn(orch, planner, messages)
            answers.append(answer)
            print(f"\nАгент: {answer}")

    checks = verify(orch)
    passed = sum(c["ok"] for c in checks)
    print(f"\n{'=' * 70}\nПРОВЕРКА ФЛОУ: вызовов {len(orch.trace)}, серверов {len({e.server for e in orch.trace})}, "
          f"ходов {len(SCENARIO)}\n{'=' * 70}")
    for c in checks:
        print(f"  {'✅' if c['ok'] else '❌'} {c['check']}: {c['detail']}")
    print(f"\nИтог: {passed}/{len(checks)} проверок пройдено {'✅' if passed == len(checks) else '❌'}")

    trace_path, report_path = save_artifacts(orch, answers, checks, name)
    print(f"Журнал: {trace_path}\nОтчёт:  {report_path}")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Оркестрация нескольких MCP-серверов")
    p.add_argument("--list", action="store_true", help="только показать реестр и каталог инструментов")
    p.add_argument("--planner", choices=["scripted", "llm"], default="scripted")
    p.add_argument("--model", help="модель AITunnel для --planner llm")
    try:
        sys.exit(asyncio.run(main(p.parse_args())))
    except (FileNotFoundError, ValueError) as e:  # ошибки реестра servers.json
        print(f"\n❌ Ошибка реестра: {e}")
        sys.exit(1)
    except BaseExceptionGroup as eg:  # ошибки запуска серверов приходят группой
        err = eg
        while isinstance(err, BaseExceptionGroup):
            err = err.exceptions[0]
        print(f"\n❌ Ошибка: {type(err).__name__}: {err}")
        sys.exit(1)
