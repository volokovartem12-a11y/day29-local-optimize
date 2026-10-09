"""
День 19. Автоматический пайплайн из трёх MCP-инструментов.

    search ──combined_text──► summarize ──summary──► save_to_file ──► output/*.md

Пайплайн сам вызывает инструменты по очереди, передавая результат
одного шага на вход следующему. На каждом стыке проверяется, что данные
дошли без потерь и искажений (sha256 с обеих сторон), а в конце файл
перечитывается с диска и сверяется с тем, что отправляли.

Если любой шаг вернул ошибку или пустой результат — цепочка
останавливается и ничего не сохраняет.

Запуск:
    python pipeline.py
    python pipeline.py --query "RAG и поиск" --sentences 3
    python pipeline.py --mode llm            # суммаризация через AITunnel (нужен ключ)
"""

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime

from mcp_client import connect, result_to_text
from tools_impl import OUTPUT_DIR, sha256


class PipelineError(Exception):
    pass


async def call_step(session, step_no: int, name: str, args: dict, trace: list, log) -> dict:
    """Вызвать MCP-инструмент, распарсить JSON-результат и записать шаг в трассировку."""
    shown = {k: (f"<{len(v)} символов>" if isinstance(v, str) and len(v) > 60 else v) for k, v in args.items()}
    log(f"\n[Шаг {step_no}] {name}({json.dumps(shown, ensure_ascii=False)})")
    result = await session.call_tool(name, args)
    text = result_to_text(result)
    if getattr(result, "isError", False):
        trace.append({"step": step_no, "tool": name, "args": shown, "error": text})
        raise PipelineError(f"Шаг {step_no} ({name}) вернул ошибку: {text}")
    data = json.loads(text)
    trace.append({"step": step_no, "tool": name, "args": shown, "result_keys": sorted(data)})
    return data


def build_report(query: str, found: dict, summary: dict) -> str:
    sources = "\n".join(f"- {d['title']} (`{d['id']}`, релевантность {d['score']})" for d in found["results"])
    points = "\n".join(f"{i}. {s}" for i, s in enumerate(summary["sentences"], 1))
    return (
        f"# Выжимка по запросу: {query}\n\n"
        f"_Сформировано автоматически пайплайном search → summarize → save_to_file, "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}_\n\n"
        f"## Главное\n\n{points}\n\n"
        f"## Источники\n\n{sources}\n\n"
        f"## Статистика\n\n"
        f"- Найдено статей: {found['found']}\n"
        f"- Исходный текст: {summary['input_chars']} символов\n"
        f"- Выжимка: {summary['output_chars']} символов (сжатие до {summary['compression']:.0%})\n"
        f"- Режим суммаризации: {summary['mode']}\n"
    )


async def run_pipeline(session, query: str, top_k: int = 3, sentences: int = 4,
                       mode: str = "extractive", filename: str = None, log=print) -> dict:
    trace, checks = [], []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": ok, "detail": detail})
        log(f"   {'✅' if ok else '❌'} {name}: {detail}")
        if not ok:
            raise PipelineError(f"Проверка не пройдена: {name}")

    # --- Шаг 1: получаем данные ---
    found = await call_step(session, 1, "search", {"query": query, "top_k": top_k}, trace, log)
    if found["found"] == 0:
        raise PipelineError(f"По запросу «{query}» ничего не найдено — дальше передавать нечего")
    text = found["combined_text"]
    log(f"   → найдено статей: {found['found']}, текст: {len(text)} символов")
    check("search → клиент", sha256(text) == found["combined_sha256"],
          "текст дошёл до клиента без искажений (sha256 совпадает)")

    # --- Шаг 2: обрабатываем то, что вернул шаг 1 ---
    summary = await call_step(session, 2, "summarize",
                              {"text": text, "max_sentences": sentences, "mode": mode}, trace, log)
    log(f"   → выжимка: {summary['output_chars']} символов, сжатие до {summary['compression']:.0%}")
    check("клиент → summarize", summary["input_sha256"] == sha256(text),
          "summarize получил ровно тот текст, что вернул search")
    if mode == "extractive":
        missing = [s for s in summary["sentences"] if s not in text]
        check("выжимка из исходных данных", not missing,
              f"все {len(summary['sentences'])} предложения взяты из найденных статей")

    # --- Шаг 3: сохраняем результат шага 2 ---
    content = build_report(query, found, summary)
    check("summary → отчёт", all(s in content for s in summary["sentences"]),
          "все предложения выжимки попали в отчёт")
    if not filename:
        slug = re.sub(r"[^\w]+", "_", query.lower()).strip("_")[:40] or "report"
        filename = f"summary_{slug}.md"
    saved = await call_step(session, 3, "save_to_file", {"filename": filename, "content": content}, trace, log)
    log(f"   → файл: {saved['path']} ({saved['bytes']} байт)")
    check("клиент → save_to_file", saved["sha256"] == sha256(content),
          "на диск записано ровно то, что отправлено")
    with open(saved["path"], "r", encoding="utf-8", newline="") as f:
        on_disk = f.read()
    check("файл на диске", sha256(on_disk) == sha256(content),
          "перечитанный файл совпадает с отчётом")

    return {"query": query, "file": saved["path"], "steps": trace, "checks": checks,
            "sizes": {"search_chars": len(text), "summary_chars": summary["output_chars"],
                      "file_bytes": saved["bytes"]}}


async def main(args) -> int:
    log = print
    print("Запускаю MCP-сервер и подключаюсь по stdio ...")
    async with connect() as session:
        tools = [t.name for t in (await session.list_tools()).tools]
        print(f"Соединение установлено ✅  Инструменты: {', '.join(tools)}")
        missing = {"search", "summarize", "save_to_file"} - set(tools)
        if missing:
            print(f"❌ На сервере нет инструментов: {', '.join(sorted(missing))}")
            return 1

        print(f"\nЗапрос: «{args.query}»   режим суммаризации: {args.mode}")
        print("Цепочка: search → summarize → save_to_file")
        try:
            report = await run_pipeline(session, args.query, args.top_k, args.sentences,
                                        args.mode, args.out, log)
        except PipelineError as e:
            print(f"\n⛔ Пайплайн остановлен: {e}\nФайл не сохранён.")
            return 1

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trace_path = os.path.join(OUTPUT_DIR, "pipeline_trace.json")
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    passed = sum(c["ok"] for c in report["checks"])
    s = report["sizes"]
    print("\n" + "=" * 64)
    print(f"ПАЙПЛАЙН ВЫПОЛНЕН АВТОМАТИЧЕСКИ ✅  шагов: 3, проверок пройдено: {passed}/{len(report['checks'])}")
    print("=" * 64)
    print(f"Поток данных: {s['search_chars']} симв. → {s['summary_chars']} симв. → файл {s['file_bytes']} байт")
    print(f"Результат:    {report['file']}")
    print(f"Трассировка:  {trace_path}")
    print("\n--- содержимое сохранённого файла ---\n")
    with open(report["file"], encoding="utf-8") as f:
        print(f.read())
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Автоматический пайплайн search → summarize → save_to_file")
    p.add_argument("--query", default="композиция MCP инструментов")
    p.add_argument("--top-k", type=int, default=3, help="сколько статей искать (1–10)")
    p.add_argument("--sentences", type=int, default=4, help="длина выжимки в предложениях")
    p.add_argument("--mode", choices=["extractive", "llm"], default="extractive")
    p.add_argument("--out", help="имя файла результата (по умолчанию — по запросу)")
    sys.exit(asyncio.run(main(p.parse_args())))
