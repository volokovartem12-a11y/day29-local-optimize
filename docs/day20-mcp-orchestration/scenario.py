"""
День 20. Сценарий длинного флоу и автоматическая проверка по журналу вызовов.

Сценарий — два хода пользователя, задействованы все 4 сервера:
  ход 1: CRM → база знаний → выжимка → задача → сохранение отчёта
  ход 2: поиск по сохранённым файлам → список задач

Проверки работают по журналу оркестратора, поэтому одинаково подходят
и для LLM-агента, и для сценарного планировщика. Порядок проверяется
через зависимости («А раньше Б»), а не жёсткую последовательность:
модель может выбрать другой, но корректный порядок.
"""

SCENARIO = [
    "Клиентка Анна Смирнова. Посмотри её заказы, найди в базе знаний рекомендации по услуге "
    "из её последнего заказа и сделай короткую выжимку. Создай задачу позвонить ей через два дня "
    "и сохрани отчёт по клиентке в файл.",
    "Найди среди сохранённых файлов отчёт по Смирновой и покажи все открытые задачи.",
]

REQUIRED_TURN1 = [("crm", "search_clients"), ("crm", "get_client_orders"), ("kb", "search"),
                  ("kb", "summarize"), ("tasks", "create_task"), ("files", "save_to_file")]
REQUIRED_TURN2 = [("files", "search"), ("tasks", "list_tasks")]

ORDER_TURN1 = [  # (раньше, позже, зачем)
    (("crm", "search_clients"), ("crm", "get_client_orders"), "сначала найти клиента, потом его заказы"),
    (("crm", "search_clients"), ("tasks", "create_task"), "задаче нужен id клиента"),
    (("crm", "get_client_orders"), ("kb", "search"), "тему поиска даёт последний заказ"),
    (("kb", "search"), ("kb", "summarize"), "выжимка делается из найденного"),
    (("kb", "summarize"), ("files", "save_to_file"), "в отчёт идёт выжимка"),
]


def _ok(trace, turn):
    return [e for e in trace if e.turn == turn and e.ok]


def _first(entries, server, tool):
    return next((e for e in entries if (e.server, e.tool) == (server, tool)), None)


def verify(orch) -> list:
    trace, checks = orch.trace, []

    def check(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    t1, t2 = _ok(trace, 1), _ok(trace, 2)

    # 1. маршрутизация
    bad = [e for e in trace if f"{e.server}__{e.tool}" not in orch.routes]
    check("Маршрутизация", not bad,
          "все вызовы ушли на сервер-владелец инструмента" if not bad
          else f"не смаршрутизированы: {', '.join(e.tool for e in bad)}")

    # 2. несколько серверов
    used = sorted({e.server for e in trace if e.ok})
    check("Инструменты с разных серверов", set(used) >= {s.name for s in orch.specs},
          f"задействованы серверы: {', '.join(used)} из {len(orch.specs)}")

    # 3–4. выбор нужных инструментов
    for turn, entries, required in ((1, t1, REQUIRED_TURN1), (2, t2, REQUIRED_TURN2)):
        missing = [f"{s}.{t}" for s, t in required if not _first(entries, s, t)]
        check(f"Ход {turn}: выбраны нужные инструменты", not missing,
              "все нужные инструменты вызваны" if not missing else f"не вызваны: {', '.join(missing)}")

    # 5. порядок вызовов
    for before, after, why in ORDER_TURN1:
        a, b = _first(t1, *before), _first(t1, *after)
        check(f"Порядок: {before[0]}.{before[1]} → {after[0]}.{after[1]}",
              a and b and a.n < b.n, why if a and b else "одного из вызовов нет")

    # 6. одноимённые инструменты: kb.search vs files.search
    wrong1 = any((e.server, e.tool) == ("files", "search") for e in t1)
    wrong2 = any((e.server, e.tool) == ("kb", "search") for e in t2)
    check("Выбор между kb.search и files.search", _first(t1, "kb", "search") and _first(t2, "files", "search")
          and not wrong1 and not wrong2,
          "ход 1 → kb.search (база знаний), ход 2 → files.search (файлы)")

    # 7. передача данных: client_id из CRM
    found = _first(t1, "crm", "search_clients")
    client = next((c for c in (found.data or {}).get("clients", []) if "Смирнова" in c["name"]), None) if found else None
    orders_call, task_call = _first(t1, "crm", "get_client_orders"), _first(t1, "tasks", "create_task")
    ids = {"get_client_orders": orders_call.args.get("client_id") if orders_call else None,
           "create_task": task_call.args.get("client_id") if task_call else None}
    check("Данные: client_id из CRM передан дальше", client and all(v == client["id"] for v in ids.values()),
          f"id клиента {client['id'] if client else '?'} → get_client_orders: {ids['get_client_orders']}, "
          f"create_task: {ids['create_task']}")

    # 8. передача данных: база знаний → выжимка
    kb, summ = _first(t1, "kb", "search"), _first(t1, "kb", "summarize")
    fragments = [d["text"][:40] for d in ((kb.data or {}).get("results", []) if kb else [])]
    check("Данные: выжимка сделана из найденных статей",
          summ and fragments and any(f in summ.args.get("text", "") for f in fragments),
          f"статей найдено: {len(fragments)}, их текст передан в summarize")

    # 9. отчёт сохранён и содержит клиента
    saved = _first(t1, "files", "save_to_file")
    check("Данные: отчёт сохранён", saved and "Смирнова" in saved.args.get("content", ""),
          f"файл {saved.data['filename']} содержит данные клиента" if saved else "файл не сохранён")

    # 10–11. второй ход видит результаты первого
    fsearch, tlist, task = _first(t2, "files", "search"), _first(t2, "tasks", "list_tasks"), \
        _first(t1, "tasks", "create_task")
    names = [f["filename"] for f in (fsearch.data or {}).get("files", [])] if fsearch else []
    check("Ход 2: найден отчёт из хода 1", saved and saved.data["filename"] in names,
          f"files.search нашёл: {', '.join(names) or 'ничего'}")
    task_ids = [t["id"] for t in (tlist.data or {}).get("tasks", [])] if tlist else []
    check("Ход 2: задача из хода 1 в списке", task and task.data["task"]["id"] in task_ids,
          f"открытые задачи: {task_ids}")

    # 12. ошибки: допустимы, только если исправлены повторным успешным вызовом
    unrecovered = [e for e in trace if not e.ok and not any(
        x.ok and x.n > e.n and (x.server, x.tool) == (e.server, e.tool) for x in trace)]
    errors = sum(not e.ok for e in trace)
    check("Ошибки", not unrecovered,
          "ошибок нет" if not errors else f"ошибок {errors}, неисправленных {len(unrecovered)}")
    return checks

