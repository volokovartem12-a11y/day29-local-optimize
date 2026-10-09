"""
День 20. Агент поверх оркестратора и два планировщика.

Цикл одного хода:  планировщик решает, какой инструмент вызвать →
оркестратор маршрутизирует вызов на нужный сервер → результат уходит
обратно планировщику → ... → финальный ответ пользователю.

Планировщики:
  LlmPlanner      — модель через AITunnel САМА выбирает инструменты из
                    общего каталога всех серверов (проверка выбора);
  ScriptedPlanner — заранее описанный сценарий, аргументы берутся из
                    результатов предыдущих шагов. Работает без ключа —
                    проверяет оркестрацию и маршрутизацию отдельно от модели.
"""

import json
import os
from datetime import date, timedelta
from types import SimpleNamespace as NS

MAX_STEPS_PER_TURN = 12


def system_prompt() -> str:
    return (
        "Ты — ассистент менеджера автосервиса. Тебе доступны инструменты нескольких MCP-серверов; "
        "имя инструмента начинается с имени сервера: crm__ (клиенты и заказы), kb__ (база знаний), "
        "tasks__ (задачи), files__ (сохранённые отчёты). "
        "Выбирай инструмент по назначению: kb__search ищет в базе знаний, files__search — по сохранённым файлам. "
        "Все данные бери только из инструментов, ничего не выдумывай. id клиента бери из crm__search_clients. "
        "Для выжимки передавай в kb__summarize текст combined_text из kb__search. "
        "Выполни все шаги запроса пользователя, затем кратко отчитайся, что сделано. "
        f"Сегодня {date.today().isoformat()}."
    )


# ---------------------------------------------------------------------------
# Цикл агента
# ---------------------------------------------------------------------------

async def run_turn(orch, planner, messages: list, log=print) -> str:
    tools = orch.llm_tools()
    for _ in range(MAX_STEPS_PER_TURN):
        msg = planner(messages, tools)
        calls = getattr(msg, "tool_calls", None) or []
        if not calls:
            answer = msg.content or ""
            messages.append({"role": "assistant", "content": answer})
            return answer

        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in calls]})

        for c in calls:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            text, is_error = await orch.call(c.function.name, args)
            e = orch.trace[-1]
            shown = json.dumps({k: (f"<{len(v)} симв.>" if isinstance(v, str) and len(v) > 50 else v)
                                for k, v in args.items()}, ensure_ascii=False)
            log(f"  {e.n:>2}. [{e.server:<5}] {e.tool:<18} {shown}  →  {'ошибка' if is_error else 'ok'} ({e.ms} мс)")
            messages.append({"role": "tool", "tool_call_id": c.id, "content": text})

    answer = "Остановлено: слишком много шагов в одном ходе."
    messages.append({"role": "assistant", "content": answer})
    return answer


# ---------------------------------------------------------------------------
# Планировщик на LLM
# ---------------------------------------------------------------------------

class LlmPlanner:
    def __init__(self, model: str = None):
        from openai import OpenAI

        key = os.environ.get("AITUNNEL_API_KEY")
        if not key:
            raise SystemExit("Не задан AITUNNEL_API_KEY — для режима без ключа используйте --planner scripted")
        self.model = model or os.environ.get("AITUNNEL_MODEL", "gpt-4o-mini")
        self.client = OpenAI(api_key=key, base_url="https://api.aitunnel.ru/v1/")

    def __call__(self, messages, tools):
        resp = self.client.chat.completions.create(model=self.model, messages=messages, tools=tools)
        return resp.choices[0].message


# ---------------------------------------------------------------------------
# Сценарный планировщик (без LLM)
# ---------------------------------------------------------------------------

def _call(name: str, args: dict, n: int):
    return NS(content=None, tool_calls=[NS(id=f"call_{n}", function=NS(
        name=name, arguments=json.dumps(args, ensure_ascii=False)))])


class ScriptedPlanner:
    """Повторяет «идеальный» маршрут сценария. Аргументы каждого шага строятся
    из результатов предыдущих — так проверяется передача данных между серверами."""

    def __init__(self):
        self.state = {}
        self.counter = 0

    def __call__(self, messages, tools):
        turn = sum(1 for m in messages if m["role"] == "user")
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        results = []
        for m in messages[last_user + 1:]:
            if m["role"] == "tool":
                if m["content"].startswith("ОШИБКА"):
                    return NS(content=f"Не удалось выполнить шаг: {m['content']}", tool_calls=None)
                results.append(json.loads(m["content"]))
        step = self._turn1 if turn == 1 else self._turn2
        self.counter += 1
        return step(len(results), results)

    def _turn1(self, i, r):
        s = self.state
        if i == 0:
            return _call("crm__search_clients", {"query": "Анна Смирнова"}, self.counter)
        if i == 1:
            s["client"] = r[0]["clients"][0]
            return _call("crm__get_client_orders", {"client_id": s["client"]["id"]}, self.counter)
        if i == 2:
            s["orders"] = r[1]
            s["last_order"] = max(r[1]["orders"], key=lambda o: o["date"])
            return _call("kb__search", {"query": s["last_order"]["item"]}, self.counter)
        if i == 3:
            s["kb"] = r[2]
            return _call("kb__summarize", {"text": r[2]["combined_text"], "max_sentences": 3}, self.counter)
        if i == 4:
            s["summary"] = r[3]
            due = (date.today() + timedelta(days=2)).isoformat()
            return _call("tasks__create_task", {
                "title": f"Позвонить клиенту: {s['last_order']['item']} — напомнить о записи",
                "client_id": s["client"]["id"], "due_date": due}, self.counter)
        if i == 5:
            s["task"] = r[4]["task"]
            c, o = s["client"], s["orders"]
            orders = "\n".join(f"- #{x['id']} {x['item']} — {x['amount']} ₽ ({x['status']}, {x['date']})"
                               for x in o["orders"])
            report = (f"# Отчёт по клиенту: {c['name']}\n\n"
                      f"Телефон: {c['phone']}, город: {c['city']}\n\n"
                      f"## Заказы\n\n{orders}\n\nИтого без отменённых: {o['total_amount_without_cancelled']} ₽\n\n"
                      f"## Рекомендации по услуге «{s['last_order']['item']}»\n\n{s['summary']['summary']}\n\n"
                      f"## Задача\n\n#{s['task']['id']} {s['task']['title']} — до {s['task']['due_date']}\n")
            return _call("files__save_to_file", {"filename": f"client_{c['id']}_report.md", "content": report},
                         self.counter)
        s["file"] = r[5]["filename"]
        return NS(content=(f"Готово: клиент {s['client']['name']} (id {s['client']['id']}), "
                           f"заказов {s['orders']['orders_count']}, выжимка по «{s['last_order']['item']}», "
                           f"задача #{s['task']['id']} до {s['task']['due_date']}, отчёт сохранён в {s['file']}."),
                  tool_calls=None)

    def _turn2(self, i, r):
        surname = self.state["client"]["name"].split()[-1]
        if i == 0:
            return _call("files__search", {"query": surname}, self.counter)
        if i == 1:
            return _call("tasks__list_tasks", {"status": "open"}, self.counter)
        files = ", ".join(f["filename"] for f in r[0]["files"]) or "нет"
        tasks = "; ".join(f"#{t['id']} {t['title']} (до {t['due_date']})" for t in r[1]["tasks"]) or "нет"
        return NS(content=f"Найденные отчёты: {files}. Открытые задачи: {tasks}.", tool_calls=None)
