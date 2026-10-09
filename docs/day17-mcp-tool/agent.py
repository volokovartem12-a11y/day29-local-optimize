"""
День 17. Агент, который вызывает MCP-инструменты и использует результат.

Как это работает:
  1) агент запускает свой MCP-сервер и получает список инструментов;
  2) переводит их в формат function calling и передаёт модели (AITunnel);
  3) модель решает, какой инструмент вызвать и с какими аргументами;
  4) агент выполняет вызов ЧЕРЕЗ MCP (session.call_tool) и возвращает
     результат модели;
  5) модель формирует ответ пользователю на основе полученных данных.
Шаги 3–4 повторяются, пока модели нужны данные (например: сначала
найти клиента, потом по его id получить заказы).

Каждый вызов инструмента печатается строкой [MCP →] / [MCP ←],
чтобы было видно, что агент действительно ходит в MCP.

Запуск (PowerShell):
    $env:AITUNNEL_API_KEY = "sk-aitunnel-xxx"
    python agent.py                                   # диалог
    python agent.py --ask "Какие заказы у Анны Смирновой?"   # один вопрос
"""

import argparse
import asyncio
import json
import os
import sys

from mcp_client import connect, result_to_text, tools_to_openai

MODEL = os.environ.get("AITUNNEL_MODEL", "gpt-4o-mini")
BASE_URL = "https://api.aitunnel.ru/v1/"
MAX_TOOL_STEPS = 6

SYSTEM_PROMPT = (
    "Ты — ассистент менеджера. У тебя есть инструменты CRM. "
    "Для любых вопросов о клиентах и заказах сначала получай данные через инструменты, "
    "ничего не выдумывай. Если нужен id клиента — найди его через search_clients. "
    "Отвечай кратко, на русском, с конкретными цифрами из CRM."
)


def short(text: str, limit: int = 160) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


async def agent_turn(session, llm, messages: list, tools: list, log=print) -> str:
    """Один ход агента: крутим цикл «модель → инструменты → модель» до финального ответа.

    llm(messages, tools) должна вернуть сообщение модели с полями content и tool_calls.
    """
    for _ in range(MAX_TOOL_STEPS):
        msg = llm(messages, tools)
        tool_calls = getattr(msg, "tool_calls", None) or []

        if not tool_calls:
            answer = msg.content or ""
            messages.append({"role": "assistant", "content": answer})
            return answer

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ],
        })

        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            log(f"  [MCP →] {name}({json.dumps(args, ensure_ascii=False)})")
            result = await session.call_tool(name, args)
            text = result_to_text(result)
            log(f"  [MCP ←] {short(text)}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": text})

    answer = "Не удалось получить ответ: слишком много вызовов инструментов подряд."
    messages.append({"role": "assistant", "content": answer})
    return answer


def make_llm():
    from openai import OpenAI

    key = os.environ.get("AITUNNEL_API_KEY")
    if not key:
        print("Ошибка: не задан AITUNNEL_API_KEY.")
        print('Задайте: $env:AITUNNEL_API_KEY = "sk-aitunnel-xxx"')
        print("Проверить сам MCP без ключа можно так: python demo_call.py")
        sys.exit(1)
    client = OpenAI(api_key=key, base_url=BASE_URL)

    def llm(messages, tools):
        # max_tokens не передаём: часть новых моделей его не принимает
        resp = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
        return resp.choices[0].message

    return llm


async def main(question: str = None) -> None:
    llm = make_llm()
    print(f"Модель: {MODEL}. Запускаю MCP-сервер ...")
    async with connect() as session:
        mcp_tools = (await session.list_tools()).tools
        tools = tools_to_openai(mcp_tools)
        print(f"MCP подключён ✅  Инструменты агента: {', '.join(t.name for t in mcp_tools)}\n")

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if question:
            print(f"Вы: {question}")
            messages.append({"role": "user", "content": question})
            print(f"Агент: {await agent_turn(session, llm, messages, tools)}")
            return

        print("Спросите что-нибудь про клиентов и заказы. /exit — выход.")
        print("Примеры: «Какие заказы у Ивана Петрова?», «Сколько потратила Анна Смирнова?»,")
        print("         «Создай заказ на шиномонтаж за 3000 для ООО Ромашка»\n")
        while True:
            try:
                user = input("Вы: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user:
                continue
            if user == "/exit":
                break
            messages.append({"role": "user", "content": user})
            print(f"Агент: {await agent_turn(session, llm, messages, tools)}\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Агент с MCP-инструментами CRM")
    parser.add_argument("--ask", help="задать один вопрос и выйти")
    args = parser.parse_args()
    asyncio.run(main(args.ask))
