"""
Mock CRM API — «бэкенд», вокруг которого построен MCP-сервер.

Имитирует внешний сервис с клиентами и заказами. Работает без сети и
ключей, поэтому MCP-часть проекта можно проверить отдельно от LLM.

Чтобы заменить на настоящий API (Яндекс.Трекер, amoCRM, свой бэкенд),
достаточно переписать функции этого модуля — MCP-сервер и агент
останутся без изменений.
"""

from copy import deepcopy
from datetime import date

ORDER_STATUSES = ("new", "in_progress", "done", "cancelled")

_CLIENTS = [
    {"id": 1, "name": "Иван Петров", "phone": "+7 900 111-22-33", "city": "Москва"},
    {"id": 2, "name": "Анна Смирнова", "phone": "+7 900 222-33-44", "city": "Казань"},
    {"id": 3, "name": "ООО «Ромашка»", "phone": "+7 495 555-66-77", "city": "Москва"},
    {"id": 4, "name": "Пётр Иванов", "phone": "+7 912 333-44-55", "city": "Екатеринбург"},
]

_ORDERS = [
    {"id": 101, "client_id": 1, "item": "Замена масла", "amount": 3500, "status": "done", "date": "2026-08-02"},
    {"id": 102, "client_id": 1, "item": "Диагностика подвески", "amount": 2000, "status": "in_progress", "date": "2026-09-10"},
    {"id": 103, "client_id": 2, "item": "Шиномонтаж", "amount": 4800, "status": "new", "date": "2026-09-18"},
    {"id": 104, "client_id": 3, "item": "ТО корпоративного авто", "amount": 15000, "status": "done", "date": "2026-07-21"},
    {"id": 105, "client_id": 3, "item": "Замена колодок", "amount": 6200, "status": "cancelled", "date": "2026-08-30"},
]


class CrmError(ValueError):
    """Ошибка бизнес-логики CRM (клиент не найден, неверный статус и т.п.)."""


def search_clients(query: str) -> list:
    """Поиск клиентов по части имени, телефона или города (без учёта регистра)."""
    q = query.strip().lower()
    if not q:
        raise CrmError("Пустой поисковый запрос")
    digits = "".join(ch for ch in q if ch.isdigit())
    found = []
    for c in _CLIENTS:
        phone_digits = "".join(ch for ch in c["phone"] if ch.isdigit())
        if q in c["name"].lower() or q in c["city"].lower() or (digits and digits in phone_digits):
            found.append(deepcopy(c))
    return found


def get_client(client_id: int) -> dict:
    for c in _CLIENTS:
        if c["id"] == client_id:
            return deepcopy(c)
    raise CrmError(f"Клиент с id={client_id} не найден")


def get_client_orders(client_id: int, status: str = None) -> dict:
    """Заказы клиента, опционально с фильтром по статусу, плюс сводка."""
    client = get_client(client_id)
    if status is not None and status not in ORDER_STATUSES:
        raise CrmError(f"Неизвестный статус '{status}'. Допустимые: {', '.join(ORDER_STATUSES)}")
    orders = [deepcopy(o) for o in _ORDERS
              if o["client_id"] == client_id and (status is None or o["status"] == status)]
    active_total = sum(o["amount"] for o in orders if o["status"] != "cancelled")
    return {
        "client": client,
        "orders": orders,
        "orders_count": len(orders),
        "total_amount_without_cancelled": active_total,
    }


def create_order(client_id: int, item: str, amount: int) -> dict:
    """Создать заказ со статусом new."""
    get_client(client_id)  # проверка, что клиент существует
    if not item.strip():
        raise CrmError("Название услуги не может быть пустым")
    if amount <= 0:
        raise CrmError("Сумма заказа должна быть больше нуля")
    order = {
        "id": max(o["id"] for o in _ORDERS) + 1,
        "client_id": client_id,
        "item": item.strip(),
        "amount": amount,
        "status": "new",
        "date": date.today().isoformat(),
    }
    _ORDERS.append(order)
    return deepcopy(order)
