"""
Дни 28–29. RAG, который целиком работает на вашем компьютере (с оптимизированной моделью, если она собрана).

    вопрос ─► эмбеддинг запроса (bge-m3, Ollama) ─► поиск по локальному индексу SQLite
           ─► фильтр + реранкер ─► генерация ответа (qwen2.5:7b, Ollama) ─► проверка цитат кодом

Ни ключей, ни интернета (после того как модели скачаны). Адрес Ollama жёстко ограничен
localhost: если задать удалённый, программа откажется работать.

    python local_rag.py doctor                          # что готово, что нет
    python local_rag.py ask "Какие три стратегии управления контекстом в проекте дня 10?"
    python local_rag.py ask                             # интерактивный режим
    python local_rag.py ask "вопрос" --cloud            # дополнительно спросить облачную модель (нужен ключ)
    python local_rag.py ask "вопрос" --baseline         # модель до оптимизации — для сравнения
"""

import argparse
import json
import os
import sys
import time

from embeddings import ollama_url
from grounded import generate, prepare, render
from local_llm import OllamaLLM, TimedLLM, has_model, ollama_get, ollama_models
from prompts import PROMPTS, response_format
from rag import BASE, LLM, load_index
from build_index import index_path
from retrieval import load_config


def doctor(args) -> int:
    ok = True
    print(f"Адрес Ollama: {ollama_url()}  (разрешён только localhost)")
    try:
        ver = ollama_get("/api/version").get("version", "?")
        print(f"✅ Ollama отвечает, версия {ver}")
        installed = ollama_models()
    except Exception as e:  # noqa: BLE001
        print(f"❌ Ollama не отвечает: {e}\n   Запустите приложение Ollama (или `ollama serve`).")
        return 2
    embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "bge-m3")
    for role, name in (("эмбеддинги", embed_model), ("генератор", args.model)):
        if has_model(installed, name):
            print(f"✅ {role}: {name} скачана")
        else:
            ok = False
            print(f"❌ {role}: {name} не скачана → ollama pull {name}")
    path = index_path("structure", "ollama")
    if os.path.exists(path):
        print(f"✅ Локальный индекс: {os.path.relpath(path, BASE)}")
    else:
        ok = False
        print("❌ Индекса нет → python build_index.py --backend ollama")
    cfg = load_config()
    th = cfg["thresholds"].get("ollama", {})
    print(("✅" if "answer_min" in th else "⚠️ ") + f" Пороги поиска: {th}"
          + ("" if "answer_min" in th else "  (не откалиброваны → python calibrate.py --backend ollama)"))
    print("\nЛокальность: поиск и генерация идут только на этот компьютер;"
          + " облако не используется (кроме явного --cloud).")
    return 0 if ok else 1


def answer_one(index, cfg, question: str, llm, label: str, system=None, prompt_fn=None) -> None:
    t0 = time.perf_counter()
    res, chunks = prepare(index, question, cfg, "ollama", "filter", None)
    t_ret = time.perf_counter() - t0
    if chunks is not None:
        generate(llm, index, question, res, chunks, system=system, prompt_fn=prompt_fn)
    summ = llm.summary()
    print(f"\n=== {label} ===")
    print(render(res))
    speed = (f", {summ['tok_per_sec']} ток/с" if summ["tok_per_sec"] else "")
    print(f"\n⏱ поиск {t_ret:.2f} с · ответ {summ['gen_wall_sec']:.1f} с{speed} · "
          f"токены: запрос {summ['prompt_tokens']}, ответ {summ['completion_tokens']}"
          + ("" if summ["calls"] else " · модель не вызывалась (отказ на этапе поиска)"))


def ask(args) -> int:
    try:
        ollama_models()
    except RuntimeError as e:
        print(f"❌ {e}")
        return 2
    index = load_index("ollama", "structure")
    cfg = load_config()
    system = prompt_fn = None
    opt_path = os.path.join(BASE, "optimized_profile.json")
    if args.baseline or not os.path.exists(opt_path):
        local = TimedLLM(OllamaLLM(args.model), "local")
        print(f"Профиль: до оптимизации ({args.model}, настройки дня 28)")
    else:
        with open(opt_path, encoding="utf-8") as f:
            saved = json.load(f)
        p = saved["profile"]
        system, prompt_fn = PROMPTS[p.get("prompt", "v1")]
        fmt = response_format(p.get("format", "auto"))
        if has_model(ollama_models(), saved.get("model_name", "advent-rag")):
            local = TimedLLM(OllamaLLM(saved["model_name"], options={}, fmt=fmt), "local")  # параметры из Modelfile
        else:
            local = TimedLLM(OllamaLLM(p["model"], options=p.get("options", {}), fmt=fmt), "local")
        print(f"Профиль: оптимизированный «{p['id']}» → модель {local.model}")
    cloud = None
    if args.cloud:
        if not os.environ.get("AITUNNEL_API_KEY"):
            print("--cloud: нет AITUNNEL_API_KEY — спрашиваю только локальную модель.")
        else:
            cloud = TimedLLM(LLM(args.cloud_model), "cloud")

    def one(q: str) -> None:
        for timed, label in [(local, f"ЛОКАЛЬНО · {local.model}")] + ([(cloud, f"ОБЛАКО · {cloud.model}")] if cloud else []):
            timed.reset()
            try:
                answer_one(index, cfg, q, timed, label,
                           *((system, prompt_fn) if timed is local else (None, None)))
            except Exception as e:  # noqa: BLE001
                print(f"\n=== {label} ===\n❌ Ошибка: {e}")

    if args.question:
        one(args.question)
        return 0
    print("Локальный RAG. Пустая строка — выход.")
    while True:
        try:
            q = input("\nВопрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        one(q)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Полностью локальный RAG (Ollama)")
    ap.add_argument("command", choices=["doctor", "ask"])
    ap.add_argument("question", nargs="?")
    ap.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "qwen2.5:7b"), help="локальная модель-генератор")
    ap.add_argument("--cloud", action="store_true", help="для сравнения спросить ещё и облачную модель")
    ap.add_argument("--baseline", action="store_true", help="модель до оптимизации (настройки дня 28)")
    ap.add_argument("--cloud-model", default=os.environ.get("AITUNNEL_MODEL", "gpt-5.6-luna"))
    a = ap.parse_args()
    try:
        sys.exit(doctor(a) if a.command == "doctor" else ask(a))
    except RuntimeError as e:
        print(f"❌ {e}")
        sys.exit(2)
