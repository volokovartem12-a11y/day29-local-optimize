"""
Генерация эмбеддингов. Два бэкенда с общим интерфейсом:

  local     — без сети и ключей. Хешированный TF-IDF: признаки — основы слов,
              биграммы основ и части идентификаторов кода (save_to_file →
              save, to, file; camelCase тоже разбивается). Хеш-функция md5,
              поэтому векторы одинаковы при любом запуске. Это ЛЕКСИЧЕСКИЕ
              векторы: ловят общие термины, но не синонимы.
  aitunnel  — СЕМАНТИЧЕСКИЕ эмбеддинги модели (по умолчанию
              text-embedding-3-small) через OpenAI-совместимый API AITunnel.

Все векторы L2-нормированы, поэтому косинусная близость = скалярное произведение.
Состояние бэкенда (IDF, модель) сохраняется в индекс, чтобы запросы
векторизовались так же, как документы.
"""

import hashlib
import math
import os
import re
from collections import Counter

import numpy as np

_WORD = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")
_CAMEL = re.compile(r"(?<=[a-zа-я])(?=[A-ZА-Я])")


def _tokens(text: str) -> list:
    words = []
    for raw in _WORD.findall(text.replace("_", " ")):
        for part in _CAMEL.split(raw):
            w = part.lower()
            if len(w) > 1:
                words.append(w[:6] if len(w) > 6 else w)  # грубая основа: русские окончания отрезаются
    return words


def _features(text: str) -> Counter:
    t = _tokens(text)
    feats = Counter(t)
    feats.update(f"{a}|{b}" for a, b in zip(t, t[1:]))
    return feats


def _bucket(feature: str, dim: int) -> tuple:
    h = int.from_bytes(hashlib.md5(feature.encode("utf-8")).digest()[:8], "little")
    return h % dim, (1.0 if (h >> 63) & 1 else -1.0)


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class LocalHashEmbedder:
    name = "local"

    def __init__(self, dim: int = 1024, idf: list = None):
        self.dim = dim
        self.idf = np.array(idf, dtype=np.float32) if idf is not None else None

    @property
    def label(self) -> str:
        return f"local-hash-tfidf-{self.dim}"

    def _tf(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for feat, cnt in _features(text).items():
            i, sign = _bucket(feat, self.dim)
            v[i] += sign * (1.0 + math.log(cnt))
        return v

    def fit(self, texts: list) -> "LocalHashEmbedder":
        df = np.zeros(self.dim, dtype=np.float32)
        for t in texts:
            df[np.nonzero(self._tf(t))[0]] += 1
        self.idf = np.log((1 + len(texts)) / (1 + df)).astype(np.float32) + 1.0
        return self

    def embed(self, texts: list) -> np.ndarray:
        if self.idf is None:
            raise RuntimeError("Сначала fit() на корпусе")
        return _normalize(np.stack([self._tf(t) * self.idf for t in texts]))

    def state(self) -> dict:
        return {"backend": "local", "dim": self.dim, "idf": self.idf.tolist(), "label": self.label}


class AITunnelEmbedder:
    name = "aitunnel"

    def __init__(self, model: str = None, batch: int = 64):
        self.model = model or os.environ.get("AITUNNEL_EMBED_MODEL", "text-embedding-3-small")
        self.batch = batch
        self._client = None

    @property
    def label(self) -> str:
        return f"aitunnel:{self.model}"

    def _get_client(self):
        if self._client is None:
            key = os.environ.get("AITUNNEL_API_KEY")
            if not key:
                raise RuntimeError("Для бэкенда aitunnel нужен AITUNNEL_API_KEY")
            from openai import OpenAI
            self._client = OpenAI(api_key=key, base_url="https://api.aitunnel.ru/v1/")
        return self._client

    def fit(self, texts: list) -> "AITunnelEmbedder":
        return self  # обучать нечего

    def embed(self, texts: list) -> np.ndarray:
        client, out = self._get_client(), []
        for i in range(0, len(texts), self.batch):
            part = [t[:8000] for t in texts[i:i + self.batch]]
            resp = client.embeddings.create(model=self.model, input=part)
            out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        return _normalize(np.array(out, dtype=np.float32))

    def state(self) -> dict:
        return {"backend": "aitunnel", "model": self.model, "label": self.label}


def make_embedder(backend: str, model: str = None):
    if backend == "local":
        return LocalHashEmbedder()
    if backend == "aitunnel":
        return AITunnelEmbedder(model)
    raise ValueError(f"Неизвестный бэкенд эмбеддингов: {backend}")


def embedder_from_state(state: dict):
    if state["backend"] == "local":
        return LocalHashEmbedder(state["dim"], state["idf"])
    return AITunnelEmbedder(state["model"])
