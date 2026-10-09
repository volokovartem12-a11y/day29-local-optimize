"""
Две стратегии chunking.

fixed      — окно фиксированного размера с перекрытием. Границы сдвигаются к
             ближайшему пробелу/переводу строки, чтобы не резать слова.
structure  — по структуре документа:
               Markdown — по заголовкам (# / ## / ###), путь раздела «A > B»;
                          «#» внутри блоков кода ``` заголовком не считается;
               Python   — по функциям, классам и константам верхнего уровня;
                          декораторы и комментарии над ними входят в раздел;
                          шапка модуля и блок __main__ — отдельные разделы;
               текст/PDF — по абзацам (PDF — по страницам).
             Слишком длинные разделы делятся по абзацам, слишком короткие
             склеиваются с соседним разделом того же файла.

У каждого чанка метаданные: chunk_id, strategy, source, file, title, section,
position, char_start, char_end. Для fixed раздел определяется по тому,
в каком разделе документа начинается чанк.
"""

import re
from dataclasses import dataclass

HEADER_SECTION = "(шапка документа)"


@dataclass
class Chunk:
    chunk_id: str
    strategy: str
    source: str
    file: str
    title: str
    section: str
    position: int
    char_start: int
    char_end: int
    text: str


# ---------------------------------------------------------------------------
# Разметка структуры документа: список (смещение, путь раздела)
# ---------------------------------------------------------------------------

_PY_DEF = re.compile(r"^(?:async )?(def|class) (\w+)")
_PY_ASSIGN = re.compile(r"^([A-Za-z_]\w*)\s*(?::[^=]*)?=(?!=)")


def _markdown_sections(text: str) -> list:
    markers, path, offset, in_fence = [(0, HEADER_SECTION)], [], 0, False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence:
            m = re.match(r"^(#{1,4})\s+(.+?)\s*#*$", line)
            if m:
                level = len(m.group(1))
                path = path[:level - 1] + [m.group(2).strip()]
                markers.append((offset, " > ".join(p for p in path if p)))
        offset += len(line) + 1
    return markers


def _python_sections(text: str) -> list:
    """Разделы Python: функции, классы, константы верхнего уровня и блок __main__.
    Декораторы и комментарии-заголовки над ними относятся к следующему разделу."""
    markers, offset, pending = [(0, "шапка модуля: docstring и импорты")], 0, None
    for line in text.split("\n"):
        name = None
        if line.startswith(("@", "#")):
            if pending is None:
                pending = offset
        elif not line.strip():
            pass  # пустые строки не рвут связь комментария/декоратора с разделом
        elif line.startswith("if __name__"):
            name = "блок __main__"
        elif _PY_DEF.match(line):
            kind, ident = _PY_DEF.match(line).groups()
            name = f"{kind} {ident}"
        elif _PY_ASSIGN.match(line):
            name = f"константа {_PY_ASSIGN.match(line).group(1)}"
        else:
            pending = None  # импорт, выражение или код внутри функции
        if name:
            markers.append((pending if pending is not None else offset, name))
            pending = None
        offset += len(line) + 1
    return markers


def _text_sections(text: str) -> list:
    markers = [(0, HEADER_SECTION)]
    for m in re.finditer(r"^## (Страница \d+)$", text, re.MULTILINE):  # PDF
        markers.append((m.start(), m.group(1)))
    return markers


def section_markers(doc) -> list:
    fn = {"markdown": _markdown_sections, "python": _python_sections}.get(doc.kind, _text_sections)
    markers = sorted(fn(doc.text))
    cleaned = []
    for off, name in markers:  # на одно смещение — последний заголовок
        if cleaned and cleaned[-1][0] == off:
            cleaned[-1] = (off, name)
        else:
            cleaned.append((off, name))
    return cleaned


def section_at(markers: list, offset: int) -> str:
    name = markers[0][1]
    for off, sec in markers:
        if off > offset:
            break
        name = sec
    return name


def _make(doc, strategy, pos, start, end, section) -> Chunk:
    raw = doc.text[start:end]
    lead = len(raw) - len(raw.lstrip())
    text = raw.strip()
    return Chunk(f"{doc.source}#{strategy}-{pos:03d}", strategy, doc.source, doc.file, doc.title,
                 section, pos, start + lead, start + lead + len(text), text)


# ---------------------------------------------------------------------------
# Стратегия 1: фиксированный размер
# ---------------------------------------------------------------------------

def fixed_chunks(doc, size: int = 1000, overlap: int = 150) -> list:
    if not 0 <= overlap < size:
        raise ValueError("overlap должен быть меньше size")
    text, markers, chunks = doc.text, section_markers(doc), []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):  # не резать слово: отступаем к последнему пробелу
            cut = max(text.rfind(" ", start + size // 2, end), text.rfind("\n", start + size // 2, end))
            if cut > start:
                end = cut
        piece = _make(doc, "fixed", len(chunks), start, end, "")
        if piece.text:
            piece.section = section_at(markers, piece.char_start)
            chunks.append(piece)
        if end >= len(text):
            break
        nxt = end - overlap
        space = text.find(" ", nxt, end)  # начало следующего окна — тоже по границе слова
        start = space + 1 if space != -1 else nxt
        if chunks and start <= chunks[-1].char_start:
            start = end  # защита от зацикливания
    return chunks


# ---------------------------------------------------------------------------
# Стратегия 2: по структуре
# ---------------------------------------------------------------------------

def _split_long(text_start: int, text: str, max_chars: int) -> list:
    """Длинный раздел → куски по абзацам (пустым строкам), каждый ≤ max_chars."""
    pieces, cur_start, cur_end = [], None, None
    for m in re.finditer(r"(?:[^\n]|\n(?!\n))+", text):  # абзацы
        a, b = text_start + m.start(), text_start + m.end()
        if cur_start is None:
            cur_start, cur_end = a, b
        elif b - cur_start <= max_chars:
            cur_end = b
        else:
            pieces.append((cur_start, cur_end))
            cur_start, cur_end = a, b
    if cur_start is not None:
        pieces.append((cur_start, cur_end))
    out = []  # один абзац длиннее лимита — режем по строкам
    for a, b in pieces:
        while b - a > max_chars:
            cut = text.rfind("\n", a - text_start, a - text_start + max_chars) + text_start
            cut = cut if cut > a else a + max_chars
            out.append((a, cut))
            a = cut
        out.append((a, b))
    return out


def _merge_small(items: list, text: str, min_chars: int, limit: int) -> list:
    """Мелкий кусок (< min_chars) склеивается с соседом, если вместе не больше limit.
    items — [(start, end, name)]; пустые (только пробелы) отбрасываются."""
    items = [it for it in items if text[it[0]:it[1]].strip()]
    size = lambda a, b: len(text[a:b].strip())  # noqa: E731
    out = []
    for start, end, name in items:
        if out:
            ps, pe, pn = out[-1]
            small = size(ps, pe) < min_chars or size(start, end) < min_chars
            if small and size(ps, end) <= limit:
                out[-1] = (ps, end, pn if name == pn else f"{pn} + {name}")
                continue
        out.append((start, end, name))
    return out


def structure_chunks(doc, max_chars: int = 1500, min_chars: int = 200) -> list:
    text, markers = doc.text, section_markers(doc)
    limit = max_chars + min_chars  # склейка мелкого куска может чуть превысить max_chars
    bounds = [off for off, _ in markers] + [len(text)]
    # мелкий раздел склеивается с соседом без ограничения размера: слишком большой
    # результат всё равно будет поделен по абзацам ниже
    sections = _merge_small([(bounds[i], bounds[i + 1], markers[i][1]) for i in range(len(markers))],
                            text, min_chars, float("inf"))

    chunks = []
    for start, end, name in sections:
        spans = [(start, end)] if end - start <= max_chars else _split_long(start, text[start:end], max_chars)
        spans = [(a, b) for a, b, _ in _merge_small([(a, b, name) for a, b in spans], text, min_chars, limit)]
        for i, (a, b) in enumerate(spans):
            section = name if len(spans) == 1 else f"{name} (часть {i + 1}/{len(spans)})"
            piece = _make(doc, "structure", len(chunks), a, b, section)
            if piece.text:
                chunks.append(piece)
    return chunks


STRATEGIES = {"fixed": fixed_chunks, "structure": structure_chunks}
