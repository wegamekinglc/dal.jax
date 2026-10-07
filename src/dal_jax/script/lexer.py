"""Script tokenizer, a port of DAL's ``script/lexer.cpp``.

Words are ASCII letters, digits, ``_`` and ``.``; operators are single
characters except ``!=``, ``<=`` and ``>=``.  An identifier immediately
followed by ``[...]`` (plus an optional ``@date`` / ``>increment`` suffix) is
one index-literal token, e.g. ``EQ[Aapl]@2026-12-31``.  Every token records its
source position; ``origins`` map offsets of a merged event text back to the
event-table row and date they came from.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from dal_jax.dates.date import Date
from dal_jax.errors import script_error

_SPACE = " \t\n\r\f\v"
_OPERATORS = "/-,;:()+*^<>="


def _is_word(c: str) -> bool:
    return ("a" <= c <= "z") or ("A" <= c <= "Z") or ("0" <= c <= "9") or c in "_."


@dataclass(frozen=True, slots=True)
class SourceLocation:
    offset: int = 0
    line: int = 1
    column: int = 1
    row: int = 0
    event_date: Date | None = None

    def describe(self) -> str:
        text = f"line={self.line}, column={self.column}, offset={self.offset}"
        if self.row:
            text += f", row={self.row}"
        if self.event_date is not None:
            text += f", event={self.event_date}"
        return text


@dataclass(frozen=True, slots=True)
class SourceOrigin:
    """Where a statement of a merged event text starts: offset, table row, event date."""

    offset: int
    row: int
    event_date: Date


@dataclass(frozen=True, slots=True)
class Token:
    source: SourceLocation
    text: str
    is_index: bool = False  # an index literal such as FX[EUR/USD]


class _Cursor:
    """Running line/column bookkeeping (DAL's ``AdvanceSource``)."""

    __slots__ = ("offset", "line", "column", "row", "event_date")

    def __init__(self) -> None:
        self.offset, self.line, self.column, self.row, self.event_date = 0, 1, 1, 0, None

    def advance(self, text: str, offset: int) -> None:
        while self.offset < offset:
            if text[self.offset] == "\n":
                self.line, self.column = self.line + 1, 1
            else:
                self.column += 1
            self.offset += 1

    def location(self) -> SourceLocation:
        return SourceLocation(self.offset, self.line, self.column, self.row, self.event_date)


def _word_end(text: str, start: int) -> int:
    while start < len(text) and _is_word(text[start]):
        start += 1
    return start


def _index_context(text: str, source: SourceLocation) -> str:
    return f"; {source.describe()}; input={text[source.offset:]}"


def _index_body_end(text: str, start: int, source: SourceLocation) -> int:
    close = text.find("]", start)
    if close == -1:
        raise script_error("InvalidIndex: missing closing ']'" + _index_context(text, source))
    if any(c in text[start:close] for c in "[\"'"):
        raise script_error("InvalidIndex: malformed index literal" + _index_context(text, source))
    return close + 1


def _argument_end(text: str, start: int) -> int:
    """Position of the next ``,`` or ``)`` (end of a function argument), or the end of the text."""
    ends = [i for i in (text.find(",", start), text.find(")", start)) if i != -1]
    return min(ends) if ends else len(text)


def _index_suffix_end(text: str, start: int, source: SourceLocation) -> int:
    if start == len(text) or text[start] not in "@>":
        return start
    end = _argument_end(text, start)
    if any(c in text[start:end] for c in "[]\"'"):
        raise script_error("InvalidIndex: malformed index suffix" + _index_context(text, source))
    return start + len(text[start:end].rstrip(_SPACE))


def _index_literal_end(text: str, start: int, source: SourceLocation) -> int:
    opening = _word_end(text, start)
    if opening == start or opening == len(text) or text[opening] != "[":
        return start
    return _index_suffix_end(text, _index_body_end(text, opening + 1, source), source)


def _script_token_end(text: str, pos: int, source: SourceLocation) -> int:
    if _is_word(text[pos]):
        return _word_end(text, pos)
    end = pos + 1
    if text[pos] in "!<>" and end < len(text) and text[end] == "=":
        return end + 1
    if text[pos] not in _OPERATORS:
        raise script_error(f"InvalidScript: unexpected character '{text[pos]}'; {source.describe()}")
    return end


def index_literal_ranges(text: str) -> list[tuple[int, int]]:
    """``[begin, end)`` spans of index literals, which macro expansion must leave untouched."""
    if "[" not in text:
        return []
    ranges = []
    cursor = _Cursor()
    pos = 0
    while pos < len(text):
        cursor.advance(text, pos)
        end = _index_literal_end(text, pos, cursor.location())
        if end != pos:
            ranges.append((pos, end))
            pos = end
        elif _is_word(text[pos]):
            pos = _word_end(text, pos)
        else:
            pos += 1
    return ranges


def lex(text: str, origins: Sequence[SourceOrigin] = ()) -> list[Token]:
    tokens = []
    cursor = _Cursor()
    origin = 0
    pos = 0
    while pos < len(text):
        if text[pos] in _SPACE:
            pos += 1
            continue
        cursor.advance(text, pos)
        while origin < len(origins) and origins[origin].offset <= pos:
            cursor.row, cursor.event_date = origins[origin].row, origins[origin].event_date
            origin += 1
        source = cursor.location()
        literal_end = _index_literal_end(text, pos, source)
        if literal_end != pos:
            tokens.append(Token(source, text[pos:literal_end], True))
            pos = literal_end
            continue
        end = _script_token_end(text, pos, source)
        tokens.append(Token(source, text[pos:end]))
        pos = end
    return tokens


def tokenize(text: str) -> list[str]:
    return [token.text for token in lex(text)]
