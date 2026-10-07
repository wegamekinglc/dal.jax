"""Export DAL's China holiday tables (``dal-cpp/dal/time/calendars/china.hpp``).

Writes ``src/dal_jax/dates/calendar_data.py`` with the CN.SSE holidays and the
CN.IB working weekends as tuples of ``(year, month, day)``.

Usage::

    python scripts/export_calendars.py <path/to/china.hpp> [output.py]
"""

import re
import sys
from pathlib import Path

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "src" / "dal_jax" / "dates" / "calendar_data.py"
DATE = re.compile(r"Date_\((\d+),\s*(\d+),\s*(\d+)\)")


def table(source: str, name: str) -> list[tuple[int, int, int]]:
    body = re.search(name + r"\s*=\s*\{(.*?)\};", source, re.S)
    if body is None:
        raise ValueError(f"{name} not found")
    return [tuple(int(x) for x in match) for match in DATE.findall(body.group(1))]


def render(name: str, dates: list[tuple[int, int, int]]) -> str:
    rows = [", ".join(f"({y}, {m}, {d})" for y, m, d in dates[i : i + 6]) for i in range(0, len(dates), 6)]
    return f"{name} = (\n" + "".join(f"    {row},\n" for row in rows) + ")\n"


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        print(__doc__)
        return 2
    source = Path(argv[1]).read_text()
    output = Path(argv[2]) if len(argv) == 3 else DEFAULT_OUTPUT
    sse = table(source, "HOLIDAYS")
    ib = table(source, "WORK_WEEKENDS")
    output.write_text(
        '"""Holiday tables exported from DAL by scripts/export_calendars.py; do not edit."""\n\n'
        + render("CN_SSE_HOLIDAYS", sse)
        + "\n"
        + render("CN_IB_WORK_WEEKENDS", ib)
    )
    print(f"wrote {len(sse)} CN.SSE holidays and {len(ib)} CN.IB working weekends to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
