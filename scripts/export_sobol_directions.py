"""Export DAL's Joe-Kuo Sobol direction numbers to ``directions.npy``.

The table lives in ``dal-cpp/dal/math/random/sobol.cpp`` as 21201 arrays
``dimK[] = {...}`` of 32 ``uint32`` each.  The output keeps DAL's layout
``DIRECTIONS[dim][bit]`` as a ``uint32`` array of shape ``(21201, 32)``.

Usage::

    python scripts/export_sobol_directions.py <path/to/sobol.cpp> [output.npy]
"""

import re
import sys
from pathlib import Path

import numpy as np

N_KNOWN = 21201
N_BITS = 32
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1] / "src" / "dal_jax" / "random" / "directions.npy"
)


def parse_directions(source: str) -> np.ndarray:
    arrays = re.findall(r"const uint_least32_t dim(\d+)\[\] = \{([^}]*)\};", source)
    if len(arrays) != N_KNOWN:
        raise ValueError(f"expected {N_KNOWN} direction arrays, found {len(arrays)}")
    table = np.empty((N_KNOWN, N_BITS), dtype=np.uint32)
    for row, (index, body) in enumerate(arrays):
        if int(index) != row + 1:
            raise ValueError(f"direction arrays out of order at dim{index}")
        values = [int(v) for v in body.split(",") if v.strip()]
        if len(values) != N_BITS:
            raise ValueError(f"dim{index} has {len(values)} entries, expected {N_BITS}")
        table[row] = values
    return table


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        print(__doc__)
        return 2
    output = Path(argv[2]) if len(argv) == 3 else DEFAULT_OUTPUT
    table = parse_directions(Path(argv[1]).read_text())
    np.save(output, table)
    print(f"wrote {table.shape} {table.dtype} to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
