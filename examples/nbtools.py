"""Shared helpers for the example notebooks: a matplotlib style and Markdown tables.

Three categorical slots (blue, orange, aqua) that stay distinguishable under
colour-vision deficiencies, solid hairline grids, 2px lines, and text in ink
colours rather than series colours.  Every chart in the notebooks is also
printed as a table, so no value depends on reading a colour.
"""

import matplotlib.pyplot as plt
from IPython.display import Markdown, display

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = (BLUE, ORANGE, AQUA)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "figure.dpi": 110,
            "figure.figsize": (6.4, 3.6),
            "axes.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titlelocation": "left",
            "axes.prop_cycle": plt.cycler(color=SERIES),
            "grid.color": GRID,
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "text.color": INK,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "legend.frameon": False,
            "legend.labelcolor": INK_SECONDARY,
            "font.family": "sans-serif",
            "font.size": 9.5,
        }
    )


def label_end(ax, x, y, text, color=INK_SECONDARY, dx=4, dy=0) -> None:
    """Direct label at a line's last point, in ink colour."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", va="center", color=color, fontsize=9)


def show_table(headers, rows, fmt="{:.6g}") -> None:
    """Render rows as a Markdown table; floats use ``fmt``, everything else ``str``."""

    def cell(value):
        return fmt.format(value) if isinstance(value, float) else str(value)

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join([":---"] + ["---:"] * (len(headers) - 1)) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    display(Markdown("\n".join(lines)))
