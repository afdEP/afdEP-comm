from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
YELLOW, GRAY, BLACK = "#FCBA00", "#9E9D9D", "#000000"
# File suffix: (ink, background, muted ink).
THEMES = {"": (BLACK, "#FFFFFF", "#5F5E5E"), "-dark": ("#FFFFFF", BLACK, GRAY)}
MONO = "DejaVu Sans Mono"
LANE_ORDER = (
    (
        "Attention",
        (
            "dispatch 0a",
            "dispatch 1a",
            "collect 0a",
            "dispatch 0b",
            "collect 1a",
            "dispatch 1b",
        ),
    ),
    (
        "FFN",
        (
            "receive 0a",
            "receive 1a",
            "reply 0a",
            "receive 0b",
            "reply 1a",
            "receive 1b",
        ),
    ),
)


def box(ax, x, y, w, h, text, fc, ink, size=11, ec="none"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.08",
            fc=fc,
            ec=ec,
            lw=1.2,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        color=ink,
        fontsize=size,
        family=MONO,
        weight="bold",
    )


def calls(ink, muted):
    """The four calls of one exchange, top to bottom in time."""
    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    ax.set_xlim(-0.4, 10.4)
    ax.set_ylim(0, 6)
    ax.axis("off")
    xa, xf = 2.2, 7.8
    for x, name in ((xa, "Attention rank"), (xf, "FFN rank")):
        box(ax, x - 1.5, 5.2, 3.0, 0.6, name, "none", ink, ec=ink)
        ax.plot([x, x], [0.3, 5.2], color=muted, lw=1, ls=(0, (4, 4)))
    steps = (
        (4.5, xa, "1  source_dispatch", "tokens, expert IDs, weights", 1),
        (3.5, xf, "2  worker_receive", "", 0),
        (2.2, xf, "3  worker_reply", "expert outputs", -1),
        (1.2, xa, "4  source_collect", "", 0),
    )
    for y, x, name, payload, direction in steps:
        box(ax, x - 1.5, y - 0.28, 3.0, 0.56, name, YELLOW, BLACK, size=10.5)
        if direction:
            a, b = (xa + 1.6, xf - 0.1) if direction > 0 else (xf - 1.6, xa + 0.1)
            ax.annotate(
                "",
                xy=(b, y),
                xytext=(a, y),
                arrowprops=dict(arrowstyle="-|>", color=ink, lw=1.6),
            )
            ax.text(5, y + 0.2, payload, ha="center", color=ink, fontsize=10)
    ax.text(
        xf + 1.7,
        3.5,
        "returns\nReceiveResult",
        ha="left",
        va="center",
        color=muted,
        fontsize=9,
    )
    ax.text(
        xa - 1.7,
        1.2,
        "returns\nBF16 [T,H]",
        ha="right",
        va="center",
        color=muted,
        fontsize=9,
    )
    ax.text(
        5,
        2.85,
        "the caller runs the experts",
        ha="center",
        color=muted,
        fontsize=9,
        style="italic",
    )
    return fig


def lanes(ink, muted):
    """Two lanes and layers a, b: the call order on each role."""
    fig, ax = plt.subplots(figsize=(9.4, 2.9))
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for y, (role, names) in zip((2.5, 1.3), LANE_ORDER, strict=True):
        ax.text(
            1.5,
            y + 0.35,
            role,
            ha="right",
            va="center",
            color=ink,
            fontsize=11,
            weight="bold",
        )
        for i, name in enumerate(names):
            color = GRAY if name.split()[1][0] == "1" else YELLOW
            box(ax, 1.75 + 1.92 * i, y, 1.8, 0.7, name, color, BLACK, size=9)
    ax.annotate(
        "",
        xy=(13.2, 0.85),
        xytext=(1.75, 0.85),
        arrowprops=dict(arrowstyle="->", color=muted, lw=1.2),
    )
    ax.text(7.5, 0.45, "call order on every rank", ha="center", color=muted, fontsize=9)
    for x, color, name in ((1.75, YELLOW, "lane 0"), (3.4, GRAY, "lane 1")):
        box(ax, x, 3.55, 0.3, 0.25, "", color, ink)
        ax.text(x + 0.42, 3.675, name, va="center", color=ink, fontsize=9.5)
    return fig


for name, draw in (("api-calls", calls), ("api-lanes", lanes)):
    for suffix, (ink, bg, muted) in THEMES.items():
        fig = draw(ink, muted)
        fig.tight_layout(pad=0.3)
        fig.savefig(HERE / f"{name}{suffix}.svg", facecolor=bg, metadata={"Date": None})
        plt.close(fig)
