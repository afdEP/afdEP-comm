# SPDX-License-Identifier: Apache-2.0
"""Draw docs/assets/token-layout{,-dark}.svg: python docs/assets/plot_token_layout.py

One worked dispatch, as FFN GPU 0 receives it in each receive layout.
"""

from __future__ import annotations

from pathlib import Path

WIDTH, HEIGHT = 1240, 990
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
LIGHT = dict(
    bg="#fcfcfb", ink="#0b0b0b", sub="#52514e", panel="#f6f2ee", cell="#ffffff",
    line="#c3c2b7", other="#e1e0d9", orange="#eb6834", blue="#2a78d6",
    t0="#fbd9c9", t2="#d6e6fa", t3="#d5ecd4", s0="#eb6834", s2="#2a78d6",
    s3="#3f9a4a",
)  # fmt: skip
DARK = dict(
    bg="#1a1a19", ink="#f0efec", sub="#c3c2b7", panel="#25211e", cell="#2c2c2a",
    line="#4a4a46", other="#383835", orange="#d95926", blue="#3987e5",
    t0="#5a2b17", t2="#1d3350", t3="#1f3d24", s0="#d95926", s2="#3987e5",
    s3="#4fae5b",
)  # fmt: skip
ROW = 26

# 2 Attention GPUs, 2 FFN GPUs, 4 experts per FFN GPU, top_k 3, bucket 2.
# (token, source, global expert IDs, weights)
TOKENS = (
    (0, 0, (1, 5, 2), (0.5, 0.3, 0.2)),
    (1, 0, (6, 7, 4), (0.5, 0.3, 0.2)),
    (2, 1, (0, 1, 6), (0.6, 0.3, 0.1)),
    (3, 1, (3, 4, 5), (0.7, 0.2, 0.1)),
)
EXPERTS, BUCKET, SOURCES, TOP_K, ALIGNMENT = 4, 2, 2, 3, 1


def draw(c: dict[str, str]) -> str:
    out: list[str] = []

    def text(x, y, s, size=13, fill="ink", weight=400, anchor="start", mono=False):
        family = f' font-family="{MONO}"' if mono else ""
        s = s.replace("&", "&amp;").replace("<", "&lt;")
        out.append(
            f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
            f'fill="{c[fill]}" text-anchor="{anchor}"{family}>{s}</text>'
        )

    def box(x, y, w, h, fill="cell", stroke="line", rx=3, width=1, dash=False):
        dashed = ' stroke-dasharray="4 3"' if dash else ""
        out.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{c[fill]}" stroke="{c[stroke]}" stroke-width="{width}"{dashed}/>'
        )

    def cell(x, y, s, fill="cell", w=40, stroke="line", ink="ink", dash=False):
        box(x, y, w, ROW - 2, fill, stroke, width=2 if stroke != "line" else 1,
            dash=dash)  # fmt: skip
        text(x + w / 2, y + 16.5, s, 12, ink, anchor="middle", mono=True)

    def panel(x, y, w, h, title, flags=""):
        box(x, y, w, h, "panel", "panel", 14)
        text(x + 18, y + 28, title, 16, weight=700)
        text(x + w - 18, y + 28, flags, 12, "sub", anchor="end", mono=True)

    def arrow(x1, y, x2, color="sub"):
        out.append(
            f'<path d="M{x1},{y} H{x2 - 6} M{x2 - 6},{y - 4} L{x2},{y} '
            f'L{x2 - 6},{y + 4}" fill="none" stroke="{c[color]}" stroke-width="1.5"/>'
        )

    def absent(x, y, s):
        cell(x, y, s, "cell", stroke="orange", ink="orange")

    def local(expert):
        return expert < EXPERTS

    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc" '
        f'font-family="{FONT}">'
    )
    out.append('<title id="title">FFN receive layouts: expert and token</title>')
    out.append(
        '<desc id="desc">Two Attention GPUs dispatch four tokens, each with three '
        "expert choices, to two FFN GPUs. FFN GPU 0 receives three of them. The "
        "expert layout copies a token once per local expert and groups rows by "
        "expert. The token layout keeps one row per token with local expert IDs "
        "and weights; ID -1 and weight 0 mark an expert on another GPU and unused "
        "rows, written by the copy kernel.</desc>"
    )
    box(0, 0, WIDTH, HEIGHT, "bg", "bg", 0)
    text(20, 36, "What an FFN GPU receives: expert layout or token layout", 24,
         weight=700)  # fmt: skip
    text(
        20, 79,
        "A = 2 Attention GPUs  |  bucket = 2 tokens per Attention GPU at most  |  "
        "K = top_k = 3  |  E = 4 experts per FFN GPU  |  H = hidden size",
        12, "sub", mono=True,
    )  # fmt: skip
    text(
        20, 59,
        "Example: 2 Attention GPUs send 4 tokens, each picks top_k = 3 of 8 experts. "
        "FFN GPU 0 holds experts 0-3, FFN GPU 1 holds experts 4-7.",
        14, "sub",
    )  # fmt: skip

    # 1. Dispatch.
    panel(20, 86, 1200, 184, "1  Attention GPUs dispatch",
          "source_dispatch(x, topk_ids, topk_weights)")  # fmt: skip
    y0 = 148
    text(132, y0 - 8, "token", 12, "sub")
    text(212, y0 - 8, "top-3 experts : weight", 12, "sub")
    box(760, 136, 190, 46, "cell", "blue", 8, 2)
    text(855, 156, "FFN GPU 0", 14, weight=700, anchor="middle")
    text(855, 174, "experts 0-3", 12, "sub", anchor="middle")
    box(760, 204, 190, 46, "other", "line", 8)
    text(855, 224, "FFN GPU 1", 14, weight=700, anchor="middle")
    text(855, 242, "experts 4-7", 12, "sub", anchor="middle")
    for row, (token, source, ids, weights) in enumerate(TOKENS):
        y = y0 + row * ROW
        mid = y + 12
        to_zero = any(local(e) for e in ids)
        fill = f"t{token}" if to_zero else "other"
        if row % BUCKET == 0:
            text(40, y + 16.5, f"Attention {source}", 12, "sub")
        cell(132, y, f"t{token}", fill, 60)
        for k, (expert, weight) in enumerate(zip(ids, weights, strict=True)):
            cell(212 + k * 82, y, f"e{expert} : {weight:.1f}",
                 fill if local(expert) else "other", 78)  # fmt: skip
        if any(not local(e) for e in ids):
            out.append(
                f'<path d="M462,{mid} C620,{mid} 620,227 752,227" fill="none" '
                f'stroke="{c["line"]}" stroke-width="1.5"/>'
            )
        if to_zero:
            out.append(
                f'<path d="M462,{mid} C620,{mid} 620,159 752,159" fill="none" '
                f'stroke="{c[f"s{token}"]}" stroke-width="2.5"/>'
            )
    arrow(748, 159, 758, "blue")
    arrow(748, 227, 758)
    text(980, 152, "A token travels once to each FFN GPU", 13)
    text(980, 171, "that holds one of its experts,", 13)
    text(980, 190, "with all 3 IDs and weights.", 13)
    text(980, 224, "t1 picks only experts 4-7:", 13, "sub")
    text(980, 243, "FFN GPU 0 never receives it.", 13, "sub")

    rows = sorted(
        (expert, token, weight)
        for token, _, ids, weights in TOKENS
        for expert, weight in zip(ids, weights, strict=True)
        if local(expert)
    )
    counts = [sum(1 for e, _, _ in rows if e == expert) for expert in range(EXPERTS)]
    padded = [-(-count // ALIGNMENT) * ALIGNMENT for count in counts]
    psum = [sum(padded[: e + 1]) for e in range(EXPERTS)]
    expert_bound = SOURCES * BUCKET * min(TOP_K, EXPERTS) + (ALIGNMENT - 1) * EXPERTS
    expert_bound = -(-expert_bound // ALIGNMENT) * ALIGNMENT
    live = [token for token in TOKENS if any(local(e) for e in token[2])]
    allocated = SOURCES * BUCKET

    # 2a. Expert layout.
    panel(20, 286, 590, 344, "2a  Expert layout (default)", 'receive_layout="expert"')
    text(40, 336, "One row per (token, expert): a token is copied for each of its "
         "local experts.", 13, "sub")  # fmt: skip
    x0, y0 = 84, 376
    text(x0, y0 - 8, "recv_x", 12, "sub", mono=True)
    text(x0 + 100, y0 - 8, "weight", 12, "sub", mono=True)
    text(x0 + 214, y0 - 8, "worker_reply rows", 12, "sub", mono=True)
    text(x0 + 386, y0 - 8, "recv_count_psum", 12, "sub", mono=True)
    for row, (expert, token, weight) in enumerate(rows):
        y = y0 + row * ROW
        if row == 0 or rows[row - 1][0] != expert:
            text(x0 - 10, y + 16.5, f"e{expert}", 12, "blue", 700, "end", mono=True)
            if row:
                out.append(
                    f'<path d="M{x0 - 44},{y - 1} H{x0 + 150}" stroke="{c["blue"]}" '
                    'stroke-width="1"/>'
                )
        cell(x0, y, f"x(t{token})", f"t{token}", 90)
        cell(x0 + 100, y, f"{weight:.1f}", f"t{token}", 50)
        cell(x0 + 214, y, f"{weight:.1f}*E{expert}(t{token})", f"t{token}", 140)
    middle = y0 + len(rows) * ROW / 2
    arrow(x0 + 160, middle, x0 + 206)
    text(x0 + 183, middle - 10, "MLPs", 12, "sub", anchor="middle")
    for e, value in enumerate(psum):
        cell(x0 + 386 + e * 28, y0, str(value), w=26)
    text(x0 + 386, y0 + 44, "end row of each", 12, "sub")
    text(x0 + 386, y0 + 60, "expert block", 12, "sub")
    text(40, 546, "rows", 13, weight=700)
    text(100, 546, f"{len(rows)} here, at most {expert_bound}", 13)
    text(260, 546, "align(A*bucket*min(K, E) + (align-1)*E)", 12, "sub", mono=True)
    text(40, 572, "shapes", 13, weight=700)
    text(100, 572, f"recv_x [{len(rows)}, H]   recv_topk_weights [{len(rows)}]   "
         "recv_topk_idx None", 12, mono=True)  # fmt: skip
    text(40, 598, "reply", 13, weight=700)
    text(100, 598, f"{len(rows)} rows, one per (token, expert)", 13)

    # 2b. Token layout.
    panel(630, 286, 590, 344, "2b  Token layout", 'receive_layout="token"')
    text(650, 336, "One row per token, no copies: the IDs name the local experts "
         "to run.", 13, "sub")  # fmt: skip
    x0 = 650
    text(x0, y0 - 8, "recv_x", 12, "sub", mono=True)
    text(x0 + 92, y0 - 8, "recv_topk_idx", 12, "sub", mono=True)
    text(x0 + 228, y0 - 8, "recv_topk_weights", 12, "sub", mono=True)
    text(x0 + 406, y0 - 8, "worker_reply rows", 12, "sub", mono=True)
    for row in range(allocated):
        y = y0 + row * ROW
        if row >= len(live):
            cell(x0, y, "unused", w=80, dash=True)
            for k in range(TOP_K):
                absent(x0 + 92 + k * 42, y, "-1")
                absent(x0 + 228 + k * 42, y, "0")
            cell(x0 + 406, y, "unused", w=150, dash=True)
            continue
        token, _, ids, weights = live[row]
        fill = f"t{token}"
        cell(x0, y, f"x(t{token})", fill, 80)
        terms = []
        for k, (expert, weight) in enumerate(zip(ids, weights, strict=True)):
            if local(expert):
                cell(x0 + 92 + k * 42, y, str(expert), fill)
                cell(x0 + 228 + k * 42, y, f"{weight:.1f}", fill)
                terms.append(f"{weight:.1f}*E{expert}")
            else:
                absent(x0 + 92 + k * 42, y, "-1")
                absent(x0 + 228 + k * 42, y, "0")
        cell(x0 + 406, y, "+".join(terms), fill, 150)
    middle = y0 + len(live) * ROW / 2
    arrow(x0 + 358, middle, x0 + 400)
    text(x0 + 379, middle - 10, "MLPs", 12, "sub", anchor="middle")
    y = y0 + len(live) * ROW - 1
    out.append(
        f'<path d="M{x0 - 8},{y} H{x0 + 560}" stroke="{c["orange"]}" '
        'stroke-width="1.5" stroke-dasharray="5 3"/>'
    )
    text(x0 + 406, y0 + allocated * ROW + 18,
         f"recv_token_count = {len(live)}", 12, "orange", 700, mono=True)  # fmt: skip
    absent(x0, y0 + allocated * ROW + 20, "-1")
    absent(x0 + 42, y0 + allocated * ROW + 20, "0")
    text(x0 + 92, y0 + allocated * ROW + 37,
         "expert on another GPU, or unused row", 13)  # fmt: skip
    text(650, 546, "rows", 13, weight=700)
    text(710, 546, f"{len(live)} here, at most {allocated}", 13)
    text(870, 546, "A*bucket", 12, "sub", mono=True)
    text(950, 546, "recv_count_psum", 12, "sub", mono=True)
    for e, value in enumerate(psum):
        cell(1068 + e * 28, 529, str(value), w=26)
    text(650, 572, "shapes", 13, weight=700)
    text(710, 572, f"recv_x [{allocated}, H]   recv_topk_idx [{allocated}, K]   "
         f"recv_topk_weights [{allocated}, K]", 12, mono=True)  # fmt: skip
    text(650, 598, "reply", 13, weight=700)
    text(710, 598, f"{allocated} rows, one per token, summed over its local experts",
         13)  # fmt: skip

    # 3. Collect.
    box(20, 644, 1200, 40, "panel", "panel", 14)
    text(38, 669, "3  Attention GPUs sum each token's reply rows from all FFN GPUs",
         16, weight=700)  # fmt: skip
    text(1202, 669, "source_collect(): same output in both layouts", 12, "sub",
         anchor="end", mono=True)  # fmt: skip

    # Who writes -1 and 0.
    panel(20, 700, 590, 270, "Token layout: who writes -1 and 0",
          "dispatch_copy_epilogue.cuh")  # fmt: skip
    token, _, ids, weights = TOKENS[0]
    x0 = 190
    text(x0, 750, "x", 12, "sub", mono=True)
    text(x0 + 92, 750, "IDs, global", 12, "sub", mono=True)
    text(x0 + 228, 750, "weights", 12, "sub", mono=True)
    for y, label in ((758, "t0 on the wire"), (832, "t0 on FFN GPU 0")):
        text(40, y + 16.5, label, 13, weight=700)
        cell(x0, y, "x(t0)", "t0", 80)
    for k, (expert, weight) in enumerate(zip(ids, weights, strict=True)):
        cell(x0 + 92 + k * 42, 758, str(expert), "t0" if local(expert) else "other")
        cell(x0 + 228 + k * 42, 758, f"{weight:.1f}",
             "t0" if local(expert) else "other")  # fmt: skip
        if local(expert):
            cell(x0 + 92 + k * 42, 832, str(expert), "t0")
            cell(x0 + 228 + k * 42, 832, f"{weight:.1f}", "t0")
        else:
            absent(x0 + 92 + k * 42, 832, "-1")
            absent(x0 + 228 + k * 42, 832, "0")
    out.append(
        f'<path d="M{x0 + 176},786 V824 M{x0 + 172},818 L{x0 + 176},824 '
        f'L{x0 + 180},818" fill="none" stroke="{c["orange"]}" stroke-width="2"/>'
    )
    text(x0 + 190, 810, "copy kernel, kAfdTokenRoutes", 12, "orange", 700, mono=True)
    text(40, 890, "ID       local ? ID - first local ID : -1", 12, mono=True)
    text(40, 912, "weight   ID = -1 ? 0 : weight", 12, mono=True)
    text(40, 934, "rows from recv_token_count to A*bucket: every ID -1, weight 0",
         12, mono=True)  # fmt: skip
    text(40, 956, "Copy kernel: moves arrived tokens from the network buffer "
         "into recv_x.", 13, "sub")  # fmt: skip

    # Count-first and support.
    panel(630, 700, 590, 270, "When the copy kernel runs",
          "AFDEP_COUNT_FIRST_MIN_BUCKET")  # fmt: skip
    for index, (label, first) in enumerate(
        (("payload-first", False), ("count-first", True))
    ):
        y = 742 + index * 44
        text(650, y + 16.5, label, 13, weight=700)
        cell(760, y, "counts", "other", 60)
        cell(824, y, "payload arrives", "t2", 230)
        if first:
            cell(824, y + ROW - 2, "copy kernel, row by row", "cell", 240, "orange")
        else:
            cell(1058, y, "copy kernel", "cell", 140, "orange")
    text(650, 846, "Payload-first copies after every row arrived; count-first "
         "copies each row as it arrives.", 13, "sub")  # fmt: skip
    heads = ("", "1 NVLink domain", "count-first", "several domains", "FP8 replies")
    table = (
        ("expert", "yes", "yes", "yes", "yes"),
        ("token", "yes", "yes", "rejected", "rejected"),
    )
    widths = (70, 125, 120, 125, 110)
    for r, line in enumerate((heads, *table)):
        x = 650
        for value, width in zip(line, widths, strict=True):
            y = 862 + r * 28
            if r == 0:
                text(x + width / 2, y + 16.5, value, 12, "sub", anchor="middle")
            elif x == 650:
                text(x, y + 16.5, value, 12, weight=700, mono=True)
            elif value == "yes":
                cell(x + 2, y, value, "t3", width - 4)
            else:
                cell(x + 2, y, value, "cell", width - 4, "orange", "orange")
            x += width
    text(650, 958, "Several domains: no run exists yet for the token layout.", 13,
         "sub")  # fmt: skip
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    assets = Path(__file__).parent
    (assets / "token-layout.svg").write_text(draw(LIGHT))
    (assets / "token-layout-dark.svg").write_text(draw(DARK))


if __name__ == "__main__":
    main()
