# SPDX-License-Identifier: Apache-2.0
"""Draw docs/assets/token-layout{,-dark}.svg: python docs/assets/plot_token_layout.py

One worked dispatch, as FFN GPU 0 receives it in each receive layout.
"""

from __future__ import annotations

from pathlib import Path

WIDTH, HEIGHT = 1240, 1112
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
    out.append('<g transform="translate(0,-34)">')

    # 1. Dispatch.
    panel(20, 86, 1200, 184, "Attention GPUs dispatch",
          "source_dispatch(x, topk_ids, topk_weights)")  # fmt: skip
    y0 = 148
    text(132, y0 - 8, "token", 12, "sub")
    text(212, y0 - 8, f"top_k = {TOP_K} experts : weight", 12, "sub")
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
            text(40, y + ROW + 14, f"bucket = {BUCKET}", 11, "sub")
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
    text(748, 130, f"once per FFN GPU: x + {TOP_K} IDs + {TOP_K} weights", 12, "blue",
         anchor="end")  # fmt: skip
    text(748, 264, f"t1: all {TOP_K} experts on FFN GPU 1", 12, "sub", anchor="end")
    for y, zero in ((147, True), (215, False)):
        got = [t for t, _, ids, _ in TOKENS if any(local(e) == zero for e in ids)]
        arrow(954, y + 12, 972)
        text(980, y + 16.5, "receives", 12, "sub")
        for i, token in enumerate(got):
            cell(1040 + i * 44, y, f"t{token}", f"t{token}" if zero else "other")

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
    panel(20, 286, 590, 318, "Expert layout - what FFN GPU 0 receives",
          'receive_layout="expert" (default)')  # fmt: skip
    text(40, 336, "One row per (token, local expert): tokens are copied", 13, "sub")
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
    text(260, 546, f"= {SOURCES} Attention GPUs x {BUCKET} tokens x "
         f"{min(TOP_K, EXPERTS)} experts", 12, "sub")  # fmt: skip
    text(40, 572, "shapes", 13, weight=700)
    text(100, 572, f"recv_x [{len(rows)}, hidden]   recv_topk_weights [{len(rows)}]   "
         "recv_topk_idx None", 12, mono=True)  # fmt: skip

    # 2b. Token layout.
    panel(630, 286, 590, 318, "Token layout - what FFN GPU 0 receives",
          'receive_layout="token"')  # fmt: skip
    text(650, 336,
         f"One row per token; columns = its {TOP_K} chosen experts (top_k), not GPUs",
         13, "sub")  # fmt: skip
    x0 = 650
    text(x0, y0 - 20, "recv_x", 12, "sub", mono=True)
    text(x0 + 92, y0 - 20, "recv_topk_idx", 12, "sub", mono=True)
    text(x0 + 228, y0 - 20, "recv_topk_weights", 12, "sub", mono=True)
    text(x0 + 406, y0 - 20, "worker_reply rows", 12, "sub", mono=True)
    for k, choice in enumerate(("1st", "2nd", "3rd")[:TOP_K]):
        for x in (x0 + 92, x0 + 228):
            text(x + k * 42 + 20, y0 - 5, choice, 10, "sub", anchor="middle")
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
                cell(x0 + 92 + k * 42, y, f"e{expert}", fill)
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
    text(x0 + 556, y0 + allocated * ROW + 10,
         f"recv_token_count = {len(live)}", 11, "orange", 700, mono=True,
         anchor="end")  # fmt: skip
    absent(x0, y0 + allocated * ROW + 20, "-1")
    absent(x0 + 42, y0 + allocated * ROW + 20, "0")
    token, _, ids, _ = live[0]
    away = next(e for e in ids if not local(e))
    text(x0 + 92, y0 + allocated * ROW + 29,
         "ID -1, weight 0 = skip: expert is on another GPU, or row is unused",
         12)  # fmt: skip
    text(x0 + 92, y0 + allocated * ROW + 45,
         f"e.g. t{token} chose {', '.join(f'e{e}' for e in ids)}; e{away} is on "
         f"FFN GPU {away // EXPERTS}, which runs it", 12, "sub")  # fmt: skip
    text(650, 546, "rows", 13, weight=700)
    text(710, 546, f"{len(live)} here, at most {allocated}", 13)
    text(870, 546, f"= {SOURCES} Attention GPUs x {BUCKET} tokens", 12, "sub")
    text(650, 572, "shapes", 13, weight=700)
    text(710, 572, f"recv_x [{allocated}, hidden]   recv_topk_idx [{allocated}, "
         f"{TOP_K}]   recv_topk_weights [{allocated}, {TOP_K}]", 12,
         mono=True)  # fmt: skip

    out.append('</g><g transform="translate(0,-60)">')

    # 3. Collect.
    box(20, 644, 1200, 40, "panel", "panel", 14)
    text(38, 669, "Attention GPUs sum each token's reply rows from all FFN GPUs",
         16, weight=700)  # fmt: skip
    text(1202, 669, "source_collect(): same output in both layouts", 12, "sub",
         anchor="end", mono=True)  # fmt: skip

    # Who writes -1 and 0.
    panel(20, 700, 590, 240, "Token layout: who writes -1 and 0",
          "dispatch_copy_epilogue.cuh")  # fmt: skip
    token, _, ids, weights = TOKENS[0]
    x0 = 190
    text(x0, 750, "x", 12, "sub", mono=True)
    text(x0 + 92, 750, "IDs, global", 12, "sub", mono=True)
    text(x0 + 228, 750, "weights", 12, "sub", mono=True)
    text(40, 774.5, "t0 as sent", 13, weight=700)
    cell(x0, 758, "x(t0)", "t0", 80)
    for k, (expert, weight) in enumerate(zip(ids, weights, strict=True)):
        cell(x0 + 92 + k * 42, 758, f"e{expert}", "t0" if local(expert) else "other")
        cell(x0 + 228 + k * 42, 758, f"{weight:.1f}",
             "t0" if local(expert) else "other")  # fmt: skip
    out.append(
        f'<path d="M316,786 V804 M171,832 V804 H461 V832 M167,826 L171,832 L175,826 '
        f'M457,826 L461,832 L465,826" fill="none" stroke="{c["orange"]}" '
        'stroke-width="2"/>'
    )
    text(326, 798, "copy into recv_x, kAfdTokenRoutes", 12, "orange", 700, mono=True)
    for gpu, x in enumerate((40, 330)):
        text(x, 850, f"FFN GPU {gpu}", 13, weight=700)
        text(x + 80, 850, f"experts {gpu * EXPERTS}-{(gpu + 1) * EXPERTS - 1}", 12,
             "sub")  # fmt: skip
        text(x, 868, "recv_topk_idx", 12, "sub", mono=True)
        text(x + 136, 868, "recv_topk_weights", 12, "sub", mono=True)
        for k, (expert, weight) in enumerate(zip(ids, weights, strict=True)):
            if expert // EXPERTS == gpu:
                fill = "t0" if gpu == 0 else "other"
                cell(x + k * 42, 874, str(expert - gpu * EXPERTS), fill)
                cell(x + 136 + k * 42, 874, f"{weight:.1f}", fill)
                text(x + k * 42 + 20, 909, f"= e{expert}", 10, "sub", anchor="middle")
            else:
                absent(x + k * 42, 874, "-1")
                absent(x + 136 + k * 42, 874, "0")
    text(40, 928, "local expert: stored as ID - first local ID", 12, "sub")
    text(330, 928, "expert on another GPU: ID -1, weight 0", 12, "orange")

    # Count-first and support.
    panel(630, 700, 590, 240, "When tokens are copied into recv_x",
          "AFDEP_COUNT_FIRST_MIN_BUCKET")  # fmt: skip
    for index, (label, first) in enumerate(
        (("payload-first", False), ("count-first", True))
    ):
        y = 742 + index * 44
        text(650, y + 16.5, label, 13, weight=700)
        cell(760, y, "counts", "other", 60)
        cell(824, y, "tokens arrive", "t2", 200)
        if first:
            cell(824, y + ROW - 2, "copy to recv_x on arrival", "cell", 200, "orange")
        else:
            cell(1028, y, "then copy to recv_x", "cell", 170, "orange")
    heads = ("Works with:", "one NVLink domain", "count-first copy",
             "several domains", "FP8 replies")  # fmt: skip
    table = (
        ("expert layout", "yes", "yes", "yes", "yes"),
        ("token layout", "yes", "yes", "no: error", "no: BF16"),
    )
    widths = (104, 112, 104, 124, 106)
    for r, line in enumerate((heads, *table)):
        x = 650
        for value, width in zip(line, widths, strict=True):
            y = 850 + r * 28
            if r == 0 and x == 650:
                text(x, y + 16.5, value, 12, weight=700)
            elif r == 0:
                text(x + width / 2, y + 16.5, value, 12, "sub", anchor="middle")
            elif x == 650:
                text(x, y + 16.5, value, 12, weight=700)
            elif value == "yes":
                cell(x + 2, y, value, "t3", width - 4)
            elif "error" not in value:
                cell(x + 2, y, value, "other", width - 4)
            else:
                cell(x + 2, y, value, "cell", width - 4, "orange", "orange")
            x += width

    # Across NVLink domains: how t0 reaches both FFN GPUs.
    panel(20, 956, 1200, 196, "Across NVLink domains: how t0 reaches both FFN GPUs",
          "allow_hybrid_mode")  # fmt: skip
    modes = (
        ("hybrid (True, default)", "1 RDMA write per NVLink domain"),
        ("direct (False)", "1 RDMA write per FFN GPU"),
    )
    for index, (label, note) in enumerate(modes):
        x = 40 + index * 600
        text(x, 1004, label, 13, weight=700)
        for side, names in enumerate((("Attention 0", "Attention 1"),
                                      ("FFN GPU 0", "FFN GPU 1"))):  # fmt: skip
            bx = x + side * 330
            box(bx, 1014, 150, 92, "bg", "line", 8, dash=True)
            text(bx + 10, 1030, f"NVLink domain {side}", 11, "sub")
            for row, name in enumerate(names):
                fill = "other" if name == "Attention 1" else "t0"
                cell(bx + 15, 1040 + row * 32, name, fill, 120)
        orange = f'fill="none" stroke="{c["orange"]}" stroke-width="2"'
        blue = f'fill="none" stroke="{c["blue"]}" stroke-width="2"'
        out.append(f'<path d="M{x + 137},1052 H{x + 343} m-7,-4 l7,4 l-7,4" {orange}/>')
        text(x + 240, 1045, note, 11, "orange", anchor="middle")
        if index == 0:
            out.append(
                f'<path d="M{x + 465},1052 H{x + 480} V1084 H{x + 467} '
                f'm7,-4 l-7,4 l7,4" {blue}/>'
            )
            text(x + 488, 1072, "NVLink copy", 11, "blue")
        else:
            out.append(
                f'<path d="M{x + 137},1058 C{x + 240},1058 {x + 240},1084 '
                f'{x + 343},1084 m-7,-4 l7,4 l-7,4" {orange}/>'
            )
    text(40, 1134, "The token record is the same in both layouts and both modes: "
         f"x + {TOP_K} IDs + {TOP_K} weights. Only the receiving GPU's copy kernel "
         "differs.", 12, "sub")  # fmt: skip
    out.append("</g>")
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    assets = Path(__file__).parent
    (assets / "token-layout.svg").write_text(draw(LIGHT))
    (assets / "token-layout-dark.svg").write_text(draw(DARK))


if __name__ == "__main__":
    main()
