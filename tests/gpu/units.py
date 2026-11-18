# SPDX-License-Identifier: Apache-2.0
"""One Buffer per Attention unit over shared FFN ranks; no model or performance claims.

Example: EP_DISABLE_GIN=1 torchrun --standalone --nproc-per-node=4 \
    -m tests.gpu.units --ffn-ranks 2 --units 2
The FFN ranks serve the units in unit order and wait for each one, as one Buffer does.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from datetime import timedelta

import torch
import torch.distributed as dist

from afdep import Config, Units
from afdep._build import source_digest
from tests.gpu.deepep import GRAPH_STEPS, Case, Exchange, verify

# Odd, so units differ in routes (epoch mod experts) as well as in payload:
# an exchange that reached another unit's Buffer fails verify.
UNIT_EPOCH_STRIDE = 37


def unit_case(case: Case, unit: int) -> Case:
    return replace(case, epoch=case.epoch + UNIT_EPOCH_STRIDE * (unit + 1))


def eager_checks(held, cases, *, host_counts):
    """Every rank runs its units one after another, each on a new exchange."""
    for case in cases:
        for unit, buffer in held:
            rows = buffer.config.capacity_tokens_per_rank
            if buffer.role == "attention":
                rows = case.counts[buffer.role_rank]
            exchange = Exchange(buffer, rows)
            exchange.update(unit_case(case, unit))
            snapshot = exchange.run(host_counts=host_counts, delay=True)
            torch.cuda.synchronize()
            verify(snapshot, unit_case(case, unit), buffer, rows)
        dist.barrier()


def graph_checks(held, cases, steps):
    """One captured graph per rank holds every unit it serves; inputs change."""
    if not held:
        return
    stream = torch.cuda.Stream()
    exchanges = [
        Exchange(buffer, buffer.config.capacity_tokens_per_rank) for _, buffer in held
    ]

    def update(case):
        for (unit, _), exchange in zip(held, exchanges, strict=True):
            exchange.update(unit_case(case, unit))
        stream.wait_stream(torch.cuda.current_stream())

    def run_units(**kwargs):
        return [exchange.run(**kwargs) for exchange in exchanges]

    update(cases[0])
    with torch.cuda.stream(stream):
        for _ in range(2):
            run_units()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = [run_units(delay=True, alternate=True) for _ in range(steps)]
    snapshots = []
    for case in cases:
        update(case)
        with torch.cuda.stream(stream):
            graph.replay()
            snapshots.append(
                [
                    [(value.clone(), counts) for value, counts in step]
                    for step in captured
                ]
            )
        torch.cuda.synchronize()
    for case, replay in zip(cases, snapshots, strict=True):
        for index, step in enumerate(replay):
            for (unit, buffer), exchange, snapshot in zip(
                held, exchanges, step, strict=True
            ):
                sign = -1 if index % 2 == 0 else 1
                verify(
                    snapshot, unit_case(case, unit), buffer, exchange.rows, sign=sign
                )
    graph.reset()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--hybrid", choices=("on", "off"), default="on")
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, choices=(2, 8), default=2)
    parser.add_argument("--experts-per-ffn", type=int, default=4)
    parser.add_argument("--ffn-ranks", type=int, default=2)
    parser.add_argument("--units", type=int, default=2)
    parser.add_argument("--num-sms", type=int, default=8)
    parser.add_argument("--graph-steps", type=int, default=GRAPH_STEPS)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo", timeout=timedelta(seconds=300))
    rank, world = dist.get_rank(), dist.get_world_size()
    sources, rest = divmod(world - args.ffn_ranks, args.units)
    if sources < 1 or rest:
        parser.error("the Attention ranks must split evenly into --units")
    config = Config(
        capacity_tokens_per_rank=args.capacity,
        hidden_size=args.hidden_size,
        num_experts=args.experts_per_ffn * args.ffn_ranks,
        top_k=args.top_k,
        ffn_ranks=args.ffn_ranks,
        attention_ranks=sources,
        payload=args.payload,
        num_sms=args.num_sms,
        allow_hybrid_mode=args.hybrid == "on",
    )
    units = Units(
        group=dist.group.WORLD,
        config=config,
        units=args.units,
        buffer_per_attention_unit=True,
    )
    is_ffn = rank < args.ffn_ranks
    my_unit = (rank - args.ffn_ranks) // sources
    held = list(enumerate(units)) if is_ffn else [(my_unit, units[0])]
    assert len(units) == (args.units if is_ffn else 1)
    assert all(buffer.config.world_size == args.ffn_ranks + sources for buffer in units)

    n = args.capacity
    full, empty = (n,) * sources, (0,) * sources
    inactive = tuple(0 if i == min(1, sources - 1) else n for i in range(sources))
    prefix = tuple(min((3, 1, 2)[i % 3], n) for i in range(sources))
    cases = [
        Case("mixed", full, 1),
        Case("changed_routes", full, 2),
        Case("skew", full, 3, "skew"),
        Case("inactive_source", inactive, 4),
        Case("all_skipped", full, 5, "skip"),
        Case("all_empty", empty, 6),
        Case("partial_prefix", prefix, 7),
    ]
    for host_counts in (False, True):
        eager_checks(held, cases, host_counts=host_counts)
    graph_checks(held, cases * 2, args.graph_steps)
    dist.barrier()
    # The last unit leaves: only its ranks and the FFN ranks take part in the close.
    # The other units keep exchanging on Buffers that were not rebuilt.
    remaining = [(unit, buffer) for unit, buffer in held if unit != args.units - 1]
    for unit, buffer in held:
        if unit == args.units - 1:
            buffer.close()
    eager_checks(remaining, cases, host_counts=False)
    graph_checks(remaining, cases, args.graph_steps)
    dist.barrier()

    summary = {
        "rank": rank,
        "role": "ffn" if is_ffn else "attention",
        "units_held": [unit for unit, _ in held],
        "units_after_one_left": [unit for unit, _ in remaining],
        "unit_ranks": args.ffn_ranks + sources,
        "gpu": torch.cuda.get_device_name(),
        "native_source_sha256": source_digest(),
        "registered_transport_bytes": [buffer.num_buffer_bytes for buffer in units],
    }
    summaries = [None] * world
    dist.all_gather_object(summaries, summary)
    units.close()
    dist.destroy_process_group()
    if rank == 0:
        print(
            json.dumps(
                {
                    "ok": True,
                    "validation": "one Buffer per Attention unit",
                    "payload": args.payload,
                    "hybrid": args.hybrid,
                    "units": args.units,
                    "eager_cases": len(cases) * 2,
                    "graph_replays": len(cases) * 2,
                    "eager_cases_after_one_left": len(cases),
                    "graph_replays_after_one_left": len(cases),
                    "ranks": summaries,
                }
            )
        )


if __name__ == "__main__":
    main()
