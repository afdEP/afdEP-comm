# SPDX-License-Identifier: Apache-2.0
"""Pull vs push of one producer's rows to R readers over NVLink (one node).

Push: rank 0 stores its rows into every reader's buffer (CTAS per reader).
Pull: every reader loads rank 0's rows into its own buffer (CTAS each).
Both copies are the same Triton grid-stride kernel on peer pointers (no copy
engine). Communication-only: no serving claims.
    torchrun --standalone --nproc-per-node=4 -m benchmarks.pull_push
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

from afdep._build import source_digest

TOKENS = (1, 8, 32, 128, 512)
HIDDEN = (2048, 7168)
CTAS = 8
BLOCK = 1024
WARMUP, ITERS = 5, 50
PRODUCER = 0


@triton.jit
def copy_kernel(src_ptrs, dst_ptrs, numel, BLOCK: tl.constexpr):
    """Grid (CTAS, pairs): program_id(1) selects one (src, dst) pointer pair."""
    pair = tl.program_id(1)
    src = tl.load(src_ptrs + pair).to(tl.pointer_type(tl.bfloat16))
    dst = tl.load(dst_ptrs + pair).to(tl.pointer_type(tl.bfloat16))
    for start in range(tl.program_id(0) * BLOCK, numel, tl.num_programs(0) * BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < numel
        tl.store(dst + offsets, tl.load(src + offsets, mask=mask), mask=mask)


def pattern(numel: int) -> torch.Tensor:
    return (torch.arange(numel, device="cuda") % 251).to(torch.bfloat16)


def run(handle, buffer, mode: str, numel: int, world: int) -> float | None:
    """Mean microseconds of one copy on this rank (None if it copies nothing)."""
    rank = dist.get_rank()
    peers = [handle.get_buffer(r, (numel,), torch.bfloat16) for r in range(world)]
    if mode == "push" and rank == PRODUCER:
        src = [buffer.data_ptr()] * (world - 1)
        dst = [peers[r].data_ptr() for r in range(world) if r != PRODUCER]
    elif mode == "pull" and rank != PRODUCER:
        src, dst = [peers[PRODUCER].data_ptr()], [buffer.data_ptr()]
    else:
        src = dst = []
    src_t = torch.tensor(src, dtype=torch.int64, device="cuda")
    dst_t = torch.tensor(dst, dtype=torch.int64, device="cuda")
    times = []
    for i in range(WARMUP + ITERS):
        if rank != PRODUCER:
            buffer.zero_()
        handle.barrier()  # every reader's buffer is cleared before any copy
        begin, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        begin.record()
        if src:
            copy_kernel[(CTAS, len(src))](src_t, dst_t, numel, BLOCK=BLOCK)
        end.record()
        handle.barrier()  # every copy is complete before readers check or clear
        if src and i >= WARMUP:
            times.append((begin, end))
        if i == 0 and rank != PRODUCER:
            torch.cuda.synchronize()
            assert torch.equal(buffer, pattern(numel)), f"{mode} rank {rank}"
    torch.cuda.synchronize()
    return statistics.mean(b.elapsed_time(e) * 1e3 for b, e in times) if times else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    world, rank = dist.get_world_size(), dist.get_rank()
    numel_max = max(TOKENS) * max(HIDDEN)
    buffer_all = symm_mem.empty(numel_max, dtype=torch.bfloat16, device="cuda")
    handle = symm_mem.rendezvous(buffer_all, dist.group.WORLD.group_name)
    rows = []
    for hidden in HIDDEN:
        for tokens in TOKENS:
            numel = tokens * hidden
            buffer = buffer_all[:numel]
            if rank == PRODUCER:
                buffer.copy_(pattern(numel))
            torch.cuda.synchronize()
            for mode in ("push", "pull"):
                us = run(handle, buffer, mode, numel, world)
                gathered = [None] * world
                dist.all_gather_object(gathered, us)
                # Push: the producer's copy; pull: the slowest reader's copy.
                us = max(t for t in gathered if t is not None)
                rows.append(
                    {
                        "mode": mode,
                        "tokens": tokens,
                        "hidden": hidden,
                        "readers": world - 1,
                        "us": us,
                        "bytes_per_reader": numel * 2,
                        "aggregate_GBps": numel * 2 * (world - 1) / us / 1e3,
                    }
                )
    if rank == 0:
        print(
            json.dumps(
                {
                    "gpu": torch.cuda.get_device_name(),
                    "ctas": CTAS,
                    "native_source_sha256": source_digest(),
                    "rows": rows,
                }
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
