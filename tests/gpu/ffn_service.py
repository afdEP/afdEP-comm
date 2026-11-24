# SPDX-License-Identifier: Apache-2.0
"""FFN as a service (afdep/service.py): one FFN process serves K Attention clients.

Scan kernel only (one GPU, fake arrival counters):
    python -m tests.gpu.ffn_service --kernel
Every client k is one Attention DP group with its own Buffer to the FFN ranks:
    torchrun --standalone --nproc-per-node=4 -m tests.gpu.ffn_service \
        --ffn 2 --clients 1,1 --mode host
Two policies for the FFN ranks, over the same Buffers:
    take everyone     --mode static        every unit in one round, waiting for all
    take first ready  --mode host|device   the first unit whose tokens have arrived
    take first layer group  --layers L     that unit and the next units ready at its
                                           layer (exchange index mod L), one exchange
Equal units get their Buffers from afdep.Units (--manual-buffers builds them here).
Communication-only evidence: no serving claims.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import time
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

from afdep import Buffer, Config, Units
from afdep._build import source_digest
from afdep.service import (
    CHOICE_DONE,
    CHOICE_STOP,
    CHOICE_TIMEOUT,
    ROUNDS_PER_LAUNCH,
    FfnService,
    client_row,
    order_ring,
)
from tests.gpu.deepep import RECORDED_ENV_PREFIXES, Case, Exchange, verify

EXPERTS_PER_FFN = 4
SEED = 1234
KERNEL_TIMEOUT_NS = 2_000_000
KERNEL_SENDERS, KERNEL_CTAS = 2, 2
KERNEL_DEVICE_QUOTA = 40  # several ROUNDS_PER_LAUNCH launches per client
MODES = ("host", "device", "static")
# Qwen3.5 MoE attention (config.json of the captured checkpoint).
HEAD_DIM, KV_HEADS, ROTARY_DIM, ROPE_THETA, NORM_EPS = 256, 2, 64, 1e7, 1e-6
MAX_PROMPT = 1024  # benchmarks/capture.py keeps prompts up to this length


class Replay:
    """One decoder layer of a captured checkpoint (benchmarks/capture.py) on one rank.

    Decode steps of this part's ShareGPT conversations under continuous batching:
    attention over the captured hidden states (the reference reply is teacher
    forced), router and shared expert here, routed experts on the FFN ranks.
    ``client=None`` is a prefill rank: every pass is the next ``rows`` prompt
    tokens of all conversations (a prompt cut by a pass continues in the next).
    """

    def __init__(self, path, client, clients, split, rounds, exchange):
        self.prefill = client is None
        self.exchange, rows = exchange, exchange.rows
        weights = torch.load(f"{path}/weights.pt", mmap=True)
        self.w = {
            k: v.cuda() for k, v in weights.items() if not k.startswith("experts.")
        }
        data = torch.load(f"{path}/hidden.pt", mmap=True)
        self.meta = data["meta"]
        offsets, prompts = data["offsets"].tolist(), data["prompt_len"].tolist()
        order = list(range(len(prompts)))
        if self.prefill:
            mine = order
        elif split == "length":  # client 0 holds the shortest conversations
            order.sort(key=lambda i: offsets[i + 1] - offsets[i])
            mine = order[
                client * len(order) // clients : (client + 1) * len(order) // clients
            ]
        else:
            mine = order[client::clients]
        ends = [
            offsets[i] + prompts[i] if self.prefill else offsets[i + 1] for i in mine
        ]
        take = torch.cat(
            [torch.arange(offsets[i], e) for i, e in zip(mine, ends, strict=True)]
        )
        self.h_in, self.h_out = data["h_in"][take].cuda(), data["h_out"][take].cuda()
        spans, first = [], 0  # (first token, first decode token, end) of a conversation
        for i, e in zip(mine, ends, strict=True):
            spans.append((first, first + prompts[i], first + e - offsets[i]))
            first = spans[-1][2]
        self.tokens = sum(end - begin for begin, _, end in spans)
        position = torch.cat([torch.arange(end - begin) for begin, _, end in spans])
        # Continuous batching: a finished sequence is replaced by the client's next one.
        generator, feed = random.Random(SEED + (client or 0)), itertools.cycle(spans)
        active = []
        for _ in range(
            0 if self.prefill else rows
        ):  # steady state: every row starts inside its reply
            begin, decode, end = next(feed)
            active.append([begin, generator.randrange(decode, end), end])
        pos, start, self.kv_len = [], [], []
        for _ in range(0 if self.prefill else rounds):
            pos.append([a[1] for a in active])
            start.append([a[0] for a in active])
            self.kv_len.append(max(a[1] - a[0] for a in active) + 1)
            for row, a in enumerate(active):
                a[1] += 1
                if a[1] == a[2]:
                    begin, decode, end = next(feed)
                    active[row] = [begin, decode, end]
        self.pos, self.start = torch.tensor(pos).cuda(), torch.tensor(start).cuda()
        if self.prefill:
            self.conv = torch.repeat_interleave(
                torch.arange(len(spans)),
                torch.tensor([end - begin for begin, _, end in spans]),
            ).cuda()
            first = torch.arange(rounds)[:, None] * rows
            self.pos = ((first + torch.arange(rows)) % self.tokens).cuda()
            self.start = self.pos - position.cuda()[self.pos]
            # Keys: the pass and the MAX_PROMPT tokens before it (a cut prompt's head).
            self.keys = torch.arange(-MAX_PROMPT, rows).cuda()
            self.kv_len = [rows + MAX_PROMPT] * rounds
        self.arange = torch.arange(max(self.kv_len), device="cuda")
        self.inv_freq = 1.0 / ROPE_THETA ** (
            torch.arange(0, ROTARY_DIM, 2, device="cuda").float() / ROTARY_DIM
        )
        # Keys and values of every position of every conversation.
        position = position.cuda()
        n = self.norm(self.h_in, "input_layernorm")
        k = (n @ self.w["self_attn.k_proj"].T).view(-1, KV_HEADS, HEAD_DIM)
        self.k = self.rope(self.norm(k, "self_attn.k_norm"), position)
        self.v = (n @ self.w["self_attn.v_proj"].T).view(-1, KV_HEADS, HEAD_DIM)
        self.errors = []

    def norm(self, x, name):
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + NORM_EPS)
        return (x * (1.0 + self.w[name].float())).bfloat16()

    def rope(self, x, position):
        angle = position[:, None].float() * self.inv_freq
        angle = torch.cat((angle, angle), -1)[:, None]
        rot, rest = x[..., :ROTARY_DIM], x[..., ROTARY_DIM:]
        half = torch.cat(
            (-rot[..., ROTARY_DIM // 2 :], rot[..., : ROTARY_DIM // 2]), -1
        )
        rot = rot * angle.cos().to(x.dtype) + half * angle.sin().to(x.dtype)
        return torch.cat((rot, rest), -1)

    def step(self, r: int) -> None:
        w, exchange = self.w, self.exchange
        pos, start, rows = self.pos[r], self.start[r], self.exchange.rows
        x = self.h_in[pos]
        n = self.norm(x, "input_layernorm")
        q, gate = (
            (n @ w["self_attn.q_proj"].T).view(rows, -1, 2 * HEAD_DIM).chunk(2, -1)
        )
        q = self.rope(self.norm(q, "self_attn.q_norm"), pos - start)
        if self.prefill:
            # ponytail: keys and values are projected ahead of the timed pass.
            idx = (pos[0] + self.keys) % self.tokens
            mask = (self.conv[idx] == self.conv[pos][:, None]) & (idx <= pos[:, None])
            o = F.scaled_dot_product_attention(
                q.transpose(0, 1)[None],
                self.k[idx].transpose(0, 1)[None],
                self.v[idx].transpose(0, 1)[None],
                attn_mask=mask,
                enable_gqa=True,
            )[0].transpose(0, 1)
        else:
            # ponytail: padded per-round copy of keys and values; a paged cache
            # avoids it.
            steps = self.arange[: self.kv_len[r]]
            idx = (start[:, None] + steps).clamp_(max=self.k.shape[0] - 1)
            mask = steps <= (pos - start)[:, None]
            o = F.scaled_dot_product_attention(
                q[:, :, None],
                self.k[idx].transpose(1, 2),
                self.v[idx].transpose(1, 2),
                attn_mask=mask[:, None, None],
                enable_gqa=True,
            )
        o = o.reshape(rows, -1) * torch.sigmoid(gate.reshape(rows, -1))
        mid = x + o @ w["self_attn.o_proj"].T
        n = self.norm(mid, "post_attention_layernorm")
        top, ids = F.softmax(n @ w["mlp.gate"].T, dim=-1, dtype=torch.float).topk(
            exchange.config.top_k
        )
        exchange.input.copy_(n)
        exchange.ids.copy_(ids)
        exchange.weights.copy_(top / top.sum(-1, keepdim=True))
        shared = F.silu(n @ w["mlp.shared_expert.gate_proj"].T) * (
            n @ w["mlp.shared_expert.up_proj"].T
        )
        shared = shared @ w["mlp.shared_expert.down_proj"].T
        self.local = mid + torch.sigmoid(n @ w["mlp.shared_expert_gate"].T) * shared

    def check(self, r: int, routed) -> None:
        """Relative error of the layer output against the checkpoint's own output."""
        want = self.h_out[self.pos[r]].float()
        got = (self.local + routed).float()
        self.errors.append((got - want).norm() / want.norm())


class Experts:
    """SwiGLU experts of one FFN rank over the expert-major received rows.

    Expert e's group runs to its aligned end, so every group offset is a multiple
    of the receive alignment; padding rows are computed and then zeroed.
    """

    def __init__(self, config: Config, gate_up, down):
        self.align = config.expert_alignment
        self.gate_up, self.down = gate_up.cuda(), down.cuda()

    def __call__(self, received):
        assert received.recv_scale is None, "BF16 payload only"
        ends = received.recv_count_psum
        offs = (((ends + self.align - 1) // self.align) * self.align).int()
        gate, up = torch._grouped_mm(received.recv_x, self.gate_up, offs=offs).chunk(
            2, -1
        )
        out = torch._grouped_mm(F.silu(gate) * up, self.down, offs=offs)
        starts = torch.cat((offs.new_zeros(1), offs[:-1]))
        rows = torch.arange(out.shape[0], device="cuda")[:, None]
        real = ((rows >= starts) & (rows < ends)).any(dim=1)
        weighted = out.float() * received.recv_topk_weights[:, None]
        return torch.where(real[:, None], weighted, 0).bfloat16()


def experts_check() -> dict:
    """Experts against a per-expert loop on uneven, padded expert segments."""
    config = Config(
        capacity_tokens_per_rank=8,
        hidden_size=256,
        num_experts=4,
        top_k=2,
        ffn_ranks=1,
        attention_ranks=1,
    )
    experts = Experts(  # random weights: this checks the grouped arithmetic only
        config,
        torch.randn(4, 256, 128, dtype=torch.bfloat16) / 16,
        torch.randn(4, 64, 256, dtype=torch.bfloat16) / 8,
    )
    counts = [3, 0, 130, 1]
    starts, ends, end = [], [], 0
    for c in counts:
        start = -(-end // config.expert_alignment) * config.expert_alignment
        starts.append(start)
        end = start + c
        ends.append(end)
    rows = -(-end // config.expert_alignment) * config.expert_alignment + 128
    x = torch.full((rows, 256), float("nan"), dtype=torch.bfloat16, device="cuda")
    weights = torch.rand(rows, device="cuda")
    for s, e in zip(starts, ends, strict=True):
        x[s:e] = torch.randn(e - s, 256, dtype=torch.bfloat16, device="cuda")
    received = SimpleNamespace(
        recv_x=x,
        recv_scale=None,
        recv_topk_weights=weights,
        recv_count_psum=torch.tensor(ends, dtype=torch.int32, device="cuda"),
    )
    got = experts(received).float()
    want = torch.zeros_like(got)
    for i, (s, e) in enumerate(zip(starts, ends, strict=True)):
        gate, up = (x[s:e] @ experts.gate_up[i]).chunk(2, -1)
        want[s:e] = (F.silu(gate) * up) @ experts.down[i] * weights[s:e, None]
    assert torch.isfinite(got).all()
    error = ((got - want).abs().max() / want.abs().max()).item()
    assert error < 0.02, error
    return {"relative_error": error}


def merge_rows(ends, real_rows, align):
    """Indices that merge K receives into one expert-major layout.

    ``ends[k]`` is receive k's ``recv_count_psum`` and ``real_rows[k]`` bounds its
    valid rows. Returns per receive (source row, merged row, valid) of static size
    ``real_rows[k]``, and the merged ``recv_count_psum``. In expert e's merged
    group the rows of receive k follow those of the receives before it.
    """
    ends = [e.long() for e in ends]
    starts = [
        torch.cat((e.new_zeros(1), (e[:-1] + align - 1) // align * align)) for e in ends
    ]
    counts = torch.stack([e - s for e, s in zip(ends, starts, strict=True)])
    earlier = counts.cumsum(0) - counts
    padded = (counts.sum(0) + align - 1) // align * align
    merged_starts = padded.cumsum(0) - padded
    index = []
    for k, start in enumerate(starts):
        psum = counts[k].cumsum(0)
        j = torch.arange(real_rows[k], device=psum.device)
        expert = torch.searchsorted(psum, j, right=True).clamp(max=len(psum) - 1)
        within = j - (psum[expert] - counts[k][expert])
        index.append(
            (
                start[expert] + within,
                merged_starts[expert] + earlier[k][expert] + within,
                j < psum[-1],
            )
        )
    return index, (merged_starts + counts.sum(0)).int()


def merged_pass(exchanges, experts) -> None:
    """Take Everyone over several Buffers: receive all, one expert pass, reply all.

    Ceiling: the rows are copied into the merged layout and back, about 5-8% of a
    round at 1024 tokens per rank. A receive that writes several Buffers into one
    layout would remove the copies.
    """
    received = [exchange.start() for exchange in exchanges]
    real_rows = [
        e.config.attention_ranks
        * e.rows
        * min(e.config.top_k, e.config.experts_per_rank)
        for e in exchanges
    ]
    index, psum = merge_rows(
        [r.recv_count_psum for r in received], real_rows, experts.align
    )
    experts_per_rank = len(psum)
    total = sum(real_rows) + (experts.align - 1) * experts_per_rank
    total = -(-total // experts.align) * experts.align
    # Row ``total`` collects the unused indices; Experts zeroes it as padding.
    x = torch.empty(
        (total + 1, received[0].recv_x.shape[1]), dtype=torch.bfloat16, device="cuda"
    )
    weights = torch.empty(total + 1, device="cuda")
    for r, (source, merged, valid) in zip(received, index, strict=True):
        merged = torch.where(valid, merged, total)
        rows, row_weights = r.recv_x, r.recv_topk_weights
        if experts.align > 1:  # without padding the valid rows are already a prefix
            source = source.clamp(max=len(rows) - 1)
            rows, row_weights = rows.index_select(0, source), row_weights[source]
        x.index_copy_(0, merged, rows)
        weights.index_copy_(0, merged, row_weights)
    out = experts(
        SimpleNamespace(
            recv_x=x, recv_scale=None, recv_topk_weights=weights, recv_count_psum=psum
        )
    )
    for exchange, r, (source, merged, valid) in zip(
        exchanges, received, index, strict=True
    ):
        reply = out.index_select(0, torch.where(valid, merged, total))
        if experts.align > 1:  # padding rows of a reply are unspecified
            rows = r.recv_x.shape[0]
            padded = torch.empty(
                (rows + 1, out.shape[1]), dtype=out.dtype, device="cuda"
            )
            reply = padded.index_copy_(0, torch.where(valid, source, rows), reply)[
                :rows
            ]
        exchange.buffer.worker_reply(reply, r.receipt)


def merge_check() -> dict:
    """merge_rows against a per-expert loop on uneven receives."""
    align, real_rows = 4, [9, 6, 5]
    counts = [[3, 0, 5, 1], [0, 0, 6, 0], [2, 1, 0, 2]]
    ends = []
    for row in counts:
        end, psum = 0, []
        for c in row:
            end = -(-end // align) * align + c
            psum.append(end)
        ends.append(torch.tensor(psum, dtype=torch.int32))
    index, psum = merge_rows(ends, real_rows, align)
    want_psum, want, end = [], [[] for _ in counts], 0
    for e in range(len(counts[0])):
        end = -(-end // align) * align
        for k, row in enumerate(counts):
            start = -(-(ends[k][e - 1].item() if e else 0) // align) * align
            want[k] += [(start + i, end + i) for i in range(row[e])]
            end += row[e]
        want_psum.append(end)
    assert psum.tolist() == want_psum, (psum, want_psum)
    for (source, merged, valid), pairs in zip(index, want, strict=True):
        got = list(zip(source[valid].tolist(), merged[valid].tolist(), strict=True))
        assert got == pairs, (got, pairs)
        assert int(valid.sum()) == len(pairs)
    return {"merged_rows": want_psum[-1]}


def fake_clients(count: int):
    """Arrival words of a role-path Buffer with one FFN rank and two senders."""
    ranks = 1 + KERNEL_SENDERS
    words = [
        torch.zeros(2 * ranks + 3, dtype=torch.int64, device="cuda")
        for _ in range(count)
    ]
    rows = [[w.data_ptr(), ranks, 1, KERNEL_SENDERS, KERNEL_CTAS] for w in words]
    return words, rows


def release(words, rounds: int = 1) -> None:
    """Every sender released ``rounds`` more rounds (afdep_signal_receivers)."""
    words[1 : 1 + KERNEL_SENDERS] += rounds * KERNEL_CTAS


def kernel_checks() -> dict:
    words, rows = fake_clients(3)
    ranks = 1 + KERNEL_SENDERS
    graphs = []
    for w in words:
        # A served exchange consumes one round, and its client sends the next one.
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        stream = torch.cuda.Stream()
        with torch.cuda.graph(graph, stream=stream):
            w[2 * ranks].add_(1)
            w[1 : 1 + KERNEL_SENDERS].add_(KERNEL_CTAS)
        graphs.append(graph)
    torch.cuda.synchronize()
    for w in words:
        w.zero_()
    service = FfnService(rows, graphs, timeout_ns=KERNEL_TIMEOUT_NS)
    assert service.scan() == CHOICE_TIMEOUT
    words[0][1] = KERNEL_CTAS  # sender 2 of client 0 has not arrived
    assert service.scan() == CHOICE_TIMEOUT
    release(words[1])
    assert service.scan() == 1  # FCFS: the only ready client
    assert service.scan() == CHOICE_TIMEOUT  # its next round was not released
    words[0][1] = 0
    for w in words:
        release(w)
    assert [service.scan() for _ in range(3)] == [2, 0, 1]  # round robin
    for w in words:
        release(w)
    release(words[0])
    service.rotate = False
    assert [service.scan() for _ in range(4)] == [0, 0, 1, 2]  # fixed priority
    service.rotate = True
    service.reset([1, 1, 1])
    assert [service.scan() for _ in range(4)] == [0, 1, 2, CHOICE_DONE]
    service.reset([0, 0, 0])
    service.stop()
    assert service.scan() == CHOICE_STOP
    results = {}
    for mode in ("host", "device"):
        for w in words:  # one released round each, so the order is round robin
            w.zero_()
            release(w)
        service.reset([KERNEL_DEVICE_QUOTA] * 3)
        before = [int(w[2 * ranks]) for w in words]
        choice = getattr(service, f"serve_{mode}")()
        served, order = service.served()
        torch.cuda.synchronize()
        assert choice == CHOICE_DONE, choice
        assert served == [KERNEL_DEVICE_QUOTA] * 3, served
        assert order == [0, 1, 2] * KERNEL_DEVICE_QUOTA, order[:12]
        rounds = [int(w[2 * ranks]) - b for w, b in zip(words, before, strict=True)]
        assert rounds == [KERNEL_DEVICE_QUOTA] * 3, rounds
        results[mode] = {
            "served": served,
            "device_launches": -(-len(order) // ROUNDS_PER_LAUNCH)
            if mode == "device"
            else 0,
        }
    service.close()
    results["groups"] = group_checks(words, rows)
    return results


def group_checks(words, rows) -> dict:
    """Take First Layer Group: the first ready client and neighbours at its layer."""
    k, ranks = len(words), 1 + KERNEL_SENDERS
    graphs = [None] * (k * k)
    for count in range(1, k + 1):
        for first in range(k - count + 1):
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
                for w in words[first : first + count]:
                    w[2 * ranks].add_(1)
                    w[1 : 1 + KERNEL_SENDERS].add_(KERNEL_CTAS)
            graphs[first + (count - 1) * k] = graph
    torch.cuda.synchronize()
    for w in words:
        w.zero_()
        release(w)
    service = FfnService(rows, graphs, timeout_ns=KERNEL_TIMEOUT_NS, layers=2)
    assert service.scan() == (k - 1) * k  # every client is ready at layer 0
    release(words[0])
    release(words[2])
    assert [service.scan() for _ in range(2)] == [0, 2]  # client 1 is not ready
    for w in words:
        release(w)
    # Client 1 is one layer behind its neighbours: no group.
    assert [service.scan() for _ in range(3)] == [0, 1, 2]
    release(words[1])
    assert service.scan() == 1  # client 1 catches up
    release(words[0])
    release(words[1])
    assert service.scan() == k  # clients 0 and 1; client 2 is not ready
    results = {}
    for mode in ("host", "device"):
        for w in words:
            w.zero_()
            release(w)
        service.reset([KERNEL_DEVICE_QUOTA] * k)
        choice = getattr(service, f"serve_{mode}")()
        served, picks = service.served()
        torch.cuda.synchronize()
        assert choice == CHOICE_DONE, choice
        assert served == [KERNEL_DEVICE_QUOTA] * k, served
        # A device scan runs beside the exchange that releases the next round, so
        # it may see a part of the group.
        assert mode == "device" or picks == [(k - 1) * k] * KERNEL_DEVICE_QUOTA
        assert [int(w[2 * ranks]) for w in words] == [KERNEL_DEVICE_QUOTA] * k
        results[mode] = {"served": served, "picks": len(picks)}
    service.close()
    return results


def layout(args):
    """(unit, ranks, tokens per rank) of every buffer; a unit has ``--dbo`` buffers."""
    sizes = [int(s) for s in args.clients.split(",")]

    def per_unit(text):
        values = [int(v) for v in text.split(",")]
        return values * len(sizes) if len(values) == 1 else values

    capacity, dbo = per_unit(args.capacity), per_unit(args.dbo)
    buffers, first = [], args.ffn
    for unit, size in enumerate(sizes):
        ranks = list(range(first, first + size))
        buffers += [(unit, ranks, capacity[unit] // dbo[unit])] * dbo[unit]
        first += size
    return sizes, buffers


def cases(units, rounds):
    """Round r of buffer k: changing routes, skew, prefixes, skipped and empty."""
    plans = []
    for _, ranks, capacity in units:
        size = len(ranks)
        full, empty = (capacity,) * size, (0,) * size
        prefix = tuple(min(1 + i, capacity) for i in range(size))
        kinds = [
            (full, "mixed"),
            (full, "skew"),
            (prefix, "mixed"),
            (full, "skip"),
            (empty, "mixed"),
        ]
        plans.append(
            [
                Case(f"r{r}", counts, r + 1, routing)
                for r, (counts, routing) in (
                    (r, kinds[r % len(kinds)]) for r in range(rounds)
                )
            ]
        )
    return plans


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else 0.0


def unit_config(args, capacity, attention_ranks) -> Config:
    return Config(
        capacity_tokens_per_rank=capacity,
        hidden_size=args.hidden_size,
        num_experts=args.experts_per_ffn * args.ffn,
        top_k=args.top_k,
        ffn_ranks=args.ffn,
        attention_ranks=attention_ranks,
        num_sms=args.num_sms,
        expert_alignment=args.expert_alignment,
    )


def serve_layout(args) -> dict:
    """Every client replays its captured exchange ``rounds`` times; FFN ranks serve."""
    rank, world = dist.get_rank(), dist.get_world_size()
    sizes, units = layout(args)
    if args.ffn + sum(sizes) != world:
        raise ValueError("--ffn plus client ranks must equal the world size")
    delays = [int(d) for d in args.delay.split(",")] if args.delay else [0] * len(sizes)
    is_ffn = rank < args.ffn
    buffers, held = [], None
    uniform = len({(len(ranks), capacity) for _, ranks, capacity in units}) == 1
    if uniform and len(units) == len(sizes) and not args.manual_buffers:
        # Equal units without micro-batches: afdep.Units builds one Buffer per unit.
        _, ranks, capacity = units[0]
        held = Units(
            group=dist.group.WORLD,
            config=unit_config(args, capacity, len(ranks)),
            units=len(units),
        )
        mine = range(len(units)) if is_ffn else [(rank - args.ffn) // len(ranks)]
        buffers = list(zip(mine, held, strict=True))
    else:
        # Every rank creates every group in the same order; members build its Buffer.
        for k, (_, ranks, capacity) in enumerate(units):
            group = dist.new_group(list(range(args.ffn)) + ranks)
            if is_ffn or rank in ranks:
                config = unit_config(args, capacity, len(ranks))
                buffers.append((k, Buffer(group=group, config=config)))
    # One set of expert weights per FFN rank, shared by every client it serves.
    experts = None
    if is_ffn and args.capture:
        weights = torch.load(f"{args.capture}/weights.pt", mmap=True)
        if len(weights["experts.gate_up"]) != args.experts_per_ffn * args.ffn:
            # The router emits every checkpoint expert id; ids are not checked later.
            raise ValueError(
                "--experts-per-ffn * --ffn must equal the capture's experts"
            )
        mine = slice(rank * args.experts_per_ffn, (rank + 1) * args.experts_per_ffn)
        experts = Experts(
            buffers[0][1].config,
            weights["experts.gate_up"][mine],
            weights["experts.down"][mine],
        )
    exchanges = [(k, Exchange(b, units[k][2], experts)) for k, b in buffers]
    grouped = args.layers > 0
    if grouped and args.mode == "static":
        raise ValueError("--layers needs --mode host or device")
    replays = []
    if args.capture and not is_ffn:
        # Every (rank, decode buffer) pair replays its own share of conversations.
        pairs = [
            (member, k)
            for k, (unit, ranks, _) in enumerate(units)
            if unit != args.prefill
            for member in ranks
        ]
        for k, exchange in exchanges:
            part = None if units[k][0] == args.prefill else pairs.index((rank, k))
            replays.append(
                Replay(
                    args.capture, part, len(pairs), args.split, args.rounds, exchange
                )
            )
    plan = cases(units, args.rounds)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for j, (k, exchange) in enumerate(exchanges):  # FFN ranks: buffer order
            if replays:
                replays[j].step(0)
            else:
                exchange.update(plan[k][0])
            for _ in range(2):
                exchange.run()
    torch.cuda.synchronize()
    graphs, outputs = [], []
    if is_ffn and experts and args.mode == "static" and len(exchanges) > 1:
        # Take Everyone: one graph serves every Buffer, with one expert pass for
        # micro-batch i of all units (a unit sends i + 1 after the reply to i).
        per_unit = [
            [exchange for k, exchange in exchanges if units[k][0] == unit]
            for unit in range(len(sizes))
        ]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for batch in itertools.zip_longest(*per_unit):
                merged_pass([e for e in batch if e is not None], experts)
        graphs.append(graph)
    elif is_ffn and grouped:
        # graphs[first + (count - 1) * K] serves Buffers [first, first + count).
        # One graph runs at a time, so all of them share one memory pool.
        graphs, pool = [None] * len(exchanges) ** 2, None
        for count in range(1, len(exchanges) + 1):
            for first in range(len(exchanges) - count + 1):
                span = [exchange for _, exchange in exchanges[first : first + count]]
                graph = torch.cuda.CUDAGraph(keep_graph=args.mode == "device")
                with torch.cuda.graph(graph, pool=pool, stream=stream):
                    if experts and count > 1:
                        merged_pass(span, experts)
                    else:
                        for exchange in span:
                            exchange.run()
                pool = graph.pool()
                graphs[first + (count - 1) * len(exchanges)] = graph
    else:
        for _, exchange in exchanges:
            graph = torch.cuda.CUDAGraph(keep_graph=args.mode == "device")
            with torch.cuda.graph(graph, stream=stream):
                outputs.append(exchange.run()[0])
            graphs.append(graph)
    torch.cuda.synchronize()
    dist.barrier()
    record = {"rank": rank}
    if is_ffn:
        service = None
        if args.mode != "static":
            rows = [client_row(exchange.buffer) for _, exchange in exchanges]
            service = FfnService(
                rows,
                graphs,
                rotate=not args.fixed,
                stream=stream,
                order=order_ring(exchanges[0][1].buffer) if args.leader else (0, 0, 0),
                layers=args.layers,
            )
            service.reset([args.rounds] * len(rows))
            if args.mode == "device":  # instantiation waits for an idle device
                service.build_device()
        dist.barrier()
        start = time.perf_counter()
        if service is None:  # baseline: fixed cyclic order, no readiness scan
            with torch.cuda.stream(stream):
                for _ in range(args.rounds):
                    for graph in graphs:
                        graph.replay()
            choice, served = CHOICE_DONE, [args.rounds] * len(exchanges)
            order = list(range(len(exchanges))) * args.rounds
        else:
            choice = getattr(service, f"serve_{args.mode}")()
            served, picks = service.served()
            n = len(served)
            # A pick is first + (count - 1) * K: the order lists its units.
            order = [u for p in picks for u in range(p % n, p % n + p // n + 1)]
            sizes = [p // n + 1 for p in picks]
            record["group_sizes"] = [sizes.count(c) for c in range(1, n + 1)]
        torch.cuda.synchronize()
        record.update(wall_s=time.perf_counter() - start, choice=choice, served=served)
        record["order"] = order
        assert choice == CHOICE_DONE, f"FFN rank {rank}: choice {choice}"
        assert served == [args.rounds] * len(served), served
        if service is not None:
            service.close()
    else:
        generator = random.Random(SEED + rank)
        events = [
            [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(args.rounds)
            ]
            for _ in exchanges
        ]
        # One stream per buffer: a unit's micro-batches overlap (dual batch overlap).
        streams = [stream] + [torch.cuda.Stream() for _ in exchanges[1:]]
        snapshots = []
        dist.barrier()
        start = time.perf_counter()
        for r in range(args.rounds):
            for j, (k, exchange) in enumerate(exchanges):
                begin, end = events[j][r]
                delay = delays[units[k][0]]
                with torch.cuda.stream(streams[j]):
                    if replays:
                        replays[j].step(r)
                    else:
                        exchange.update(plan[k][r])
                    if delay:  # Attention compute stand-in with a random length
                        torch.cuda._sleep(generator.randint(0, delay))
                    begin.record()
                    graphs[j].replay()
                    end.record()
                    if replays and args.verify:
                        replays[j].check(r, outputs[j])
                    elif args.verify:
                        snapshots.append((exchange, plan[k][r], outputs[j].clone()))
        torch.cuda.synchronize()
        latency = [[b.elapsed_time(e) * 1e3 for b, e in row] for row in events]
        record.update(
            client=exchanges[0][0],
            wall_s=time.perf_counter() - start,
            latency_us_p50=[percentile(row, 0.5) for row in latency],
            latency_us_p99=[percentile(row, 0.99) for row in latency],
        )
        for exchange, case, value in snapshots:
            verify((value, None), case, exchange.buffer, exchange.rows)
        record["verified_rounds"] = len(snapshots)
        if replays:
            record.update(capture=replays[0].meta, tokens=replays[0].tokens)
            record["kv_len_mean"] = sum(replays[0].kv_len) / args.rounds
            errors = [e for replay in replays for e in replay.errors]
            if errors:
                errors = torch.stack(errors)
                record["relative_error_mean"] = errors.mean().item()
                record["relative_error_max"] = errors.max().item()
    dist.barrier()
    record["buffers"] = "units" if held else "manual"
    if held:
        held.close()
    for _, buffer in buffers if not held else ():
        buffer.close()
    return record


def summarize(records, args) -> dict:
    ffn = [r for r in records if "order" in r]
    clients = [r for r in records if "client" in r]
    orders = [r.pop("order") for r in ffn]
    total = len(orders[0])
    diverged = sum(any(o[i] != orders[0][i] for o in orders[1:]) for i in range(total))
    gaps = []
    for order in orders:
        last = {}
        for i, c in enumerate(order):
            gaps.append(i - last.get(c, -1) - 1)
            last[c] = i
    rates = [args.rounds / r["wall_s"] for r in clients]
    jain = sum(rates) ** 2 / (len(rates) * sum(x * x for x in rates))
    # Exchanges served until the first buffer ends: every buffer is still active.
    served, active = [0] * (1 + max(orders[0])), total
    for i, c in enumerate(orders[0]):
        served[c] += 1
        if served[c] == args.rounds:
            active = i + 1
            break
    return {
        "exchanges_per_s": total / max(r["wall_s"] for r in ffn),
        "all_active_per_s": active / min(r["wall_s"] for r in clients),
        "all_active_served": served,
        "client_rates": rates,
        "order_divergence": diverged / total,
        "max_gap": max(gaps),
        "jain_client_rate": jain,
        "order_head": orders[0][:24],
        "ranks": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", action="store_true", help="scan kernel only")
    parser.add_argument("--ffn", type=int, default=2)
    parser.add_argument("--clients", default="1,1", help="Attention ranks per unit")
    parser.add_argument("--dbo", default="1", help="buffers per unit (micro-batches)")
    parser.add_argument("--prefill", type=int, default=-1, help="prefill unit index")
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="host",
        help="static: take everyone; host, device: take first ready",
    )
    parser.add_argument("--fixed", action="store_true", help="fixed priority scan")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--capacity", default="8", help="tokens per rank, per unit")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--experts-per-ffn", type=int, default=EXPERTS_PER_FFN)
    parser.add_argument("--num-sms", type=int, default=8)
    parser.add_argument("--delay", default="", help="max sleep cycles per client")
    parser.add_argument("--expert-alignment", type=int, default=128)
    parser.add_argument("--capture", default="", help="benchmarks/capture.py output")
    parser.add_argument("--split", choices=("random", "length"), default="random")
    parser.add_argument("--no-verify", dest="verify", action="store_false")
    parser.add_argument(
        "--manual-buffers",
        action="store_true",
        help="build the Buffers here even when afdep.Units fits the layout",
    )
    parser.add_argument(
        "--leader", action="store_true", help="FFN rank 0 picks, the others follow"
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=0,
        help="take first layer group: a unit's layer is its exchange index mod this",
    )
    args = parser.parse_args()
    local, gpus = int(os.environ.get("LOCAL_RANK", 0)), torch.cuda.device_count()
    if local >= gpus:  # more ranks than GPUs: the extra ones share the Attention GPUs
        local = args.ffn + (local - args.ffn) % (gpus - args.ffn)
    torch.cuda.set_device(local)
    base = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "native_source_sha256": source_digest(),
        "env": {
            k: v for k, v in os.environ.items() if k.startswith(RECORDED_ENV_PREFIXES)
        },
    }
    if args.kernel:
        print(
            json.dumps(
                {
                    **base,
                    "kernel": kernel_checks(),
                    "experts": experts_check(),
                    "merge": merge_check(),
                },
                indent=2,
            )
        )
        return
    dist.init_process_group("gloo", timeout=timedelta(seconds=300))
    record = serve_layout(args)
    records = [None] * dist.get_world_size()
    dist.all_gather_object(records, record)
    if dist.get_rank() == 0:
        summary = summarize(records, args)
        assert not args.leader or summary["order_divergence"] == 0
        print(json.dumps({**base, "args": vars(args), **summary}))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
