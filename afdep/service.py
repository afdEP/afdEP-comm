# SPDX-License-Identifier: Apache-2.0
"""FFN as a service (local experiment): one FFN process serves K client Buffers.

Each client (an Attention DP group) owns one role-path Buffer with the FFN ranks.
A one-thread scan kernel peeks each client's dispatch arrival counters (read only;
the dispatch kernel still waits with acquire) and picks the first ready client,
starting after the one served last: FCFS when one is ready, round robin when
several are. Readiness counts rounds the scan already issued, so host mode scans
on its own stream while the previous exchange runs and queues the chosen client's
captured exchange behind it. Device mode runs ROUNDS_PER_BODY rounds per WHILE
iteration inside one CUDA graph, each scan beside the previous round's
switch(client graph), for up to ROUNDS_PER_LAUNCH rounds per CPU launch.
With ``order_ring``, FFN rank 0 publishes its picks in NCCL window words and the
other FFN ranks serve that order instead of picking on their own.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import torch
from cuda.bindings import driver, runtime

from ._build import nccl_paths
from .deepep import Buffer

CHOICE_STOP, CHOICE_DONE, CHOICE_TIMEOUT = -1, -2, -3
ROUNDS_PER_LAUNCH = 32
ROUNDS_PER_BODY = 4  # unrolled rounds per WHILE iteration; divides ROUNDS_PER_LAUNCH
# Arrival base, Buffer ranks, first sender, senders, arrivals per sender, and the
# dispatch round at reset (written by reset).
CLIENT_WORDS = 6
STATE_LAST, STATE_ROUNDS, STATE_TOTAL, STATE_SERVED = 0, 1, 2, 3
CONTROL_STOP, CONTROL_CHOICE = 0, 1
DEFAULT_TIMEOUT_NS = 30_000_000_000
LOG_CAPACITY = 1 << 20
SCAN_THREADS = 32
# The leader's order ring: arrival words no exit uses (above 2 * 1024 ranks + 3,
# below the count-first words at 4110). A leader is at most one pick per client
# ahead of a follower, because a client sends again only after every reply.
ORDER_BASE, ORDER_WORDS = 2064, 1024

SCAN_SOURCE = Path(__file__).with_suffix(".cu").read_text()


def _ok(result):
    """cuda-python returns (error, *values)."""
    error, *values = result
    if int(error) != 0:
        raise RuntimeError(f"CUDA error {error!r}")
    return values[0] if len(values) == 1 else values


def _load_scan():
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
    key = hashlib.sha256((SCAN_SOURCE + arch).encode()).hexdigest()[:16]
    cache = Path(os.environ.get("AFDEP_CACHE_DIR", Path.home() / ".cache" / "afdep"))
    cubin = cache / f"ffn_scan_{key}.cubin"
    if not cubin.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as work:
            source = Path(work) / "scan.cu"
            source.write_text(SCAN_SOURCE)
            nvcc = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin" / "nvcc"
            subprocess.run(
                [
                    str(nvcc),
                    "-cubin",
                    f"-arch={arch}",
                    f"-I{nccl_paths()[0]}",
                    "-o",
                    f"{work}/scan.cubin",
                    str(source),
                ],
                check=True,
            )
            os.replace(f"{work}/scan.cubin", cubin)
    _ok(driver.cuInit(0))
    module = _ok(driver.cuModuleLoad(str(cubin).encode()))
    return module, _ok(driver.cuModuleGetFunction(module, b"afdep_ffn_scan"))


def client_row(buffer: Buffer) -> list[int]:
    """The scan's view of one client Buffer on this FFN rank (5 words)."""
    if buffer.role != "ffn":
        raise ValueError("FfnService runs on FFN ranks")
    base, directed, per_cta, _ = buffer.transport._runtime.get_afd_arrival_words()
    if not directed:
        raise ValueError("FfnService needs the directed exit of one NVLink domain")
    c = buffer.config
    ctas = buffer.transport._resolved_num_sms if per_cta else 1
    return [base, c.world_size, c.ffn_ranks, c.attention_ranks, ctas]


def order_ring(buffer: Buffer) -> tuple[int, int, int]:
    """(this FFN rank's order ring words, the Buffer's NCCL window, this rank)."""
    if buffer.role != "ffn":
        raise ValueError("FfnService runs on FFN ranks")
    base, directed, _, window = buffer.transport._runtime.get_afd_arrival_words()
    if not directed:  # a follower loads the leader's word through the window mapping
        raise ValueError("FfnService needs the directed exit of one NVLink domain")
    return base + ORDER_BASE * 8, window, buffer.rank


class FfnService:
    """Serve ``graphs[k]`` (client k's captured FFN exchange) when client k is ready.

    ``rows[k]`` is ``client_row`` of this FFN rank's Buffer for client k.
    ``quota[k]`` exchanges end client k (0: unlimited; stop with ``stop()``).
    Graphs for device mode need ``torch.cuda.CUDAGraph(keep_graph=True)``.
    ``order`` is ``order_ring`` of one client Buffer (the same client on every FFN
    rank): every FFN rank then serves FFN rank 0's order. Every rank must
    ``reset`` before any rank serves.
    """

    def __init__(
        self,
        rows: list[list[int]],
        graphs: list[torch.cuda.CUDAGraph],
        *,
        rotate: bool = True,
        timeout_ns: int = DEFAULT_TIMEOUT_NS,
        stream: torch.cuda.Stream | None = None,
        order: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        if not rows or len(rows) != len(graphs):
            raise ValueError("FfnService needs one graph per client")
        if len(graphs) > ORDER_WORDS:
            raise ValueError(f"FfnService serves at most {ORDER_WORDS} clients")
        self.order = order
        self.graphs, self.rotate, self.timeout_ns = graphs, rotate, timeout_ns
        self.stream = stream or torch.cuda.Stream()
        self.scan_stream = torch.cuda.Stream()
        self.rows = rows
        self.clients = torch.tensor(
            [[*row, 0] for row in rows], dtype=torch.int64, device="cuda"
        )
        self.state = torch.zeros(
            STATE_SERVED + len(graphs), dtype=torch.int32, device="cuda"
        )
        self.quota = torch.zeros(len(graphs), dtype=torch.int32, device="cuda")
        self.log = torch.full((LOG_CAPACITY,), -1, dtype=torch.int32, device="cuda")
        # Pinned host words are device-visible through UVA: [stop, last choice].
        self.control = torch.zeros(2, dtype=torch.int32, pin_memory=True)
        self._module, self._function = _load_scan()
        self._exec = self._graph = None
        self.reset([0] * len(graphs))

    def reset(self, quota: list[int]) -> None:
        """New quotas; served counts, the order log and the stop flag restart."""
        torch.cuda.synchronize()
        self.quota.copy_(torch.tensor(quota, dtype=torch.int32))
        self.state.zero_()
        self.state[STATE_LAST] = len(self.graphs) - 1
        self.log.fill_(-1)
        self.control.zero_()
        if self.order[0] and not self.order[2]:  # the leader's ring
            _ok(driver.cuMemsetD32(self.order[0], 0, 2 * ORDER_WORDS))
        for c, (base, ranks, *_) in enumerate(self.rows):  # dispatch round word
            _ok(
                driver.cuMemcpyDtoD(
                    self.clients.data_ptr() + (c * CLIENT_WORDS + 5) * 8,
                    base + 2 * ranks * 8,
                    8,
                )
            )
        torch.cuda.synchronize()

    def stop(self) -> None:
        """Ask a spinning scan to return CHOICE_STOP (safe from another thread)."""
        self.control[CONTROL_STOP] = 1

    def _args(self, rounds: int, loop: int = 0, branch: int = 0, last: int = 0):
        p, i, ll, ull = (
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_longlong,
            ctypes.c_ulonglong,
        )
        values = (
            self.clients.data_ptr(),
            len(self.graphs),
            self.state.data_ptr(),
            self.quota.data_ptr(),
            self.control.data_ptr(),
            self.log.data_ptr(),
            LOG_CAPACITY,
            self.timeout_ns,
            int(self.rotate),
            rounds,
            last,
            loop,
            branch,
            self.order[0],
            ORDER_WORDS,
            *self.order[1:],
        )
        return values, (p, i, p, p, p, p, i, ll, i, i, i, ull, ull, p, i, p, i)

    def _launch_scan(self, stream: int, args) -> None:
        _ok(
            driver.cuLaunchKernel(
                self._function,
                1,
                1,
                1,
                SCAN_THREADS,
                1,
                1,
                0,
                driver.CUstream(stream),
                args,
                0,
            )
        )

    def scan(self) -> int:
        """Issue one round: the next ready client, or a negative choice (blocks)."""
        self._launch_scan(self.scan_stream.cuda_stream, self._args(0))
        self.scan_stream.synchronize()
        return int(self.control[CONTROL_CHOICE])

    def serve_host(self) -> int:
        """Host-driven: queue each issued client's graph, until no choice."""
        with torch.cuda.stream(self.stream):
            while (choice := self.scan()) >= 0:
                self.graphs[choice].replay()
        return choice

    def build_device(self) -> None:
        """Device-driven: WHILE(ROUNDS_PER_BODY x (scan -> SWITCH(client graphs))).

        Scan k+1 runs beside switch k and after switch k-1, so it issues one round
        ahead; switches run in issue order.
        """
        graph = _ok(runtime.cudaGraphCreate(0))
        loop = _ok(
            runtime.cudaGraphConditionalHandleCreate(
                graph,
                1,
                runtime.cudaGraphConditionalHandleFlags.cudaGraphCondAssignDefault,
            )
        )
        body = self._conditional(graph, [], loop, "cudaGraphCondTypeWhile", 1)[0]
        scans, switches = [], []
        for k in range(ROUNDS_PER_BODY):
            branch = _ok(
                runtime.cudaGraphConditionalHandleCreate(body, len(self.graphs), 0)
            )
            args = self._args(
                ROUNDS_PER_LAUNCH, int(loop), int(branch), k == ROUNDS_PER_BODY - 1
            )
            scans.append(self._capture_scan(body, scans[-1:] + switches[-2:-1], args))
            switch, bodies = self._conditional(
                body,
                [scans[-1], *switches[-1:]],
                branch,
                "cudaGraphCondTypeSwitch",
                len(self.graphs),
                node=True,
            )
            switches.append(switch)
            for child, client in zip(bodies, self.graphs, strict=True):
                _ok(
                    runtime.cudaGraphAddChildGraphNode(
                        child, None, 0, runtime.cudaGraph_t(client.raw_cuda_graph())
                    )
                )
        self._graph = graph
        self._exec = _ok(runtime.cudaGraphInstantiate(graph, 0))
        _ok(runtime.cudaGraphUpload(self._exec, self.stream.cuda_stream))

    def _capture_scan(self, graph, dependencies, args):
        """Add one scan kernel node to ``graph`` after ``dependencies``."""
        before = {int(n) for n in self._nodes(graph)}
        capture = torch.cuda.Stream()
        _ok(
            runtime.cudaStreamBeginCaptureToGraph(
                capture.cuda_stream,
                graph,
                dependencies or None,
                None,
                len(dependencies),
                runtime.cudaStreamCaptureMode.cudaStreamCaptureModeRelaxed,
            )
        )
        self._launch_scan(capture.cuda_stream, args)
        _ok(runtime.cudaStreamEndCapture(capture.cuda_stream))
        (node,) = [n for n in self._nodes(graph) if int(n) not in before]
        return node

    @staticmethod
    def _nodes(graph) -> list:
        count = _ok(runtime.cudaGraphGetNodes(graph, 0))[1]
        return list(_ok(runtime.cudaGraphGetNodes(graph, count))[0])[:count]

    @staticmethod
    def _conditional(graph, dependencies, handle, kind: str, size: int, node=False):
        params = runtime.cudaGraphNodeParams()
        params.type = runtime.cudaGraphNodeType.cudaGraphNodeTypeConditional
        params.conditional.handle = handle
        params.conditional.type = getattr(runtime.cudaGraphConditionalNodeType, kind)
        params.conditional.size = size
        added = _ok(
            runtime.cudaGraphAddNode(
                graph, dependencies, None, len(dependencies), params
            )
        )
        bodies = list(params.conditional.phGraph_out)[:size]
        return (added, bodies) if node else bodies

    def serve_device(self) -> int:
        """Relaunch the device loop (ROUNDS_PER_LAUNCH rounds each) until no choice."""
        if self._exec is None:
            self.build_device()
        while True:
            _ok(runtime.cudaGraphLaunch(self._exec, self.stream.cuda_stream))
            self.stream.synchronize()
            if (choice := int(self.control[CONTROL_CHOICE])) < 0:
                return choice

    def served(self) -> tuple[list[int], list[int]]:
        """(exchanges served per client, service order) since the last reset."""
        state = self.state.tolist()
        total = min(state[STATE_TOTAL], LOG_CAPACITY)
        return state[STATE_SERVED:], self.log[:total].tolist()

    def close(self) -> None:
        if self._exec is not None:
            _ok(runtime.cudaGraphExecDestroy(self._exec))
            _ok(runtime.cudaGraphDestroy(self._graph))
            self._exec = self._graph = None
        if self._module is not None:
            _ok(driver.cuModuleUnload(self._module))
            self._module = None
