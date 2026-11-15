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

SCAN_SOURCE = r"""
#include <cuda_runtime.h>
extern "C" __global__ void afdep_ffn_scan(
        const long long* clients, const int num_clients, int* state,
        const int* quota, volatile int* control, int* log, const int log_capacity,
        const long long timeout_ns,
        const int rotate, const int rounds_per_launch, const int last_in_body,
        const cudaGraphConditionalHandle loop,
        const cudaGraphConditionalHandle branch) {
    if (threadIdx.x != 0)
        return;
    // Device mode: state[1] counts this launch's rounds, -1 once a scan ended it.
    const bool device = rounds_per_launch > 0, halted = device and state[1] < 0;
    long long start, now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    int choice = -3;
    for (bool found = halted; not found;) {
        if (control[0]) { choice = -1; break; }
        bool pending = false;
        for (int i = 1; i <= num_clients and not found; ++ i) {
            const int c = ((rotate ? state[0] : num_clients - 1) + i) % num_clients;
            if (quota[c] > 0 and state[3 + c] >= quota[c])
                continue;
            pending = true;
            const long long* w = clients + c * 6;
            const volatile long long* arrival =
                reinterpret_cast<const volatile long long*>(w[0]);
            // Ready: every sender released the next unissued round (as
            // afdep_wait_senders), so a scan may run ahead of the exchanges.
            const long long target = (w[5] + state[3 + c] + 1) * w[4];
            bool ready = true;
            for (long long s = w[2]; s < w[2] + w[3] and ready; ++ s)
                ready = arrival[s] >= target;
            if (ready) { choice = c; found = true; }
        }
        if (not found and not pending) { choice = -2; break; }
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
        if (not found and now - start > timeout_ns) { choice = -3; break; }
    }
    if (halted) {  // an earlier scan in this body ended the launch
        choice = -3;
    } else if (choice >= 0) {
        state[0] = choice;
        state[3 + choice] += 1;
        if (state[2] < log_capacity)
            log[state[2]] = choice;
        state[2] += 1;
    }
    if (not halted)
        control[1] = choice;  // the host reads it after the stream completes
    if (device) {
        if (not halted)
            state[1] = choice >= 0 ? state[1] + 1 : -1;
        cudaGraphSetConditional(branch, choice >= 0 ? choice : num_clients);
        if (last_in_body) {
            const bool more = state[1] >= 0 and state[1] < rounds_per_launch;
            if (not more)
                state[1] = 0;
            cudaGraphSetConditional(loop, more ? 1 : 0);
        }
    }
}
"""


def _ok(result):
    """cuda-python returns (error, *values)."""
    error, *values = result
    if int(error) != 0:
        raise RuntimeError(f"CUDA error {error!r}")
    return values[0] if len(values) == 1 else values


def _load_scan(name: bytes = b"afdep_ffn_scan"):
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
                    "-o",
                    f"{work}/scan.cubin",
                    str(source),
                ],
                check=True,
            )
            os.replace(f"{work}/scan.cubin", cubin)
    _ok(driver.cuInit(0))
    module = _ok(driver.cuModuleLoad(str(cubin).encode()))
    return module, _ok(driver.cuModuleGetFunction(module, name))


def client_row(buffer: Buffer) -> list[int]:
    """The scan's view of one client Buffer on this FFN rank (5 words)."""
    if buffer.role != "ffn":
        raise ValueError("FfnService runs on FFN ranks")
    base, directed, per_cta = buffer.transport._runtime.get_afd_arrival_words()
    if not directed:
        raise ValueError("FfnService needs the directed exit of one NVLink domain")
    c = buffer.config
    ctas = buffer.transport._resolved_num_sms if per_cta else 1
    return [base, c.world_size, c.ffn_ranks, c.attention_ranks, ctas]


class FfnService:
    """Serve ``graphs[k]`` (client k's captured FFN exchange) when client k is ready.

    ``rows[k]`` is ``client_row`` of this FFN rank's Buffer for client k.
    ``quota[k]`` exchanges end client k (0: unlimited; stop with ``stop()``).
    Graphs for device mode need ``torch.cuda.CUDAGraph(keep_graph=True)``.
    """

    def __init__(
        self,
        rows: list[list[int]],
        graphs: list[torch.cuda.CUDAGraph],
        *,
        rotate: bool = True,
        timeout_ns: int = DEFAULT_TIMEOUT_NS,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        if not rows or len(rows) != len(graphs):
            raise ValueError("FfnService needs one graph per client")
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
        )
        return values, (p, i, p, p, p, p, i, ll, i, i, i, ull, ull)

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
