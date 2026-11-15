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

Resident mode launches one kernel for the whole run. It picks clients the same
way and does the receive and the reply itself; the experts run either as CUDA
graphs on a second stream (persistent kernel) or inside it (megakernel).
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

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
RESIDENT_THREADS = 256
EXTRA_WORDS = 12  # afdep_ffn_resident's Client fields after the client row
SHARED_FINISHED, SHARED_RING, RING_SIZE = 1, 10, 16  # afdep_ffn_resident's words
UNIT_ROWS = 8  # rows per megakernel GEMM unit
# ncclWindow_vidmem as int32 words: LSA rank and the peer stride in 4 GiB units.
WINDOW_WORDS, WINDOW_LSA_RANK, WINDOW_STRIDE = 8, 4, 6

SCAN_SOURCE = r"""
#include <cassert>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

static __device__ long long timer() {
    long long now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    return now;
}

// The first ready client after the one served last; -2: every quota is met,
// -4: none is ready.
static __device__ int next_client(const long long* clients, const int num_clients,
                                  const int* state, const int* quota,
                                      const int rotate) {
    bool pending = false;
    for (int i = 1; i <= num_clients; ++ i) {
        const int c = ((rotate ? state[0] : num_clients - 1) + i) % num_clients;
        if (quota[c] > 0 and state[3 + c] >= quota[c])
            continue;
        pending = true;
        const long long* w = clients + c * 6;
        const volatile long long* arrival =
            reinterpret_cast<const volatile long long*>(w[0]);
        // Ready: every sender released the next unissued round (as
        // afdep_wait_senders), so a pick may run ahead of the exchanges.
        const long long target = (w[5] + state[3 + c] + 1) * w[4];
        bool ready = true;
        for (long long s = w[2]; s < w[2] + w[3] and ready; ++ s)
            ready = arrival[s] >= target;
        if (ready)
            return c;
    }
    return pending ? -4 : -2;
}

static __device__ void record(int* state, int* log, const int log_capacity,
    const int choice) {
    state[0] = choice;
    state[3 + choice] += 1;
    if (state[2] < log_capacity)
        log[state[2]] = choice;
    state[2] += 1;
}

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
    const long long start = timer();
    int choice = -3;
    while (not halted) {
        if (control[0]) { choice = -1; break; }
        choice = next_client(clients, num_clients, state, quota, rotate);
        if (choice != -4)
            break;
        if (timer() - start > timeout_ns) { choice = -3; break; }
    }
    if (not halted and choice >= 0)
        record(state, log, log_capacity, choice);
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

enum { GO, FINISHED, PHASE, CLIENT, UNITS, NEXT, RECEIVED, COMPUTED, ISSUED, HALT,
    RING };
enum { IDLE, COUNT, COPY, GATE_UP, DOWN, REPLY, EXIT };
const int RING_SIZE = 16, UNIT_ROWS = 8;
// WorkspaceLayout: received rank and expert counts.
const int COUNT_BYTES = 16 + 2 * 3072 * 8;

struct Client {
    long long arrival, ctas, workspace, buffer, peer_bytes;
    int ranks, ffn, me, bucket, capacity;
    __nv_bfloat16 *x, *reply;
    float* weights;
    int *psum, *meta, *rank_psum;
    __device__ Client(const long long* w, const long long* e)
        : arrival(w[0]), ctas(w[4]), workspace(e[0]), buffer(e[1]), peer_bytes(e[2]),
          ranks(w[1]), ffn(w[2]), me(e[3]), bucket(e[4]), capacity(e[11]),
          x(reinterpret_cast<__nv_bfloat16*>(e[5])),
              reply(reinterpret_cast<__nv_bfloat16*>(e[9])),
          weights(reinterpret_cast<float*>(e[6])), psum(reinterpret_cast<int*>(e[7])),
          meta(reinterpret_cast<int*>(e[8])),
              rank_psum(reinterpret_cast<int*>(e[10])) {}
};

// A published count, then cleared.
static __device__ int take(volatile long long* word) {
    int value;
    do value = -static_cast<int>(*word) - 1; while (value < 0);
    *word = 0;
    return value;
}

// One launch serves the whole run. CTA 0 thread 0 leads: it picks a client and
// walks every CTA through count, copy, (gate_up, down), reply. Without expert
// weights the reply waits for the gate kernel's graph on another stream.
extern "C" __global__ void afdep_ffn_resident(
        const long long* clients, const long long* extra, const int num_clients,
            int* state,
        const int* quota, volatile int* control, int* log, const int log_capacity,
        const long long timeout_ns, const int rotate, volatile int* shared,
            const int depth,
        const int hidden, const int topk, const int experts, const int alignment,
        const __nv_bfloat16* gate_up, const __nv_bfloat16* down, const int inter,
        __nv_bfloat16* scratch, int* work) {
    extern __shared__ __nv_bfloat16 staged[];
    __shared__ int unit;
    const int tid = threadIdx.x, lane = tid & 31, threads = blockDim.x, ctas =
        gridDim.x;
    const int global = blockIdx.x * threads + tid, warp = global / 32, warps = ctas *
        threads / 32;
    const bool leader = global == 0, mega = gate_up != nullptr;
    const int hidden_bytes = hidden * 2, meta_at = (hidden_bytes + 31) / 32 * 32;
    const int token_bytes = meta_at + (topk * 12 + 4 + 31) / 32 * 32;
    const int reply_bytes = meta_at + (topk * 8 + 31) / 32 * 32, meta_words = 2 + topk;
    int phase = IDLE, client = 0, received = 0, replied = 0, items = 0;  // leader only
    for (int step = 1;; ++ step) {
        if (leader) {
            while (shared[FINISHED] != ctas) {}
            shared[FINISHED] = 0;
            const Client c(clients + client * 6, extra + client * 12);
            if (phase == COUNT) {
                c.psum[0] = 0;
                for (int e = 0; e < experts; ++ e)
                    c.psum[e + 1] += c.psum[e];
                for (int s = c.ffn; s < c.ranks; ++ s) {
                    assert(c.rank_psum[s] <= c.bucket);
                    c.rank_psum[s] += c.rank_psum[s - 1];
                }
                assert(c.psum[experts] <= c.capacity);
                ++ *reinterpret_cast<long long*>(c.arrival + 2 * c.ranks * 8);
                phase = COPY;
            } else if (phase == COPY) {
                shared[RING + received % RING_SIZE] = client;
                __threadfence();
                shared[RECEIVED] = ++ received;
                phase = IDLE;
                if (mega) {
                    items = 0;
                    for (int e = 0; e < experts; ++ e) {
                        const int end = e ? c.psum[e - 1] : 0;
                        const int first = (end + alignment - 1) / alignment * alignment;
                        for (int r = first; r < c.psum[e]; r += UNIT_ROWS, ++ items) {
                            work[3 * items] = r;
                            work[3 * items + 1] = e;
                            work[3 * items + 2] = min(UNIT_ROWS, c.psum[e] - r);
                        }
                    }
                    shared[UNITS] = items * (2 * inter / threads);
                    shared[NEXT] = 0;
                    phase = GATE_UP;
                }
            } else if (phase == GATE_UP) {
                shared[UNITS] = items * (hidden / threads);
                shared[NEXT] = 0;
                phase = DOWN;
            } else if (phase == DOWN) {
                phase = REPLY;
            } else if (phase == REPLY) {
                for (int s = c.ffn; s < c.ranks; ++ s)
                    atomicAdd(reinterpret_cast<unsigned long long*>(
                                  c.arrival + (c.ranks + c.me) * 8 + (s - c.me) *
                                      c.peer_bytes),
                              static_cast<unsigned long long>(c.ctas));
                ++ replied;
                phase = IDLE;
            }
            const long long start = timer();
            for (int end = 0; phase == IDLE;) {
                if (not mega and shared[COMPUTED] != replied) {
                    phase = REPLY;
                    client = shared[RING + replied % RING_SIZE];
                    break;
                }
                int choice = -4;
                if (received - replied < depth) {
                    choice = control[0] ? -1 : next_client(clients, num_clients,
                        state, quota, rotate);
                    if (choice >= 0) {
                        record(state, log, log_capacity, choice);
                        phase = COUNT;
                        client = choice;
                        break;
                    }
                }
                if (received == replied and choice != -4)
                    end = choice;
                else if (timer() - start > timeout_ns)
                    end = -3;
                if (end) {
                    control[1] = end;
                    shared[HALT] = 1;
                    phase = EXIT;
                }
            }
            shared[PHASE] = phase;
            shared[CLIENT] = client;
            __threadfence();
            shared[GO] = step;
        }
        if (tid == 0)
            while (shared[GO] != step) {}
        __syncthreads();
        const int p = shared[PHASE];
        if (p == EXIT)
            return;
        const Client c(clients + shared[CLIENT] * 6, extra + shared[CLIENT] * 12);
        if (p == COUNT) {
            volatile long long* count =
                reinterpret_cast<volatile long long*>(c.workspace + COUNT_BYTES);
            if (global < experts) {
                int sum = 0;
                for (int s = c.ffn; s < c.ranks; ++ s)
                    sum += take(count + c.ranks + experts * s + global);
                c.psum[global + 1] = (sum + alignment - 1) / alignment * alignment;
            } else if (global < experts + c.ranks - c.ffn) {
                c.rank_psum[c.ffn + global - experts] = take(count + c.ffn +
                    global - experts);
            }
        } else if (p == COPY) {
            // As dispatch_copy_epilogue: one warp per token, one row per local expert.
            const int total = c.rank_psum[c.ranks - 1];
            int s = c.ffn;
            for (int i = warp; i < total; i += warps) {
                while (i >= c.rank_psum[s])
                    ++ s;
                const char* token = reinterpret_cast<const char*>(
                    c.buffer + ((s - c.ffn) * static_cast<long long>(c.bucket) +
                        i - c.rank_psum[s - 1]) * token_bytes);
                const int* ids = reinterpret_cast<const int*>(token + meta_at);
                const int e = lane < topk ? ids[lane] - experts * c.me : -1;
                const bool mine = e >= 0 and e < experts;
                int row = -1;
                if (mine) {
                    row = atomicAdd(c.psum + e, 1);
                    c.weights[row] = reinterpret_cast<const float*>(ids + topk)[lane];
                }
                if (lane < topk)
                    c.meta[i * meta_words + 2 + lane] = row;
                const unsigned mask = __ballot_sync(0xffffffff, mine);
                if (lane == 0) {
                    c.meta[i * meta_words] = ids[2 * topk];
                    c.meta[i * meta_words + 1] = s * topk + 31 - __clz(mask);
                }
                for (int k = 0; k < topk; ++ k) {
                    const int r = __shfl_sync(0xffffffff, row, k);
                    if (r < 0)
                        continue;
                    uint4* out = reinterpret_cast<uint4*>(c.x +
                        static_cast<long long>(r) * hidden);
                    for (int j = lane; j < hidden_bytes / 16; j += 32)
                        out[j] = reinterpret_cast<const uint4*>(token)[j];
                }
            }
        } else if (p == REPLY) {
            // As combine: a token's local rows, summed in float, to its source rank.
            const int total = c.rank_psum[c.ranks - 1];
            for (int i = warp; i < total; i += warps) {
                const int* m = c.meta + i * meta_words;
                const int region = c.ranks <= topk ? c.me : m[1] % topk;
                uint4* out = reinterpret_cast<uint4*>(
                    c.buffer + (m[1] / topk - c.me) * c.peer_bytes +
                    (region * static_cast<long long>(c.bucket) + m[0] % c.bucket) *
                        reply_bytes);
                for (int j = lane; j < hidden_bytes / 16; j += 32) {
                    float sum[8] = {};
                    uint4 v;
                    __nv_bfloat16* h = reinterpret_cast<__nv_bfloat16*>(&v);
                    for (int k = 0; k < topk; ++ k) {
                        if (m[2 + k] < 0)
                            continue;
                        v = reinterpret_cast<const uint4*>(c.reply +
                            static_cast<long long>(m[2 + k]) * hidden)[j];
                        for (int n = 0; n < 8; ++ n)
                            sum[n] += __bfloat162float(h[n]);
                    }
                    for (int n = 0; n < 8; ++ n)
                        h[n] = __float2bfloat16(sum[n]);
                    out[j] = v;
                }
            }
        } else {
            // SwiGLU without tensor cores: a CTA takes UNIT_ROWS rows of one expert
            // and `threads` output columns, with the rows staged in shared memory.
            const bool up = p == GATE_UP;
            const int K = up ? hidden : inter, C = up ? 2 * inter : hidden, blocks =
                C / threads;
            const int units = shared[UNITS];
            for (;;) {
                if (tid == 0)
                    unit = atomicAdd(const_cast<int*>(shared) + NEXT, 1);
                __syncthreads();
                if (unit >= units)
                    break;
                const int* item = work + 3 * (unit / blocks);
                const long long row = item[0];
                const int col = unit % blocks * threads + tid;
                for (int j = tid; j < UNIT_ROWS * K; j += threads) {
                    const int r = j / K, k = j % K;
                    float value = 0;
                    if (r < item[2] and up) {
                        value = __bfloat162float(c.x[(row + r) * hidden + k]);
                    } else if (r < item[2]) {
                        const __nv_bfloat16* in = scratch + (row + r) * 2 * inter + k;
                        const float gate = __bfloat162float(in[0]);
                        value = __bfloat162float(__float2bfloat16(gate / (1 +
                            expf(-gate)))) *
                                __bfloat162float(in[inter]);
                    }
                    staged[j] = __float2bfloat16(value);
                }
                __syncthreads();
                float acc[UNIT_ROWS] = {};
                const __nv_bfloat16* weight = (up ? gate_up : down) +
                    static_cast<long long>(item[1]) * K * C + col;
                for (int k = 0; k < K; ++ k) {
                    const float w =
                        __bfloat162float(weight[static_cast<long long>(k) * C]);
                    #pragma unroll
                    for (int r = 0; r < UNIT_ROWS; ++ r)
                        acc[r] += w * __bfloat162float(staged[r * K + k]);
                }
                for (int r = 0; r < item[2]; ++ r) {
                    const __nv_bfloat16 out = __float2bfloat16(acc[r]);
                    if (up)
                        scratch[(row + r) * 2 * inter + col] = out;
                    else
                        c.reply[(row + r) * hidden + col] =
                            __float2bfloat16(__bfloat162float(out) * c.weights[row +
                                r]);
                }
                __syncthreads();
            }
        }
        __syncthreads();
        if (tid == 0) {
            __threadfence_system();
            atomicAdd(const_cast<int*>(shared) + FINISHED, 1);
        }
    }
}

// Second stream of the persistent kernel: WHILE(gate -> SWITCH(client experts)).
// A new iteration means the previous client's reply rows are complete.
extern "C" __global__ void afdep_ffn_gate(
        volatile int* shared, const int num_clients,
        const cudaGraphConditionalHandle loop,
            const cudaGraphConditionalHandle branch) {
    if (threadIdx.x != 0)
        return;
    const int issued = shared[ISSUED];
    shared[COMPUTED] = issued;
    __threadfence();
    while (shared[RECEIVED] == issued and not shared[HALT]) {}
    const bool more = shared[RECEIVED] != issued;
    if (more)
        shared[ISSUED] = issued + 1;
    cudaGraphSetConditional(branch, more ? shared[RING + issued % RING_SIZE] :
        num_clients);
    cudaGraphSetConditional(loop, more ? 1 : 0);
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


def inbox(buffer: Buffer) -> SimpleNamespace:
    """Static receive tensors of one client for the resident kernel."""
    c = buffer.config
    bucket = c.capacity_tokens_per_rank
    rows = c.receive_rows(bucket)

    def zeros(*shape, dtype=torch.int32):
        return torch.zeros(*shape, dtype=dtype, device="cuda")

    psum = zeros(c.experts_per_rank + 1)
    return SimpleNamespace(
        recv_x=zeros(rows, c.hidden_size, dtype=torch.bfloat16),
        recv_scale=None,
        recv_topk_weights=zeros(rows, dtype=torch.float32),
        recv_count_psum=psum[:-1],
        psum=psum,
        meta=zeros(c.attention_ranks * bucket * (2 + c.top_k)),
        ranks=zeros(c.world_size),
        rows=rows,
    )


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

    def _launch_scan(
        self, stream: int, args, function=None, grid=1, threads=SCAN_THREADS, shared=0
    ) -> None:
        _ok(
            driver.cuLaunchKernel(
                function or self._function,
                grid,
                1,
                1,
                threads,
                1,
                1,
                shared,
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

    def _capture_scan(self, graph, dependencies, args, function=None):
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
        self._launch_scan(capture.cuda_stream, args, function)
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

    def build_resident(
        self,
        buffers: list[Buffer],
        boxes: list[SimpleNamespace],
        replies: list[torch.Tensor],
        *,
        gate_up: torch.Tensor | None = None,
        down: torch.Tensor | None = None,
        depth: int = 2,
        ctas: int = 16,
    ) -> None:
        """One resident launch per run over ``boxes[k] = inbox(buffers[k])``.

        Persistent kernel: ``graphs[k]`` computes ``replies[k]`` (weighted expert
        rows) from ``boxes[k]`` and runs on ``scan_stream``; up to ``depth``
        clients are received ahead of their reply. Megakernel: with ``gate_up``
        and ``down`` the kernel computes ``replies[k]`` itself, on every CTA, so
        ``ctas`` must all be resident at once.
        """
        c = buffers[0].config
        shape = (c.hidden_size, c.top_k, c.experts_per_rank, c.expert_alignment)
        mega = gate_up is not None
        inter = down.shape[1] if mega else 0
        # Ceiling: the vendored single-FFN-rank kernels use another wire format.
        if c.ffn_ranks < 2 or not 0 < depth <= RING_SIZE:
            raise ValueError("resident mode needs several FFN ranks and a ring depth")
        if ctas * RESIDENT_THREADS < c.experts_per_rank + c.attention_ranks:
            raise ValueError("resident grid is smaller than the count words")
        if mega and (2 * inter % RESIDENT_THREADS or c.hidden_size % RESIDENT_THREADS):
            raise ValueError("megakernel columns must divide into CTAs")
        extra = []
        header = torch.zeros(WINDOW_WORDS, dtype=torch.int32)
        for buffer, box, reply in zip(buffers, boxes, replies, strict=True):
            k = buffer.config
            if shape != (
                k.hidden_size,
                k.top_k,
                k.experts_per_rank,
                k.expert_alignment,
            ):
                raise ValueError("resident clients must share the expert layout")
            workspace, data, window = buffer.transport._runtime.get_afd_service_words()
            _ok(driver.cuMemcpyDtoH(header.data_ptr(), window, header.numel() * 4))
            if header[WINDOW_LSA_RANK] != buffer.rank:
                raise ValueError("unexpected NCCL window layout")
            extra.append(
                [
                    workspace,
                    data,
                    (int(header[WINDOW_STRIDE]) & 0xFFFFFFFF) << 32,
                    buffer.role_rank,
                    k.capacity_tokens_per_rank,
                    box.recv_x.data_ptr(),
                    box.recv_topk_weights.data_ptr(),
                    box.psum.data_ptr(),
                    box.meta.data_ptr(),
                    reply.data_ptr(),
                    box.ranks.data_ptr(),
                    box.rows,
                ]
            )
        assert len(extra[0]) == EXTRA_WORDS
        rows = max(box.rows for box in boxes)
        self._keep = (boxes, replies, gate_up, down)
        self.extra = torch.tensor(extra, dtype=torch.int64, device="cuda")
        self.shared = torch.zeros(
            SHARED_RING + RING_SIZE, dtype=torch.int32, device="cuda"
        )
        self.scratch = torch.empty(
            rows if mega else 0, 2 * inter, dtype=torch.bfloat16, device="cuda"
        )
        self.work = torch.zeros(
            3 * (rows // UNIT_ROWS + c.experts_per_rank),
            dtype=torch.int32,
            device="cuda",
        )
        self._resident = _ok(
            driver.cuModuleGetFunction(self._module, b"afdep_ffn_resident")
        )
        staged = UNIT_ROWS * max(c.hidden_size, inter) * 2 if mega else 0
        attribute = driver.CUfunction_attribute
        _ok(
            driver.cuFuncSetAttribute(
                self._resident,
                attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                staged,
            )
        )
        p, i, ll = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong
        values = (
            self.clients.data_ptr(),
            self.extra.data_ptr(),
            len(self.graphs),
            self.state.data_ptr(),
            self.quota.data_ptr(),
            self.control.data_ptr(),
            self.log.data_ptr(),
            LOG_CAPACITY,
            self.timeout_ns,
            int(self.rotate),
            self.shared.data_ptr(),
            depth,
            *shape,
            gate_up.data_ptr() if mega else 0,
            down.data_ptr() if mega else 0,
            inter,
            self.scratch.data_ptr(),
            self.work.data_ptr(),
        )
        types = (p, p, i, p, p, p, p, i, ll, i, p, i, i, i, i, i, p, p, i, p, p)
        self._resident_launch = ((values, types), ctas, staged)
        if mega:
            return
        graph = _ok(runtime.cudaGraphCreate(0))
        loop = _ok(
            runtime.cudaGraphConditionalHandleCreate(
                graph,
                1,
                runtime.cudaGraphConditionalHandleFlags.cudaGraphCondAssignDefault,
            )
        )
        body = self._conditional(graph, [], loop, "cudaGraphCondTypeWhile", 1)[0]
        branch = _ok(
            runtime.cudaGraphConditionalHandleCreate(body, len(self.graphs), 0)
        )
        ull = ctypes.c_ulonglong
        gate = self._capture_scan(
            body,
            [],
            (
                (self.shared.data_ptr(), len(self.graphs), int(loop), int(branch)),
                (p, i, ull, ull),
            ),
            _ok(driver.cuModuleGetFunction(self._module, b"afdep_ffn_gate")),
        )
        bodies = self._conditional(
            body, [gate], branch, "cudaGraphCondTypeSwitch", len(self.graphs)
        )
        for child, client in zip(bodies, self.graphs, strict=True):
            _ok(
                runtime.cudaGraphAddChildGraphNode(
                    child, None, 0, runtime.cudaGraph_t(client.raw_cuda_graph())
                )
            )
        self._graph = graph
        self._exec = _ok(runtime.cudaGraphInstantiate(graph, 0))

    def serve_resident(self) -> int:
        """Run the resident kernel (and the experts graph loop) until no choice."""
        args, ctas, staged = self._resident_launch
        self.shared.zero_()
        self.shared[SHARED_FINISHED] = ctas
        torch.cuda.synchronize()
        self._launch_scan(
            self.stream.cuda_stream,
            args,
            self._resident,
            ctas,
            RESIDENT_THREADS,
            staged,
        )
        if self._exec is not None:
            _ok(runtime.cudaGraphLaunch(self._exec, self.scan_stream.cuda_stream))
            self.scan_stream.synchronize()
        self.stream.synchronize()
        return int(self.control[CONTROL_CHOICE])

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
