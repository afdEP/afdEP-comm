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

// d += a * b on tensor cores (bf16 pairs in, FP32 out): row `lane / 4` of 8 rows
// by 8 columns, 16 inputs per call. The second 8 rows of the 16x8 tile are zero.
static __device__ void mma(float* d, const unsigned a0, const unsigned a2,
                           const unsigned b0, const unsigned b1) {
    const unsigned none = 0;
    float zero = 0, unused0, unused1;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(unused0), "=f"(unused1)
        : "r"(a0), "r"(none), "r"(a2), "r"(none), "r"(b0), "r"(b1),
          "f"(d[0]), "f"(d[1]), "f"(zero), "f"(zero));
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
    extern __shared__ __align__(16) __nv_bfloat16 staged[];
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
            // SwiGLU on tensor cores: a CTA takes UNIT_ROWS rows of one expert and
            // `threads` output columns (32 per warp), rows staged in shared memory.
            // Weights are [expert][column][input].
            const bool up = p == GATE_UP;
            const int K = up ? hidden : inter, C = up ? 2 * inter : hidden;
            const int blocks = C / threads, units = shared[UNITS];
            const int group = lane >> 2, pair = lane & 3;
            for (;;) {
                if (tid == 0)
                    unit = atomicAdd(const_cast<int*>(shared) + NEXT, 1);
                __syncthreads();
                if (unit >= units)
                    break;
                const int* item = work + 3 * (unit / blocks);
                const long long row = item[0] + group;
                for (int j = tid; j < UNIT_ROWS * K; j += threads) {
                    const int r = j / K, k = j % K;
                    float value = 0;
                    if (r < item[2] and up) {
                        value = __bfloat162float(c.x[(item[0] + r) * hidden + k]);
                    } else if (r < item[2]) {
                        const __nv_bfloat16* in =
                            scratch + (item[0] + r) * 2 * inter + k;
                        const float gate = __bfloat162float(in[0]);
                        const float act = gate / (1 + expf(-gate));
                        value = __bfloat162float(__float2bfloat16(act)) *
                                __bfloat162float(in[inter]);
                    }
                    staged[j] = __float2bfloat16(value);
                }
                __syncthreads();
                const int col = unit % blocks * threads + tid / 32 * 32;
                const unsigned* a =
                    reinterpret_cast<const unsigned*>(staged + group * K) + pair;
                const unsigned* b = reinterpret_cast<const unsigned*>(
                    (up ? gate_up : down) +
                    (static_cast<long long>(item[1]) * C + col + group) * K) + pair;
                float d[4][2] = {};
                for (int k = 0; k < K / 2; k += 8) {
                    #pragma unroll
                    for (int t = 0; t < 4; ++ t)
                        mma(d[t], a[k], a[k + 4], b[t * 4 * K + k],
                            b[t * 4 * K + k + 4]);
                }
                for (int n = 0; n < 8 and group < item[2]; ++ n) {
                    const __nv_bfloat16 out = __float2bfloat16(d[n / 2][n % 2]);
                    const int to = col + n / 2 * 8 + pair * 2 + n % 2;
                    if (up)
                        scratch[row * 2 * inter + to] = out;
                    else
                        c.reply[row * hidden + to] =
                            __float2bfloat16(__bfloat162float(out) * c.weights[row]);
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
