#include <cuda_runtime.h>
#include <nccl_device.h>

static __device__ long long timer() {
    long long now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    return now;
}

static __device__ bool ended(const int* state, const int* quota, const int c) {
    return quota[c] > 0 and state[3 + c] >= quota[c];
}

// Ready: every sender released the next unissued round (as afdep_wait_senders),
// so a pick may run ahead of the exchanges.
static __device__ bool ready(const long long* clients, const int* state, const int c) {
    const long long* w = clients + c * 6;
    const volatile long long* arrival =
        reinterpret_cast<const volatile long long*>(w[0]);
    const long long target = (w[5] + state[3 + c] + 1) * w[4];
    for (long long s = w[2]; s < w[2] + w[3]; ++ s)
        if (arrival[s] < target)
            return false;
    return true;
}

// The first ready client after the one served last; -2: every quota is met,
// -4: none is ready. layers > 0 (Take First Layer Group): the pick grows over
// the next clients while they are ready at the first one's layer (exchanges
// served mod layers), and is first + (count - 1) * num_clients.
static __device__ int next_client(const long long* clients, const int num_clients,
                                  const int* state, const int* quota,
                                  const int rotate, const int layers) {
    bool pending = false;
    for (int i = 1; i <= num_clients; ++ i) {
        const int c = ((rotate ? state[0] : num_clients - 1) + i) % num_clients;
        if (ended(state, quota, c))
            continue;
        pending = true;
        if (not ready(clients, state, c))
            continue;
        int count = 1;
        while (layers > 0 and c + count < num_clients and
               not ended(state, quota, c + count) and
               state[3 + c + count] % layers == state[3 + c] % layers and
               ready(clients, state, c + count))
            ++ count;
        return c + (count - 1) * num_clients;
    }
    return pending ? -4 : -2;
}

// Pick i of the leader (FFN rank 0), read from its order ring; -4: not published.
static __device__ int leader_choice(const volatile long long* order,
                                    const int order_words, const int num_clients,
                                    const int* state, const int* quota) {
    bool pending = false;
    for (int c = 0; c < num_clients; ++ c)
        pending |= not ended(state, quota, c);
    if (not pending)
        return -2;
    const long long entry = order[state[2] % order_words];
    return entry >> 32 == state[2] + 1 ? static_cast<int>(entry) : -4;
}

static __device__ void record(int* state, int* log, const int log_capacity,
    const int choice, volatile long long* order, const int order_words,
    const int num_clients) {
    if (order)  // entry i: (i + 1) << 32 | choice, so a reused word is never stale
        order[state[2] % order_words] = (state[2] + 1LL) << 32 | choice;
    const int first = choice % num_clients, count = choice / num_clients + 1;
    state[0] = first + count - 1;
    for (int c = first; c < first + count; ++ c)
        state[3 + c] += 1;
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
        const cudaGraphConditionalHandle branch,
        volatile long long* order, const int order_words, const ncclWindow_t window,
        const int rank, const int layers) {
    // order: this rank's ring words; a follower reads the same words of FFN rank 0.
    const bool follow = order and rank > 0;
    if (follow)
        order = reinterpret_cast<volatile long long*>(
            reinterpret_cast<char*>(const_cast<long long*>(order)) +
            (static_cast<char*>(ncclGetLsaPointer(window, 0, 0)) -
             static_cast<char*>(ncclGetLsaPointer(window, 0, rank))));
    if (threadIdx.x != 0)
        return;
    // Device mode: state[1] counts this launch's rounds, -1 once a scan ended it.
    const bool device = rounds_per_launch > 0, halted = device and state[1] < 0;
    const long long start = timer();
    int choice = -3;
    while (not halted) {
        if (control[0]) { choice = -1; break; }
        choice = follow
            ? leader_choice(order, order_words, num_clients, state, quota)
            : next_client(clients, num_clients, state, quota, rotate, layers);
        if (choice != -4)
            break;
        if (timer() - start > timeout_ns) { choice = -3; break; }
    }
    if (not halted and choice >= 0)
        record(state, log, log_capacity, choice, follow ? nullptr : order, order_words,
               num_clients);
    if (not halted)
        control[1] = choice;  // the host reads it after the stream completes
    if (device) {
        if (not halted)
            state[1] = choice >= 0 ? state[1] + 1 : -1;
        // No branch: a value past every pick (the switch holds at most K * K).
        cudaGraphSetConditional(branch, choice >= 0 ? choice : num_clients * num_clients);
        if (last_in_body) {
            const bool more = state[1] >= 0 and state[1] < rounds_per_launch;
            if (not more)
                state[1] = 0;
            cudaGraphSetConditional(loop, more ? 1 : 0);
        }
    }
}
