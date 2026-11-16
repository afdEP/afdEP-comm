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
