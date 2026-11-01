// Copyright (c) 2026 afdEP contributors. SPDX-License-Identifier: MIT
// afdEP per-domain FFN prefix, Attention rails (afdep_roles.hpp): every Attention GPU is a source and
// its rail's forwarder, so it receives dispatch finish tails and combine signals from the same peers.
// DeepEP keeps both in one (channel, peer) word, which only the rail halves of the entry barriers kept
// apart: a peer's dispatch-(N+1) tail could land in a word this GPU still waits on for that peer's
// combine-N signal. Combine signals therefore take their own words. Only the generated per-domain
// combine (tests/role_kernels.py) uses them.
#pragma once

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/layout.cuh>

namespace deep_ep::elastic {

// Words, as int64 indices from the signaled-tail base (kNumMaxRanks * kNumMaxChannels words, zeroed at
// construction). Hybrid tails use [0, C * domains), at most kNumMaxChannels channels x 32 domains; the
// combine signals take the same shape right after them. The sparse directional hybrid's words
// (afdep_sparse_hybrid.cuh) start 4,096 words before the region end, and the two protocols never run in
// one layout.
constexpr int kAfdDomainMaxDomains = 32;
constexpr int64_t kAfdDomainCombineWords =
    static_cast<int64_t>(deep_ep::kNumMaxChannels) * kAfdDomainMaxDomains;
EP_STATIC_ASSERT(2 * kAfdDomainCombineWords + 4096 <=
                     static_cast<int64_t>(layout::WorkspaceLayout::kNumMaxRanks) * deep_ep::kNumMaxChannels,
                 "Per-domain combine words exceed the signaled-tail region");

// The combine signal of (channel, scale-out peer), laid out as the hybrid tail of the same pair.
__device__ __forceinline__ int64_t* afdep_domain_combine_signal_ptr(const layout::WorkspaceLayout& workspace,
                                                                     const int channel_idx, const int scaleout_rank_idx) {
    return workspace.get_scaleout_channel_signaled_tail_ptr(channel_idx, scaleout_rank_idx) + kAfdDomainCombineWords;
}

}  // namespace deep_ep::elastic
