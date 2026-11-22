# Token-major FFN receive

Status: implemented on branch `support-rank-major`; validated in one NVLink domain only (see
[Validation](#validation)).

## Summary

afdEP-comm receives Attention tokens on the FFN in one layout: **expert-major** rows, one
row per (token, local expert), grouped by expert. This change adds a second,
`Config(receive_layout="token")`: **token-major** rows, one row per received token, with that
token's local expert IDs and routing weights. The FFN replies with one row per token, already
weighted and summed over the token's local experts.

Nothing changes on the wire or on the Attention side. The source dispatch already sends one
copy of a token per destination FFN rank; the expert-major layout is produced only by the
FFN's copy epilogue, and DeepEP V2 already implements the token-major variant of that
epilogue and of combine (`do_expand=false`, `use_expanded_layout=false`). afdEP-comm did not
expose it. The default stays `"expert"` and runs exactly the kernels it runs today.

## Terms

| | Expert-major (`"expert"`, today) | Token-major (`"token"`, new) |
|---|---|---|
| FFN receive row | one per (token, local expert) | one per (token, FFN rank) |
| Row order | grouped by local expert, aligned to `expert_alignment` | arrival order (source rank, slot) |
| Routing per row | 1 weight | `top_k` local expert IDs (`-1`: not local) and weights |
| FFN reply row | the expert's output times its weight | the token's weighted sum over its local experts |
| Who reduces a token's local experts | combine (local reduction) | the caller's expert code |
| Typical consumer | grouped GEMM over contiguous expert blocks (DeepGEMM) | fused MoE kernels that take token-major rows plus `topk_ids`/`topk_weights` and route internally |

"Rank-major" in AFD discussions means the same thing as token-major here: the FFN rank sees
each token once.

## Data path today

1. **Source dispatch** (`afdep_source_dispatch.cuh`). One warp per token, one lane per top-k
   choice. Each lane maps its expert to a rank (`expert / experts_per_rank`). `ptx::deduplicate`
   lets only the first lane holding a rank reserve a slot and store the token (lines 177-179
   count, 478-479 slot, 511-532 NVLink store or RDMA put). The stored row carries the hidden
   state and all `top_k` IDs and weights. **One wire copy per (token, FFN rank).**
2. **FFN copy epilogue** (`dispatch_copy_epilogue.cuh`, `kDoExpand=true`). Each received token
   is written once per local expert it selected, at an `atomicAdd` slot in that expert's block
   (line 126). Per-row weights are 1-D.
3. **Experts** (caller). One output row per expert row, multiplied by its weight.
4. **FFN combine** (`combine.cuh`, `use_expanded_layout=true`, `allow_multiple_reduction=true`).
   The rows of one token on this rank are summed (local reduction, lines 136-160) and sent as
   one row per (token, FFN rank).
5. **Source collect** (`combine_reduce_epilogue.cuh`). Sums the at most `min(F, top_k)` partial
   rows into `[T, H]`.

Token-major changes only steps 2-4 on the FFN: step 2 writes row `i` for received token `i`
(line 124) with its `top_k` local IDs and weights (lines 117-118, 187-189); step 4 sends row `i`
unchanged (`no_local_reduce`, line 137); the caller's step 3 does the local reduction.

## What blocked it

| Layer | Where | What |
|---|---|---|
| API | `Config`, `Buffer.worker_receive`/`worker_reply` | no layout field or argument |
| Wrapper | `ElasticBuffer.dispatch`/`combine` (`afdep/deepep.py`) | `do_expand=True` and `use_expanded_layout=True` passed as positional constants; native `recv_topk_idx` discarded |
| Native host | `buffer.hpp:847-850` | `EP_HOST_ASSERT(not cached_mode and do_expand)` in AFD role mode |
| afdEP kernels | `dispatch.hpp:858` (count-first streaming epilogue), `afdep_ffn_*fp8_hybrid_combine.cuh`, `combine.hpp:201`, `buffer.hpp:1556` (FP8 combine replies) | expanded layout only |

## Goals and non-goals

Goals:

- `Config(receive_layout="token")` selects token-major receive and replies for every rank.
- Same wire protocol, same source API (`source_dispatch`, `source_collect`), same lanes, graph
  and stream contracts.
- `"expert"` (default) is unchanged: same kernels, same template arguments, same cubins.
- Padding is safe for kernels that read every row: rows past the live count and choices of
  experts held elsewhere carry ID `-1` and weight `0`.

Non-goals (this change):

- Performance claims. The token layout saves FFN receive memory and the epilogue's expansion,
  but moves the local reduction into the caller's expert code; whether that wins depends on the
  expert kernel and is a serving question.
- A token-major count-first streaming epilogue or token-major FP8 combine replies.
- Multiple NVLink domains. The token layout is rejected there until validated (below).

## API

| Name | Change |
|---|---|
| `Config.receive_layout: Literal["expert", "token"] = "expert"` | New. Part of the `Config` every rank gathers and compares, so ranks cannot disagree. |
| `Config.receive_rows(bucket)` | `"token"`: `attention_ranks * bucket`. `"expert"`: unchanged, `attention_ranks * bucket * min(top_k, experts_per_rank)` plus alignment. |
| `Buffer.source_dispatch`, `Buffer.source_collect` | Unchanged signature and behavior. |
| `Buffer.worker_receive(*, bucket, read_host_counts)` | Unchanged signature. In token layout returns token-major rows (below). |
| `Buffer.worker_reply(rows, receipt)` | Unchanged signature. In token layout `rows` is `[receive rows, H]` BF16, each row the token's weighted sum over its local experts. Padding rows are ignored. |
| `Buffer.fp8_combine` | Reports `False` in token layout. |
| `ReceiveResult.recv_topk_idx` | New, `None` in expert layout. Token layout: `[N, top_k]` local expert IDs in `[0, experts_per_rank)`, or `-1`. |
| `ReceiveResult.recv_topk_weights` | Expert layout: `[N]` (unchanged). Token layout: `[N, top_k]`, `0` where the ID is `-1`. |
| `ReceiveResult.recv_count_psum` | Expert layout: unchanged (expert row ends). Token layout: inclusive prefix sums of per-expert counts, each rounded up to `expert_alignment`; with `expert_alignment=1` they are exact. |
| `ReceiveResult.recv_token_count` | New property: device `[1]` int32 count of live token rows (rows `[0, count)`). Read-only view of the receipt. |
| `ReceiveResult.expanded` / `DispatchHandle.expanded` | New: the layout this exchange used. Combine reads it from the receipt, so a reply cannot use the other layout. |
| `ElasticBuffer.dispatch(..., expand=True)` | New keyword for the low-level facade; `expand=False` requires the AFD transport (`ffn_ranks`). |

Receive tensors in token layout, `N` = live tokens with host counts, otherwise
`receive_rows(bucket)`:

- `recv_x` `[N, H]` (`float8_e4m3fn` with `recv_scale` `[N, H/128]` for FP8 payloads, same scale
  layout as today), rows `[recv_token_count, N)` unspecified.
- `recv_topk_idx` `[N, top_k]`, rows past the live count all `-1`.
- `recv_topk_weights` `[N, top_k]` FP32, `0` wherever the ID is `-1`.
- `recv_counts_host` (with `read_host_counts=True`): aligned per-expert counts, as today.

## Native changes

All in `csrc/elastic/buffer.hpp` (host code; recorded in `LOCAL_CHANGES.json`). No kernel source
changes: the token-major epilogue and combine branches are DeepEP V2's own, and the generated
afdEP role kernels inherit them.

1. **Dispatch assertion.** In AFD role mode allow `do_expand=false`; keep rejecting cached
   handles. Also assert token layout only in one NVLink domain (direct NVLink), matching the
   Python check.
2. **Padding fix-up.** After the copy epilogue, on the communication stream: if the receive was
   sized without host counts, set `recv_topk_idx` rows `>= recv_token_count` to `-1`; then set
   `recv_topk_weights` to `0` wherever the ID is `-1` (the epilogue copies all `top_k` weights,
   including those of experts held by other ranks). These are small torch ops under a
   `CUDAStreamGuard` on the communication stream, after the epilogue, so the epilogue keeps its
   PDL dependency on the dispatch kernel, CUDA graph capture includes them (the communication
   stream already joins the capture), and the caller's stream sees them through the existing
   stream event.
3. **Count-first.** The streaming epilogue (`afdep_ffn_stream_epilogue.cuh`) only expands rows
   into expert slots. In token layout the FFN receives payload-first, exactly as
   `AFDEP_COUNT_FIRST_ARM=source` already does: sources still run count-first, the count low
   halves decode unchanged and the flags go unread (`afdep_suites.hpp`). The wire is unchanged,
   so both roles keep agreeing per exchange.
4. **Combine assertion.** Token-major replies only with the direct-NVLink AFD combine.

FP8 combine replies need no change in scope: they exist only in the directional hybrid
(`buffer.hpp:148-156`), which needs more than one NVLink domain. Enabling the token layout across
domains must first gate them on the layout, on both roles (the reply row format is on the wire).

## Topology scope

Validated: one NVLink domain, global FFN prefix, with the role kernels, role-sized buffers,
directed exit, the single-FFN collect view, and count-first in both arms.

Rejected at construction with `receive_layout="token"`: more than one physical NVLink domain
(directional, union and mixed-role hybrid; per-domain FFN prefix; direct RDMA). The generated
hybrid combine kernels carry the non-expanded branches they inherit from DeepEP V2, but no afdEP
path has run them. Lifting the check needs: FP8 reply gating (above), and the GPU suite on two and
three NVLink domains for each hybrid mode.

## Memory

Worst-case FFN receive rows per exchange (no host counts):

| | Rows |
|---|---|
| Expert layout | `S * bucket * min(top_k, E) + (alignment - 1) * E`, rounded to `alignment` |
| Token layout | `S * bucket` |

`S` Attention sources, `E` experts per FFN rank. Example: 2 sources, bucket 512, top-k 8, 64
experts per rank, alignment 128: 16,256 rows versus 1,024. Expert code that expands internally
allocates its own intermediates; this table counts only afdEP-comm's receive tensors.

## Validation

Host: unit tests for the configuration, receive bounds, flag plumbing, receipt layout, the
domain check and FP8 reporting.

GPU (`tests/gpu/deepep.py --receive-layout token`): the existing suite unchanged in structure:
mixed, skewed, spread, inactive, skipped and empty routes; eager with and without host counts;
graphs with changing inputs and delayed peers; queued eager exchanges; bucket changes between
exchanges (eager and captured); two lanes in DBO order. In token layout the FFN check verifies
the live count, the per-expert counts from `recv_topk_idx`, `-1`/`0` padding and the prefix
sums, and the test experts reduce each token's local experts exactly as combine's local
reduction does, so the Attention result is compared bit-exactly against the same reference.
Matrix: 2A2F, 3A:1F (plus the collect view), 1A:3F; BF16 and FP8; hidden 256 and 7168; top-k 2
and 8; count-first off, full arm and source arm; and the expert layout as a regression control on
the same digest.

Records name the native source digest measured. Results are communication-only.

Result on native source digest `edd607bd84627560b3e3c1cefa62ae11e59973223cf53bbdc3354e58a11f075e`
(4 x H100, one NVLink domain, PyTorch 2.13.0+cu130, NCCL 2.30.7): host suite 65 passed; all 13 GPU
cases passed: 9 token-layout (2A2F BF16, FP8, hidden 7168 with top-k 8, two lanes, count-first full
arm with FP8 and source arm; 3A:1F with the collect view, and FP8 hidden 7168 top-k 8 with
count-first; 1A:3F FP8) and 4 expert-layout controls (2A2F BF16, FP8 with two lanes, count-first
full arm; 3A:1F with the collect view). At capacity 8 the 2A2F FFN receive bound is 16 rows in token
layout versus 640 expert rows. ([record](validation-token-layout.json))

## Impact on afd-plugin (DeepEP M2N connector after PR 588)

PR 588 moves the connector to the role API (`Lanes`, `source_dispatch`, `worker_receive`,
`worker_reply`, `source_collect`). With this change:

| Connector piece | Change |
|---|---|
| `DeepEPM2NConfig` | New `receive_layout`, or derived from `expert_backend` (`deepgemm` -> `"expert"`, a token-major MoE backend -> `"token"`). Passed to `afdep.Config`; added to the handshake `_MUST_MATCH` set with a `PROTOCOL_VERSION` bump. |
| Attention side (`send_attn_output`, `recv_ffn_output`, dispatch/combine custom ops) | None. |
| `recv_attn_output(ubatch_idx) -> AFDA2FTransferPayload` | Same signature. Token layout fills `payload.hidden_states` `[N, H]`, `payload.topk_ids`, `payload.topk_weights` (fields that already exist). `DeepEPM2NTransferState` keeps `recv_topk_ids` and the 2-D weights. |
| `run_experts(payload, weights) -> Tensor` | Same signature. Token layout needs an expert backend that takes token-major rows and `topk_ids`/`topk_weights` and returns `[N, H]` weighted and summed per token (for example vLLM's fused MoE kernels or Marlin MoE). DeepGEMM's grouped GEMM keeps the expert layout. |
| `send_ffn_output(ffn_output, context)` | Same signature; `worker_reply` with `[N, H]`. |
| Bucket and memory planning | Token layout receive bound is `sources * bucket` rows. |
| `AFDRoutedExpertsConnector` (`connectors/base.py`) | Same signatures. Its docstring promises one row per (token, local expert); reword it to cover both layouts or add a sibling interface. `AFDConnectorBase` is not touched. |
| Unit-test fake (`tests/unit/fakes/afdep.py`) | Accept `receive_layout`; return `recv_topk_idx` and 2-D weights in token layout. |

## Follow-ups

- Validate and enable the token layout across NVLink domains (hybrid modes, per-domain prefix),
  with FP8 replies gated on the layout.
- A token-major count-first epilogue, if count-first becomes a default.
- Exact (unaligned) per-expert counts in token layout without `expert_alignment=1`.
