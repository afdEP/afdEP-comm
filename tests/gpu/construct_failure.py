# SPDX-License-Identifier: Apache-2.0
"""Expected failure: one rank fails inside the native DeepEP constructor.

Every rank should raise. Today only the failing rank does; its peers stay inside the
collective constructor. This check exits 0 while that limit holds and exits 1 once
every rank raises: then drop the expected-failure branch and require all ranks.

Example: EP_DISABLE_GIN=1 python -m tests.gpu.construct_failure
"""

from __future__ import annotations

import faulthandler
import os
import re
import subprocess
import sys
from datetime import timedelta

import torch
import torch.distributed as dist

from afdep import Config, Lanes
from afdep._build import load_deepep

RANKS, FFN_RANKS, LANES = 4, 2, 3
FAIL_RANK, FAIL_LANE = 1, LANES - 1  # Earlier lanes are already built.
GROUP_TIMEOUT_S = 30  # The failing rank waits this long for its peers in close().
PEER_WAIT_S = 120
RAISED = "RAISED rank"


def worker() -> None:
    # A rank inside the native constructor cannot run Python; this exits it anyway.
    faulthandler.dump_traceback_later(PEER_WAIT_S, exit=True)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo", timeout=timedelta(seconds=GROUP_TIMEOUT_S))
    rank = dist.get_rank()
    if rank == FAIL_RANK:
        module = load_deepep()
        native, built = module.ElasticBuffer, []

        def failing(*args):
            if len(built) == FAIL_LANE:
                raise RuntimeError("injected failure inside the native constructor")
            built.append(native(*args))
            return built[-1]

        module.ElasticBuffer = failing
    config = Config(
        capacity_tokens_per_rank=8,
        hidden_size=256,
        num_experts=8,
        top_k=2,
        ffn_ranks=FFN_RANKS,
        attention_ranks=RANKS - FFN_RANKS,
        num_sms=8,
    )
    try:
        Lanes(group=dist.group.WORLD, config=config, lanes=LANES)
    except Exception as exc:
        print(f"{RAISED} {rank}: {type(exc).__name__}: {exc}", flush=True)
    os._exit(0)  # No collective teardown: the peers may never arrive.


def main() -> None:
    if "LOCAL_RANK" in os.environ:
        return worker()
    launch = [sys.executable, "-m", "torch.distributed.run", "--standalone"]
    output = subprocess.run(
        [*launch, f"--nproc-per-node={RANKS}", "-m", "tests.gpu.construct_failure"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout
    raised = sorted({int(rank) for rank in re.findall(rf"{RAISED} (\d+)", output)})
    waiting = sorted(set(range(RANKS)) - set(raised))
    if FAIL_RANK not in raised:
        sys.exit(f"the injected failure was not reported:\n{output}")
    if not waiting:
        sys.exit("unexpected pass: every rank raised; drop the expected failure")
    print(f"expected failure: ranks {waiting} did not raise within {PEER_WAIT_S} s")


if __name__ == "__main__":
    main()
