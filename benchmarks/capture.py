"""Capture one decoder layer of a real checkpoint on ShareGPT V3 for the replay.

Writes the dequantized layer weights and the layer's hidden-state inputs and
outputs for teacher-forced ShareGPT conversations (first prompt and first reply).
"""

import argparse
import glob
import json
import os
import random

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen3_5MoeForConditionalGeneration

DATASET = (
    "anon8231489123/ShareGPT_Vicuna_unfiltered",
    "ShareGPT_V3_unfiltered_cleaned_split.json",
)
FP8_BLOCK = 128
MIN_TOKENS, MAX_PROMPT, MAX_TOTAL = 4, 1024, 2048  # the vLLM ShareGPT benchmark filter


class Stop(Exception):
    pass


def layer_weights(model_dir, layer):
    prefix = f"model.language_model.layers.{layer}."
    raw = {}
    for shard in sorted(glob.glob(f"{model_dir}/*.safetensors")):
        with safe_open(shard, "pt") as f:
            raw.update(
                {
                    k[len(prefix) :]: f.get_tensor(k)
                    for k in f.keys()
                    if k.startswith(prefix)
                }
            )

    def weight(name):
        w, scale = raw[name + ".weight"].float(), raw.get(name + ".weight_scale_inv")
        if scale is not None:
            scale = (
                scale.float()
                .repeat_interleave(FP8_BLOCK, 0)
                .repeat_interleave(FP8_BLOCK, 1)
            )
            w = w * scale[: w.shape[0], : w.shape[1]]
        return w.bfloat16()

    names = {k[: -len(".weight")] for k in raw if k.endswith(".weight")}
    out = {n: weight(n) for n in names if ".experts." not in n}
    experts = 1 + max(int(n.split(".")[2]) for n in names if ".experts." in n)
    part = lambda e, p: weight(f"mlp.experts.{e}.{p}_proj")  # noqa: E731
    # Transposed for right-multiplication:
    # (experts, hidden, 2 * inter) and (experts, inter, hidden).
    out["experts.gate_up"] = torch.stack(
        [torch.cat([part(e, "gate"), part(e, "up")]).T for e in range(experts)]
    )
    out["experts.down"] = torch.stack([part(e, "down").T for e in range(experts)])
    return out


def conversations(tokenizer, seed):
    data = json.load(open(hf_hub_download(*DATASET, repo_type="dataset")))
    data = [d["conversations"] for d in data if len(d["conversations"]) >= 2]
    random.Random(seed).shuffle(data)
    for conv in data:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": conv[0]["value"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt = tokenizer(text, add_special_tokens=False).input_ids
        reply = tokenizer(conv[1]["value"], add_special_tokens=False).input_ids
        if (
            MIN_TOKENS <= len(prompt) <= MAX_PROMPT
            and MIN_TOKENS <= len(reply)
            and len(prompt) + len(reply) <= MAX_TOTAL
        ):
            yield prompt, reply


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/model")
    p.add_argument("--out", default="/out/afdep-capture")
    p.add_argument("--layer", type=int, default=31)
    p.add_argument("--count", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if not os.path.exists(f"{args.out}/weights.pt"):
        weights = layer_weights(args.model, args.layer)
        print({k: tuple(v.shape) for k, v in weights.items()}, flush=True)
        torch.save(weights, f"{args.out}/weights.pt")
        del weights

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        args.model, dtype="auto", device_map="auto"
    )
    convs = [
        c
        for c, _ in zip(
            conversations(tokenizer, args.seed), range(args.count), strict=False
        )
    ]

    ids = torch.tensor([convs[0][0]], device=model.device)
    text = tokenizer.decode(
        model.generate(input_ids=ids, max_new_tokens=24, do_sample=False)[
            0, ids.shape[1] :
        ]
    )
    print("generation check:", repr(text), flush=True)

    grabbed = {}

    def grab(module, inputs, kwargs, output):
        grabbed["in"] = inputs[0] if inputs else kwargs["hidden_states"]
        grabbed["out"] = output[0] if isinstance(output, tuple) else output
        raise Stop

    model.model.language_model.layers[args.layer].register_forward_hook(
        grab, with_kwargs=True
    )
    h_in, h_out = [], []
    for i, (prompt, reply) in enumerate(convs):
        try:
            model(input_ids=torch.tensor([prompt + reply], device=model.device))
        except Stop:
            pass
        h_in.append(grabbed["in"][0].cpu())
        h_out.append(grabbed["out"][0].cpu())
        if i % 50 == 0:
            print(i, len(prompt), len(reply), flush=True)

    lengths = torch.tensor([len(a) + len(b) for a, b in convs])
    torch.save(
        {
            "h_in": torch.cat(h_in),
            "h_out": torch.cat(h_out),
            "offsets": torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)]),
            "prompt_len": torch.tensor([len(a) for a, _ in convs]),
            "meta": {
                "model": model.config._name_or_path,
                "layer": args.layer,
                "dataset": DATASET,
                "seed": args.seed,
            },
        },
        f"{args.out}/hidden.pt",
    )
    print("DONE", len(convs), int(lengths.sum()), flush=True)


if __name__ == "__main__":
    main()
