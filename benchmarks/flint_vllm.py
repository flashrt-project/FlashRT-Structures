#!/usr/bin/env python3
"""Paired Flint/vLLM benchmark inputs; prefill and steady decode are separate.

Input is an external JSON list of {id, tokens} records. Both arms use the same
token IDs. Output includes lengths/digests and timing samples, never prompt text.
See docs/flint_vllm.md for the pinned host and candidate configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time


def summarize_steps(steps, concurrency):
    """A step producing a request's first token still belongs to prefill."""
    prefill = [s[0] for s in steps if not s[2]]
    decode = [s[0] for s in steps if s[2] and s[1] == concurrency]
    return {
        "prefill_s": sum(prefill),
        "decode_step_ms_median": statistics.median(decode) * 1000,
        "decode_step_ms_p90": sorted(decode)[int(len(decode) * 0.9)] * 1000,
        "decode_tok_s": concurrency / statistics.median(decode),
        "n_decode_steps": len(decode),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--record", default="prefill_100k")
    parser.add_argument("--arm", choices=["host", "native"], required=True)
    parser.add_argument("--concurrency", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--in-len", type=int, default=1022)
    parser.add_argument("--max-len", type=int, default=110592)
    parser.add_argument("--nout", type=int, default=128)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument(
        "--allow-ragged",
        action="store_true",
        help="retain short final windows; record actual input lengths",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if min(args.in_len, args.reps) < 1 or args.nout < 2:
        parser.error("input length/reps must be positive and nout at least 2")
    os.environ["FLASHRT_STRUCTURES_NATIVE"] = "1" if args.arm == "native" else "0"
    if args.arm == "host":
        os.environ["FLASHRT_SAGE_PREFILL"] = "0"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    from vllm import LLM, SamplingParams
    from vllm.plugins import load_general_plugins
    from vllm.sampling_params import RequestOutputKind
    import torch

    load_general_plugins()
    records = {p["id"]: p["tokens"] for p in json.loads(Path(args.tokens).read_text())}
    source = records[args.record]
    if not source:
        parser.error("empty source record")
    stride = max(1, len(source) // args.concurrency)
    repeated = source * ((len(source) + args.in_len - 1) // len(source) + 1)
    source_windows = source if args.allow_ragged else repeated
    prompts = [
        source_windows[i * stride : i * stride + args.in_len]
        for i in range(args.concurrency)
    ]
    if len({tuple(p) for p in prompts}) != args.concurrency:
        parser.error("source does not provide distinct windows for this concurrency")
    llm = LLM(
        model=args.model,
        hf_config_path=args.config,
        tokenizer=args.config,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=args.max_len,
        max_num_seqs=args.concurrency,
        max_num_batched_tokens=8192,
        enable_prefix_caching=False,
        kv_cache_dtype="fp8",
        seed=0,
        disable_log_stats=True,
    )
    from flashrt_structures.adapters.vllm_engine import attached

    handles = attached()
    if args.arm == "native" and not handles:
        raise RuntimeError("native load hook did not run")
    engine = llm.llm_engine
    params = SamplingParams(
        temperature=0,
        max_tokens=args.nout,
        ignore_eos=True,
        output_kind=RequestOutputKind.DELTA,
    )
    samples = []
    for rep in range(args.reps + 1):
        seen, steps = {}, []
        torch.cuda.synchronize()
        for i, ids in enumerate(prompts):
            engine.add_request(f"{rep}-{i}", {"prompt_token_ids": ids}, params)
        while engine.has_unfinished_requests():
            pure_decode = len(seen) == args.concurrency and all(seen.values())
            start = time.perf_counter()
            outputs = engine.step()
            elapsed = time.perf_counter() - start
            count = 0
            for output in outputs:
                tokens = list(output.outputs[0].token_ids)
                if tokens:
                    seen.setdefault(output.request_id, []).extend(tokens)
                count += len(tokens)
            steps.append((elapsed, count, pure_decode))
        torch.cuda.synchronize()
        if len(seen) != args.concurrency or any(
            len(ids) != args.nout for ids in seen.values()
        ):
            raise RuntimeError("request count or output length mismatch")
        if rep:
            samples.append(
                dict(
                    summarize_steps(steps, args.concurrency),
                    steps=steps,
                    output_sha256=hashlib.sha256(
                        json.dumps([seen[key] for key in sorted(seen)]).encode()
                    ).hexdigest(),
                )
            )
    ledgers = [handle.summary() for handle in handles]
    if args.arm == "native" and any(row["fallbacks"] for row in ledgers):
        raise RuntimeError("unexpected guarded fallback")
    result = {
        "arm": args.arm,
        "concurrency": args.concurrency,
        "input_lengths": list(map(len, prompts)),
        "input_sha256": [
            hashlib.sha256(json.dumps(p).encode()).hexdigest() for p in prompts
        ],
        "nout": args.nout,
        "warmup": 1,
        "reps": args.reps,
        "samples": samples,
        "ledgers": ledgers,
        "native_regions": [h.records.get("native_regions", {}) for h in handles],
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2))
    print(
        json.dumps(
            {
                "prefill_s": statistics.median(s["prefill_s"] for s in samples),
                "decode_tok_s": args.concurrency
                / (
                    statistics.median(s["decode_step_ms_median"] for s in samples)
                    / 1000
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
