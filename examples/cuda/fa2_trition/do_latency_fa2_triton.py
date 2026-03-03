#!/usr/bin/env python3
"""Triton FA2 latency benchmark for batch=1 inference scenarios."""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test import attention


def _load_model_heads(config_path):
    resolved = config_path
    candidate = os.path.join(os.path.dirname(__file__), "config", resolved)
    if not os.path.exists(resolved):
        resolved = candidate
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"model config not found: {resolved}")
    with open(resolved, "r") as f:
        cfg = json.load(f)
    text_cfg = cfg.get("text_config", cfg)
    return int(text_cfg["num_attention_heads"]), int(text_cfg["num_key_value_heads"]), resolved


def main():
    parser = argparse.ArgumentParser(description="Measure Triton FA2 latency (batch=1 focus).")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=8192)
    parser.add_argument("--nheads", type=int, default=None)
    parser.add_argument("--kv-heads", type=int, default=None)
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--causal", action="store_true", default=False)
    parser.add_argument("--sm-scale", type=float, default=1.0)
    parser.add_argument("--rep", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    torch.cuda.set_device(args.device_id)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    cfg_q, cfg_kv, resolved_config = None, None, None
    if args.model_config:
        cfg_q, cfg_kv, resolved_config = _load_model_heads(args.model_config)

    q_heads = args.nheads if args.nheads is not None else (cfg_q if cfg_q else 16)
    kv_heads = args.kv_heads if args.kv_heads is not None else (cfg_kv if cfg_kv else q_heads)
    nheads = q_heads

    # Triton FA2 layout: (batch, heads, seqlen, headdim)
    q = torch.randn(args.batch, nheads, args.seqlen, args.headdim, device="cuda", dtype=dtype)
    k = torch.randn(args.batch, nheads, args.seqlen, args.headdim, device="cuda", dtype=dtype)
    v = torch.randn(args.batch, nheads, args.seqlen, args.headdim, device="cuda", dtype=dtype)

    sm_scale = args.sm_scale

    # Warmup
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = attention(q, k, v, args.causal, sm_scale)
    torch.cuda.synchronize()

    # Measure individual kernel latencies
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.rep)]

    with torch.no_grad():
        for i in range(args.rep):
            start_events[i].record()
            _ = attention(q, k, v, args.causal, sm_scale)
            end_events[i].record()
    torch.cuda.synchronize()

    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times_ms.sort()
    trim = max(1, args.rep // 10)
    trimmed = times_ms[trim:-trim]
    avg_ms = sum(trimmed) / len(trimmed)
    min_ms = times_ms[0]
    p50_ms = times_ms[args.rep // 2]
    p99_ms = times_ms[int(args.rep * 0.99)]

    # Compute FLOPS: 4 * B * S^2 * H_q * D (QK + PV, each 2*B*S^2*H_q*D)
    flops = 4 * args.batch * args.seqlen * args.seqlen * q_heads * args.headdim
    tflops = flops / (avg_ms * 1e-3) / 1e12

    # Memory: Q(B*H*S*D) + K(B*H*S*D) + V(B*H*S*D) + O(B*H*S*D), 2B each for fp16
    elem_bytes = 2
    tensor_bytes = args.batch * nheads * args.seqlen * args.headdim * elem_bytes
    total_bytes = 4 * tensor_bytes  # Q + K + V + O
    bw_gbps = total_bytes / (avg_ms * 1e-3) / 1e9

    result = {
        "batch": args.batch,
        "seqlen": args.seqlen,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "gqa_ratio": q_heads // kv_heads,
        "headdim": args.headdim,
        "dtype": str(dtype).split(".")[-1],
        "causal": args.causal,
        "avg_ms": round(avg_ms, 4),
        "min_ms": round(min_ms, 4),
        "p50_ms": round(p50_ms, 4),
        "p99_ms": round(p99_ms, 4),
        "tflops": round(tflops, 2),
        "mem_bytes_GB": round(total_bytes / 1e9, 4),
        "bw_GBps": round(bw_gbps, 2),
        "a100_bw_util": f"{bw_gbps / 2039 * 100:.1f}%",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
