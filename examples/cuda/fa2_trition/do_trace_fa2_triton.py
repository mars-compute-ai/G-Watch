import argparse
import json
import os
import sys

import torch

from gwatch.cuda.trace import do_trace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test import attention

# Default kernel regex for Triton FA2 kernels
fa2_triton_kernel_regex = ".*_attn_fwd.*"

# Triton FA2 forward kernel scope name_id_map.
# Since the Triton kernel is JIT-compiled, trace hints would need to be
# inserted via Triton's inline assembly or custom scope markers if supported.
# These IDs correspond to the logical phases of the flash attention algorithm.
name_id_map = {
    100: "load_q",
    110: "load_k",
    120: "gemm_qk",
    130: "softmax_rescale",
    140: "load_v",
    150: "gemm_pv",
}


def _parse_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _load_model_heads(config_path: str):
    if not config_path:
        raise ValueError("config_path is required")
    resolved = config_path
    if not os.path.isabs(resolved):
        candidate = os.path.join(os.path.dirname(__file__), "config", resolved)
        if os.path.exists(resolved):
            resolved = resolved
        else:
            resolved = candidate
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"model config not found: {resolved}")
    with open(resolved, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    text_cfg = cfg.get("text_config", cfg)
    q_heads = int(text_cfg["num_attention_heads"])
    kv_heads = int(text_cfg["num_key_value_heads"])
    return q_heads, kv_heads, resolved


def _resolve_heads(args):
    resolved_config = None
    if args.model_config:
        cfg_q, cfg_kv, resolved_config = _load_model_heads(args.model_config)
    else:
        cfg_q, cfg_kv = None, None

    q_heads = args.q_heads
    if q_heads is None:
        q_heads = args.nheads if args.nheads is not None else cfg_q
    kv_heads = args.kv_heads if args.kv_heads is not None else cfg_kv
    if q_heads is None:
        q_heads = 16
    if kv_heads is None:
        kv_heads = q_heads

    if q_heads <= 0 or kv_heads <= 0:
        raise ValueError(f"Invalid heads: q_heads={q_heads}, kv_heads={kv_heads}")
    if q_heads % kv_heads != 0:
        raise ValueError(f"q_heads must be divisible by kv_heads, got q_heads={q_heads}, kv_heads={kv_heads}")

    return q_heads, kv_heads, resolved_config


def main():
    parser = argparse.ArgumentParser(description="Trace Triton FlashAttention2 kernels with do_trace.")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=8192)
    parser.add_argument("--nheads", type=int, default=None, help="Legacy alias for --q-heads.")
    parser.add_argument("--q-heads", type=int, default=None, help="Number of query heads.")
    parser.add_argument("--kv-heads", type=int, default=None, help="Number of key/value heads.")
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="JSON config path defining `num_attention_heads` and `num_key_value_heads` (relative paths may reference config/).",
    )
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--sm-scale", type=float, default=1.0)
    parser.add_argument("--kernel-regex", type=str, default=fa2_triton_kernel_regex)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--dump-path", type=str, default="./fa2_triton_trace.gwatch")
    args = parser.parse_args()

    device = f"cuda:{args.device_id}"
    dtype = _parse_dtype(args.dtype)
    q_heads, kv_heads, resolved_config = _resolve_heads(args)
    nheads = q_heads

    # Triton FA2 layout: (batch, heads, seqlen, headdim)
    q = torch.randn(args.batch, nheads, args.seqlen, args.headdim, device=device, dtype=dtype)
    k = torch.randn(args.batch, nheads, args.seqlen, args.headdim, device=device, dtype=dtype)
    v = torch.randn(args.batch, nheads, args.seqlen, args.headdim, device=device, dtype=dtype)

    sm_scale = args.sm_scale

    # Warmup to avoid one-time init kernels (including Triton JIT compilation).
    with torch.no_grad():
        _ = attention(q, k, v, args.causal, sm_scale)
    torch.cuda.synchronize()

    def run_once():
        _ = attention(q, k, v, args.causal, sm_scale)

    results = do_trace(
        fn=run_once,
        warmup=args.warmup,
        rep=args.rep,
        kernel_name_pattern=args.kernel_regex,
        name_id_map=name_id_map,
        dump_path=args.dump_path,
    )

    if len(results) == 0:
        summary = {
            "trace_results": 0,
            "note": "no trace results returned (Triton kernel may not contain trace hints)",
            "dump_path": args.dump_path,
        }
        print(json.dumps(summary, indent=2))
        return

    first = results[0]
    summary = {
        "num_runs": len(results),
        "compile_results:": first["compile_results"],
        "profile_results": first["profile_results"],
        "num_trace_records": len(first.get("trace_results", [])),
        "kernel_prototype": first.get("kernel_prototype"),
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "gqa_ratio": q_heads // kv_heads,
        "model_config": resolved_config,
        "dump_path": args.dump_path,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
