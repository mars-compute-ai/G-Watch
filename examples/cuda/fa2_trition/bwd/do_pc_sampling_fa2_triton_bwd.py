import argparse
import json
import os
import sys

import torch

import gwatch.cuda.profile as gw_profile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from test import attention


def _parse_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _parse_list(value: str):
    return [v.strip() for v in value.split(",") if v.strip()]


def _load_model_heads(config_path: str):
    if not config_path:
        raise ValueError("config_path is required")
    resolved = config_path
    if not os.path.isabs(resolved):
        candidate = os.path.join(os.path.dirname(__file__), "..", "config", resolved)
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
    parser = argparse.ArgumentParser(description="PC sampling for Triton FA2 backward kernel.")
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
        help="JSON config path.",
    )
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--sm-scale", type=float, default=1.0)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument(
        "--kernel-regex",
        type=str,
        default=".*_attn_bwd.*",
        help="Comma-separated list of kernel regex filters.",
    )
    parser.add_argument(
        "--dump-path",
        type=str,
        default="./fa2_triton_bwd_pc_sampling.gwatch",
        help="Output .gwatch path.",
    )
    args = parser.parse_args()

    device = f"cuda:{args.device_id}"
    dtype = _parse_dtype(args.dtype)
    q_heads, kv_heads, resolved_config = _resolve_heads(args)
    nheads = q_heads

    q = torch.randn(args.batch, nheads, args.seqlen, args.headdim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(args.batch, nheads, args.seqlen, args.headdim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(args.batch, nheads, args.seqlen, args.headdim, device=device, dtype=dtype, requires_grad=True)

    sm_scale = args.sm_scale

    # Warmup (forward + backward)
    o = attention(q, k, v, args.causal, sm_scale)
    do = torch.randn_like(o)
    o.backward(do, retain_graph=False)
    q.grad = None
    k.grad = None
    v.grad = None
    torch.cuda.synchronize()

    def run_once():
        o = attention(q, k, v, args.causal, sm_scale)
        g = torch.randn_like(o)
        o.backward(g, retain_graph=False)
        q.grad = None
        k.grad = None
        v.grad = None
        torch.cuda.synchronize()

    kernel_regex = _parse_list(args.kernel_regex)

    result = gw_profile.do_pc_sampling(
        fn=run_once,
        list_kernel_names=kernel_regex,
        rep=args.rep,
        dump_path=args.dump_path,
    )

    pc_sampling = result.get("pc_sampling", {})
    summary = {
        "num_kernels_profiled": pc_sampling.get("num_kernels_profiled"),
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "gqa_ratio": q_heads // kv_heads,
        "model_config": resolved_config,
        "dump_path": args.dump_path,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
