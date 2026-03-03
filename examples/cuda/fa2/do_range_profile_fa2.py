import argparse
import json
import os
import sys

import torch

import gwatch.cuda.profile as gw_profile

try:
    from flash_attn import flash_attn_func as flash_attn_func_v2
except ImportError:
    flash_attn_func_v2 = None

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


def _build_inputs(batch, seqlen, q_heads, kv_heads, headdim, dtype, device, requires_grad):
    q = torch.randn(batch, seqlen, q_heads, headdim, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(batch, seqlen, kv_heads, headdim, device=device, dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(batch, seqlen, kv_heads, headdim, device=device, dtype=dtype, requires_grad=requires_grad)
    return q, k, v


def main():
    parser = argparse.ArgumentParser(description="Range profile FlashAttention2 with do_range_profile.")
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
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--backward", action="store_true", help="Profile backward pass instead of forward.")
    parser.add_argument(
        "--metrics",
        type=str,
        default="sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
        help="Comma-separated list of metric names.",
    )
    parser.add_argument(
        "--kernel-regex",
        type=str,
        default=".*flash.*",
        help="Comma-separated list of kernel regex filters.",
    )
    parser.add_argument(
        "--dump-path",
        type=str,
        default="./fa2_range_profile.gwatch",
        help="Output .gwatch path for merged range profile.",
    )
    args = parser.parse_args()

    if flash_attn_func_v2 is None:
        raise RuntimeError("FlashAttention v2 not found. Install flash-attn package.")

    device = f"cuda:{args.device_id}"
    dtype = _parse_dtype(args.dtype)
    q_heads, kv_heads, resolved_config = _resolve_heads(args)

    q, k, v = _build_inputs(
        batch=args.batch,
        seqlen=args.seqlen,
        q_heads=q_heads,
        kv_heads=kv_heads,
        headdim=args.headdim,
        dtype=dtype,
        device=device,
        requires_grad=args.backward,
    )

    # Warmup to avoid one-time init kernels.
    with torch.no_grad():
        _ = flash_attn_func_v2(q, k, v, causal=args.causal)
    torch.cuda.synchronize()

    def run_once():
        if args.backward:
            o = flash_attn_func_v2(q, k, v, causal=args.causal)
            g = torch.randn_like(o)
            o.backward(g, retain_graph=False)
            q.grad = None
            k.grad = None
            v.grad = None
        else:
            _ = flash_attn_func_v2(q, k, v, causal=args.causal)
        torch.cuda.synchronize()

    metric_names = _parse_list(args.metrics)
    kernel_regex = _parse_list(args.kernel_regex)

    result = gw_profile.do_range_profile(
        fn=run_once,
        list_metric_names=metric_names,
        list_kernel_names=kernel_regex,
        dump_path=args.dump_path,
    )

    range_profile = result.get("range_profile", {})
    merged = range_profile.get("merged", {})
    summary = {
        "range_name": range_profile.get("range_name"),
        "num_metrics_entries": merged.get("num_metrics_entries"),
        "num_captured_launches": merged.get("num_captured_launches"),
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "gqa_ratio": q_heads // kv_heads,
        "model_config": resolved_config,
        "dump_path": args.dump_path,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
