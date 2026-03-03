import argparse
import json
import os
import sys

import torch

try:
    from flash_attn import flash_attn_func as flash_attn_func_v2
except ImportError:
    flash_attn_func_v2 = None

# Directory containing model config JSONs (shared with the fwd scripts).
_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")


def _parse_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _bwd_flops(batch, nheads, seqlen_q, seqlen_k, headdim, headdim_v, causal=False, window_size=(-1, -1)):
    """Backward FLOPs for FlashAttention-2.

    The backward pass performs 5 matmuls per (M, N) tile pair:
      dS  = dO  @ V^T    (seqlen_q x headdim) @ (headdim x seqlen_k)
      dP  = dS  * P      (element-wise, ignored in FLOPs)
      dQ  = dP  @ K      (seqlen_q x seqlen_k) @ (seqlen_k x headdim)
      dK  = dP^T @ Q     (seqlen_k x seqlen_q) @ (seqlen_q x headdim)
      dV  = P^T @ dO     (seqlen_k x seqlen_q) @ (seqlen_q x headdim)
      S   = Q   @ K^T    (recomputed in the backward, same as fwd)

    Total = 5 matmuls of cost 2*M*N*K each, so 2.5x the forward FLOPs
    (fwd = 2 matmuls: QK^T and P@V).
    """
    if causal:
        avg_seqlen = (max(0, seqlen_k - seqlen_q) + seqlen_k) / 2
    else:
        if window_size == (-1, -1):
            avg_seqlen = seqlen_k
        else:
            row_idx = torch.arange(seqlen_q, device="cuda")
            col_left = torch.maximum(row_idx + seqlen_k - seqlen_q - window_size[0], torch.tensor(0, device="cuda"))
            col_right = torch.minimum(
                row_idx + seqlen_k - seqlen_q + window_size[1], torch.tensor(seqlen_k - 1, device="cuda")
            )
            avg_seqlen = (col_right - col_left + 1).float().mean().item()
    fwd_flops = batch * nheads * 2 * seqlen_q * avg_seqlen * (headdim + headdim_v)
    return 2.5 * fwd_flops


def _time_bwd_ms(q, k, v, causal, warmup, rep):
    """Time the backward pass only (forward is excluded from timing)."""
    # Warmup
    for _ in range(warmup):
        o = flash_attn_func_v2(q, k, v, causal=causal)
        g = torch.randn_like(o)
        o.backward(g, retain_graph=False)
        q.grad = k.grad = v.grad = None
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(rep):
        o = flash_attn_func_v2(q, k, v, causal=causal)
        g = torch.randn_like(o)
        torch.cuda.synchronize()
        start.record()
        o.backward(g, retain_graph=False)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        q.grad = k.grad = v.grad = None

    return sum(times) / len(times), times


def _load_model_heads(config_path: str):
    if not config_path:
        raise ValueError("config_path is required")
    resolved = config_path
    if not os.path.isabs(resolved):
        candidate = os.path.join(_CONFIG_DIR, resolved)
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
    parser = argparse.ArgumentParser(description="Measure FlashAttention2 backward FLOPs and throughput.")
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
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    args = parser.parse_args()

    if flash_attn_func_v2 is None:
        raise RuntimeError("FlashAttention v2 not found. Install flash-attn package.")

    device = f"cuda:{args.device_id}"
    dtype = _parse_dtype(args.dtype)
    q_heads, kv_heads, resolved_config = _resolve_heads(args)

    q = torch.randn(args.batch, args.seqlen, q_heads, args.headdim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(args.batch, args.seqlen, kv_heads, args.headdim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(args.batch, args.seqlen, kv_heads, args.headdim, device=device, dtype=dtype, requires_grad=True)

    avg_ms, all_times = _time_bwd_ms(q, k, v, causal=args.causal, warmup=args.warmup, rep=args.rep)
    total_flops = _bwd_flops(
        batch=args.batch,
        nheads=q_heads,
        seqlen_q=args.seqlen,
        seqlen_k=args.seqlen,
        headdim=args.headdim,
        headdim_v=args.headdim,
        causal=args.causal,
    )
    tflops = total_flops / (avg_ms * 1e-3) / 1e12

    summary = {
        "batch": args.batch,
        "seqlen": args.seqlen,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "gqa_ratio": q_heads // kv_heads,
        "model_config": resolved_config,
        "headdim": args.headdim,
        "dtype": str(dtype).replace("torch.", ""),
        "causal": args.causal,
        "avg_ms": avg_ms,
        "min_ms": min(all_times),
        "max_ms": max(all_times),
        "total_flops": total_flops,
        "tflops": tflops,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
