import argparse
import json
import os
import sys

import torch

from gwatch.cuda.trace import do_trace

# Add flash-attention (v3) to sys.path
sys.path.append("/root/workload/flash-attention")
sys.path.append("/root/workload/flash-attention/hopper")

try:
    from flash_attn_interface import flash_attn_func as flash_attn_func_v3
except ImportError:
    flash_attn_func_v3 = None

fa3_kernel_regex = "^_ZN7cutlass13device_kernelIN5flash20enable_sm90_or_laterINS1_16FlashAttnFwdSm90INS1_25CollectiveMainloopFwdSm90ILi2EN4cute5tupleIJNS5_1CILi2EEENS7_ILi1EEES9_EEENS6_IJNS7_ILi128EEENS7_ILi176EEESB_EEELi128ENS_10bfloat16_tEfNS_4arch4Sm90ELb0ELb0ELb0ELb0ELb0ELb0ELb0ELb1ELb1ELb0ELb0ELb0EEENS1_21CollectiveEpilogueFwdINS6_IJSB_SB_SC_EEESA_SE_SG_Li256ELb0ELb0ELb0ELb0EEENS1_29StaticPersistentTileSchedulerILb0EEEEEEEEEvNT_6ParamsE$"

name_id_map = {
   # producer
    100: "[producer] load page table",
    110: "[producer] acquire v",
    111: "[producer] issue TMA load v",
    112: "[producer] copy Vt to V",
    120: "[producer] acquire q",
    121: "[producer] issue load q",
    122: "[producer] issue load qv",
    130: "[producer] wait barrier o",
    140: "[producer] acquire k",
    141: "[producer] issue TMA load k",

    # consumer
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
    parser = argparse.ArgumentParser(description="Trace FlashAttention3 kernels with do_trace.")
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
    parser.add_argument("--kernel-regex", type=str, default=fa3_kernel_regex)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--dump-path", type=str, default="./fa3_trace.gwatch")
    args = parser.parse_args()

    if flash_attn_func_v3 is None:
        raise RuntimeError("FlashAttention v3 not found.")

    device = f"cuda:{args.device_id}"
    dtype = _parse_dtype(args.dtype)
    q_heads, kv_heads, resolved_config = _resolve_heads(args)

    q = torch.randn(args.batch, args.seqlen, q_heads, args.headdim, device=device, dtype=dtype)
    k = torch.randn(args.batch, args.seqlen, kv_heads, args.headdim, device=device, dtype=dtype)
    v = torch.randn(args.batch, args.seqlen, kv_heads, args.headdim, device=device, dtype=dtype)

    # Warmup to avoid one-time init kernels.
    with torch.no_grad():
        _ = flash_attn_func_v3(q, k, v, causal=args.causal)
    torch.cuda.synchronize()

    def run_once():
        _ = flash_attn_func_v3(q, k, v, causal=args.causal)

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
            "note": "no trace results returned (kernel may not contain trace hints)",
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
