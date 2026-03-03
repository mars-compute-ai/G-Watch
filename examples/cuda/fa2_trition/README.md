# Triton FlashAttention2 Profiling Examples

This directory contains scripts that profile a Triton implementation of FlashAttention2 using G-Watch:
- range profiling via `do_range_profile`
- PC sampling via `do_pc_sampling`
- kernel tracing via `do_trace`
- FLOPs measurement via `do_flops`
- latency measurement via `do_latency`

The Triton FA2 kernel is defined in `test.py` and uses the `@triton.autotune` / `@triton.jit` decorators.

Profiling scripts (except FLOPs/latency) must be executed with G-Watch runtime injection.

## Tensor Layout

Unlike the native FA2 which uses `(batch, seqlen, heads, headdim)`, the Triton FA2 uses `(batch, heads, seqlen, headdim)`.

## FLOPs (do_flops)

Default:

```bash
python3 do_flops_fa2_triton.py
```

Custom size and repetitions:

```bash
python3 do_flops_fa2_triton.py \
  --batch 4 --seqlen 1024 --nheads 32 --headdim 64 \
  --dtype fp16 --causal --rep 100
```

## Latency (do_latency)

Default (batch=1):

```bash
python3 do_latency_fa2_triton.py
```

## Range Profiling (do_range_profile)

Forward pass, default settings:

```bash
gwatch profile python3 do_range_profile_fa2_triton.py
```

Backward pass, causal, custom size:

```bash
gwatch profile python3 do_range_profile_fa2_triton.py \
  --backward --causal --seqlen 4096 --nheads 32 --headdim 64 \
  --metrics sm__cycles_active.avg.pct_of_peak_sustained_elapsed \
  --kernel-regex ".*_attn_fwd.*,.*_attn_bwd.*" \
  --dump-path ./fa2_triton_range_profile.gwatch
```

View the `.gwatch` result:

```bash
gwatch show ./fa2_triton_range_profile.gwatch --mode range_profile
```

## PC Sampling (do_pc_sampling)

Default:

```bash
gwatch profile python3 do_pc_sampling_fa2_triton.py
```

Custom kernel filter and repetition:

```bash
gwatch profile python3 do_pc_sampling_fa2_triton.py \
  --rep 100 --kernel-regex ".*_attn_fwd.*" \
  --dump-path ./fa2_triton_pc_sampling.gwatch
```

View the `.gwatch` result:

```bash
gwatch show ./fa2_triton_pc_sampling.gwatch --mode pc_sampling
```

## Tracing (do_trace)

Default:

```bash
gwatch profile python3 do_trace_fa2_triton.py
```

Custom kernel filter and output:

```bash
gwatch profile python3 do_trace_fa2_triton.py \
  --kernel-regex ".*_attn_fwd.*" \
  --dump-path ./fa2_triton_trace.gwatch
```

View the `.gwatch` result:

```bash
gwatch show ./fa2_triton_trace.gwatch --mode trace
```

## Backward Pass Scripts

Backward-specific scripts are in the `bwd/` directory:
- `bwd/do_flops_fa2_triton_bwd.py` - backward FLOPs measurement
- `bwd/do_range_profile_fa2_triton_bwd.py` - backward range profiling
- `bwd/do_pc_sampling_fa2_triton_bwd.py` - backward PC sampling
- `bwd/do_trace_fa2_triton_bwd.py` - backward kernel tracing

## Notes

- These scripts use the Triton FA2 implementation from `test.py` in this directory.
- The first run includes Triton JIT compilation overhead; warmup iterations handle this.
- Triton FA2 supports `float16` and `float8_e5m2` dtypes. Default is `fp16`.
- Use `--sm-scale` to set the softmax scale factor (default: 1.0).
- The same model config files (llama4, qwen3, qwen3.5) are shared via symlinks to the `fa2/config/` directory.
- Triton FA2 does not natively support GQA (different Q/K/V head counts); all heads use the same count.
