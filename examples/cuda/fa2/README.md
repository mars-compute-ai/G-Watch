# FlashAttention2 Profiling Examples

This directory contains simple scripts that profile FlashAttention2 (SM80, A100) kernels using G-Watch:
- range profiling via `do_range_profile`
- PC sampling via `do_pc_sampling`
- kernel tracing via `do_trace`
- FLOPs measurement via `do_flops`

Profiling scripts must be executed with G-Watch runtime injection.

## Range Profiling (do_range_profile)

Forward pass, default settings:

```bash
gwatch profile python3 do_range_profile_fa2.py
```

Backward pass, causal, custom size:

```bash
gwatch profile python3 do_range_profile_fa2.py \
  --backward --causal --seqlen 4096 --nheads 32 --headdim 128 \
  --metrics sm__cycles_active.avg.pct_of_peak_sustained_elapsed \
  --kernel-regex ".*flash.*" \
  --dump-path ./fa2_range_profile.gwatch
```

View the `.gwatch` result:

```bash
gwatch show ./fa2_range_profile.gwatch --mode range_profile
```

## PC Sampling (do_pc_sampling)

Default:

```bash
gwatch profile python3 do_pc_sampling_fa2.py
```

Custom kernel filter and repetition:

```bash
gwatch profile python3 do_pc_sampling_fa2.py \
  --rep 100 --kernel-regex ".*flash.*" \
  --dump-path ./fa2_pc_sampling.gwatch
```

View the `.gwatch` result:

```bash
gwatch show ./fa2_pc_sampling.gwatch --mode pc_sampling
```

## Tracing (do_trace)

Default:

```bash
gwatch profile python3 do_trace_fa2.py
```

Custom kernel filter and output:

```bash
gwatch profile python3 do_trace_fa2.py \
  --kernel-regex ".*flash.*" \
  --dump-path ./fa2_trace.gwatch
```

View the `.gwatch` result:

```bash
gwatch show ./fa2_trace.gwatch --mode trace
```

## FLOPs (do_flops)

Default:

```bash
python3 do_flops_fa2.py
```

Custom size and repetitions:

```bash
python3 do_flops_fa2.py \
  --batch 8 --seqlen 4096 --nheads 32 --headdim 128 \
  --dtype bf16 --rep 100
```

## Notes

- These scripts require the `flash-attn` package (v2.x) to be installed.
- FA2 targets SM80 (A100) GPUs. On Hopper GPUs, FA2 kernels still run but use SM80 code paths.
- Use `--dtype bf16` or `--dtype fp16` to match your kernel support.
- The same model config files (llama4, qwen3, qwen3.5) are shared with FA3 examples for GQA sweeps.
