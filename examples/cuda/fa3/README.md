# FlashAttention3 Profiling Examples

This directory contains simple scripts that profile FlashAttention3 kernels using G-Watch:
- range profiling via `do_range_profile`
- PC sampling via `do_pc_sampling`
- kernel tracing via `do_trace`

Both scripts must be executed with G-Watch runtime injection.

## Range Profiling (do_range_profile)

Forward pass, default settings:

```bash
gwatch profile python3 do_range_profile_fa3.py
```

Backward pass, causal, custom size:

```bash
gwatch profile python3 do_range_profile_fa3.py \
  --backward --causal --seqlen 4096 --nheads 32 --headdim 128 \
  --metrics FBSP.TriageCompute.dramc__cycles_elapsed.avg \
  --kernel-regex ".*flash.*" \
  --dump-path ./fa3_range_profile.gwatch
```

View the `.gwatch` result:

```bash
gwatch show ./fa3_range_profile.gwatch --mode range_profile
```

## PC Sampling (do_pc_sampling)

Default:

```bash
gwatch profile python3 do_pc_sampling_fa3.py
```

Custom kernel filter and repetition:

```bash
gwatch profile python3 do_pc_sampling_fa3.py \
  --rep 100 --kernel-regex ".*flash.*" \
  --dump-path ./fa3_pc_sampling.gwatch
```

View the `.gwatch` result:

```bash
gwatch show ./fa3_pc_sampling.gwatch --mode pc_sampling
```

## Tracing (do_trace)

Default:

```bash
gwatch profile python3 do_trace_fa3.py
```

Custom kernel filter and output:

```bash
gwatch profile python3 do_trace_fa3.py \
  --kernel-regex ".*flash.*" \
  --dump-path ./fa3_trace.gwatch
```

View the `.gwatch` result:

```bash
gwatch show ./fa3_trace.gwatch --mode trace
```

## FLOPs (do_flops)

Default:

```bash
python3 do_flops_fa3.py
```

Custom size and repetitions:

```bash
python3 do_flops_fa3.py \
  --batch 8 --seqlen 4096 --nheads 32 --headdim 128 \
  --dtype bf16 --rep 100
```

## Notes

- These scripts expect FlashAttention3 to be available at `/root/workload/flash-attention/hopper`.
- Use `--dtype bf16` or `--dtype fp16` to match your kernel support.
