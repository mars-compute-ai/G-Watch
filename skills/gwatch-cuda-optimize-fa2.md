# FA2 Auto Optimization Skill (G-Watch + FlashAttention2 Ampere)
This skill teaches the agent to profile and optimize FlashAttention2 (FA2) kernels on Ampere using G-Watch.


## Goal
Your primary objective is to iteratively optimize the FlashAttention-2 (FA2) kernel and significantly improve its **TFLOPs** performance.
To achieve this,
you must adopt a systematic, data-driven optimization strategy, seamlessly alternating between the provided profiling tools to diagnose bottlenecks, formulate hypotheses, apply code modifications, and validate performance gains.


## Input
- `GWATCH_PATH`: The path to the G-Watch codebase, provided by the user when invoking this skill.
- `FA_PATH`: The path to the flash-attention codebase, provided by the user when invoking this skill.


## Scope
Use this when you need to:
- measure FA2 throughput/FLOPs baseline.
- discover and select profiling metrics.
- run range profiling, PC sampling, intra-kernel tracing or conduct static binary analysis of FA2 kernels.
- modify FA2 source code and rebuild.
- iterate profiling until bottlenecks are reduced.

Target scripts:
- `$GWATCH_PATH/examples/cuda/fa2/do_flops_fa2.py`
- `$GWATCH_PATH/examples/cuda/fa2/do_range_profile_fa2.py`
- `$GWATCH_PATH/examples/cuda/fa2/do_pc_sampling_fa2.py`
- `$GWATCH_PATH/examples/cuda/fa2/do_trace_fa2.py`

Primary optimization source files:
- `FA_PATH/csrc/flash_attn/src/flash_fwd_kernel.h`
- `FA_PATH/csrc/flash_attn/src/flash_fwd_launch_template.h`
- `FA_PATH/csrc/flash_attn/src/kernel_traits.h`
- `FA_PATH/csrc/flash_attn/src/softmax.h`


## Tool 1: Baseline FLOPs
If you need end-to-end baseline performance numbers (latency and throughput KPI) before any optimization, you should use this tool.
Expected to get: `avg_ms`, `total_flops`, `tflops`.

```bash
export GWATCH_PATH=[User provided G-Watch Path]
python3 $GWATCH_PATH/examples/cuda/fa2/do_flops_fa2.py \
    --device-id 0 \
    --batch 8 \
    --seqlen 8192 \
    --nheads 16 \
    --headdim 256 \
    --dtype bf16 \
    --model-config $GWATCH_PATH/examples/cuda/model_config/llama4.json \
    --rep 50
```

Record at minimum:
- `avg_ms`
- `total_flops`
- `tflops`

Use this as the baseline KPI before any kernel changes.

For GQA sweeps, append `--model-config $GWATCH_PATH/examples/cuda/model_config/llama4.json` (or point to `qwen3.5.json` / `qwen3.json`), or override via `--q-heads`/`--kv-heads`.


## Tool 2: Select Metrics for Range Profiling
If you need to decide which hardware metrics are meaningful for bottleneck diagnosis and then measure per-kernel metric hotspots, you should use this tool.
Expected to get: selected metric list (4-10), top runtime-dominant kernels, and compute-vs-memory trend signals.

First inspect chip topology:

```bash
gwatch profile show-topo --chip a100
```

Then list candidates, for instance:

```bash
gwatch profile list-metrics --chip a100 --unit smsp --subunit sass
```

You can list metrics of other unit by your interest.
WARN that directly running the above command could print a lots of metric names at a time,
which would waste a lots of context to consume.
You could probably using grep to filter out specific content.

Expand concrete metric names:

```bash
gwatch profile show-metric-details --chip a100 --name <metric_base_name>
# or on some versions:
gwatch profile show-metric-details --chip a100 --name <metric_base_name>
```

Pick a focused set (4-10 metrics) that can separate likely bottlenecks:
- tensor-pipe utilization (HMMA pipe),
- active/elapsed cycle ratios,
- memory throughput/latency pressure (L2, DRAM),
- instruction- or dependency-related indicators.

After selecting proper metrics to be profile, you can run the profile by:

```bash
export GWATCH_PATH=[User provided G-Watch Path]
gwatch profile python3 $GWATCH_PATH/examples/cuda/fa2/do_range_profile_fa2.py \
    --device-id 0 \
    --batch 8 \
    --seqlen 8192 \
    --nheads 16 \
    --headdim 256 \
    --dtype bf16 \
    --metrics "sm__cycles_active.avg.pct_of_peak_sustained_elapsed" \
    --model-config $GWATCH_PATH/examples/cuda/model_config/llama4.json \
    --kernel-regex ".*flash.*" \
    --dump-path /tmp/fa2_range_profile.gwatch
```

Use `--model-config $GWATCH_PATH/examples/cuda/model_config/qwen3.5.json` to profile that GQA shape, or override `--q-heads`/`--kv-heads` manually.

Note that for `--metrics` argument, better to have one metric at a time.

### View Range Profile Results
Inspect results of the range profile using `gwatch show`:

```bash
gwatch show /tmp/fa2_range_profile.gwatch --mode range_profile \
  --kernel ".*flash.*" \ # Optional: Filter by kernel name regex
  --metric "sm__cycles_active.avg.pct_of_peak_sustained_elapsed" \ # Optional: Display specific metrics (comma-separated)
  --sort desc \ # Optional: Sort by metric value (asc or desc)
  --num 20 \ # Optional: Show top N entries (integer or 'all')
```

Key `gwatch show --mode range_profile` options:
- `--kernel <regex>`: Filters the kernels displayed by a regular expression match on their names.
- `--metric <names>`: Specifies a comma-separated list of exact metric names to display columns for. If omitted, all metrics captured are shown.
- `--sort <order>`: Sorts the displayed entries by the first metric specified (if `--metric` is used) or the first available metric. `asc` for ascending, `desc` for descending.
- `--num <N|all>`: Limits the number of rows displayed to `N` (integer) or shows all rows (`all`).

## Tool 3: PC Sampling (PC + Stall Reasons)
If you need instruction-level hotspot locations and dominant stall reasons (where cycles are getting stuck), you should use this tool.
Expected to get: top PCs by stall weight, stall-reason breakdown per PC, and source/SASS-correlated hotspots.

```bash
export GWATCH_PATH=[User provided G-Watch Path]
gwatch profile python3 $GWATCH_PATH/examples/cuda/fa2/do_pc_sampling_fa2.py \
    --device-id 0 \
    --batch 8 \
    --seqlen 8192 \
    --nheads 16 \
    --headdim 256 \
    --dtype bf16 \
    --kernel-regex ".*flash.*" \
    --model-config $GWATCH_PATH/examples/cuda/model_config/llama4.json \
    --rep 50 \
    --dump-path /tmp/fa2_pc_sampling.gwatch
```

Set `--model-config $GWATCH_PATH/examples/cuda/model_config/qwen3.json` to reuse that config's head ratio.

### View PC Sampling Results
Inspect results of the PC sampling using `gwatch show`:

```bash
gwatch show /tmp/fa2_pc_sampling.gwatch --mode pc_sampling \
  --kernel ".*flash.*" \ # Optional: Filter by kernel name regex
  --sort desc \ # Optional: Sort PCs by total stall count (asc or desc)
  --num 20 \ # Optional: Show top N PCs per kernel (integer or 'all')
```

Key `gwatch show --mode pc_sampling` options:
- `--kernel <regex>`: Filters the kernels displayed by a regular expression match on their names.
- `--sort <order>`: Sorts the displayed PCs within each kernel by their total stall count. `asc` for ascending, `desc` for descending.
- `--num <N|all>`: Limits the number of PCs displayed per kernel to `N` (integer) or shows all PCs (`all`).


## Tool 4: Intra-Kernel Trace
If you need scope/timeline behavior inside a kernel (phase-level behavior across marked regions), you should use this tool.
Expected to get: trace records for instrumented scopes; if empty, this indicates missing trace hints and you should rely on range + PC + SASS.

The usage of this tool contains three steps:

### Step 1 Code Instrumentation
you can trace a specific region in the FA2 source code that doesn't have markers by adding them manually:

1. **Include Header**: Ensure `#include "gwatch/cuda/trace.hpp"` is present in the target `.h` file.
2. **Add Markers**: Wrap code with `GWATCH_CUDA_KERNEL_SCOPE_START(TAG_ID);` and `GWATCH_CUDA_KERNEL_SCOPE_END(TAG_ID);`. Use a unique integer for `TAG_ID`.
3. **Map Names**: Add your `TAG_ID` and a descriptive name to `name_id_map` in `$GWATCH_PATH/examples/cuda/fa2/do_trace_fa2.py`.
4. **Rebuild**: Follow the rebuild steps below to apply changes.

Example from `flash_fwd_kernel.h`:
```cpp
GWATCH_CUDA_KERNEL_SCOPE_START(100);
// load Q tile from global to shared memory
GWATCH_CUDA_KERNEL_SCOPE_END(100);

GWATCH_CUDA_KERNEL_SCOPE_START(110);
// load K tile from global to shared memory
GWATCH_CUDA_KERNEL_SCOPE_END(110);

GWATCH_CUDA_KERNEL_SCOPE_START(130);
// GEMM: S = Q * K^T
GWATCH_CUDA_KERNEL_SCOPE_END(130);
```

### Step 2: Recompile
Then,
you can recompile the instrumented FA2 kernel by:

```bash
cd FA_PATH/
export FLASH_ATTN_CUDA_ARCHS="80"
export FLASH_ATTENTION_DISABLE_HDIM64="TRUE"
export FLASH_ATTENTION_DISABLE_HDIM96="TRUE"
export FLASH_ATTENTION_DISABLE_HDIM192="TRUE"
export FLASH_ATTENTION_DISABLE_HDIM256="TRUE"
export FLASH_ATTENTION_DISABLE_BACKWARD="TRUE"
export FLASH_ATTENTION_DISABLE_FP8="TRUE"
python3 setup.py install
```

### Step 3: Run Tracing
Finally,
you can run the tracing by:

```bash
export GWATCH_PATH=[User provided G-Watch Path]
gwatch profile python3 $GWATCH_PATH/examples/cuda/fa2/do_trace_fa2.py \
    --device-id 0 \
    --batch 8 \
    --seqlen 8192 \
    --nheads 16 \
    --headdim 256 \
    --dtype bf16 \
    --kernel-regex ".*flash.*" \
    --model-config $GWATCH_PATH/examples/cuda/model_config/llama4.json \
    --warmup 25 \
    --rep 100 \
    --dump-path /tmp/fa2_trace.gwatch
```

Note that `do_trace_fa2.py` internally uses a `name_id_map` to map numeric scope IDs to human-readable names (e.g., `load_K`).
Trace is most useful when kernels contain trace hints/scope markers.
If no trace records are returned, continue with range + PC + SASS analysis.

### View Intra-Kernel Trace Results
After running `do_trace_fa2.py`, inspect the trace results using `gwatch show` in `trace` mode.

```bash
gwatch show /tmp/fa2_trace.gwatch --mode trace \
  --tid "[0,31]" \ # Optional: filter by thread IDs
  --longest-bubble \ # Optional: highlight the longest gap between events
  --longest "load_K" \ # Optional: find the longest instance of a specific event
  --stats \ # Optional: show aggregate statistics for all events
```

Key `gwatch show --mode trace` options:
- `--tid <ranges>`: Filters and displays specific thread IDs (e.g., `"[0,31], [64,95]"`).
- `--longest-bubble`: Identifies and highlights the longest idle period (bubble) between events across all threads.
- `--longest <EVENT_NAME>`: Finds the longest instance of a specific event (scope) by name.
- `--stats`: Displays aggregate statistics (count, min, max, avg, stddev, total duration) for each event type and for bubbles.


## Tool 5: Runtime-Control Binary Analysis (`KernelDefSASS`)
If you need runtime-loaded kernel binary structure (SASS instructions and address-to-line mapping) to conduct binary analysis, you should use this tool.
Expected to get: FA2 kernel instance, SASS instruction details, etc.

Use `ProfileContext` runtime-control APIs inside a `gwatch profile` run to inspect loaded FA2 kernels:

```python
import re
import torch
import gwatch.cuda.profile as gw_profile
import gwatch.cuda.binary as gw_binary

pcontext = gw_profile.ProfileContext()

# run FA2 workload first so kernels are loaded
run_fa2_once()
torch.cuda.synchronize()

kernel_map = pcontext.get_map_kerneldef_sass()
fa2_items = []
for mangled, kdef in kernel_map.items():
    demangled = gw_binary.BinaryUtility.demangle(mangled)
    if re.search(r"flash_fwd|flash_bwd|FlashAttn|sm80", mangled) or re.search(r"flash_fwd|flash_bwd|FlashAttn|sm80", demangled):
        fa2_items.append((mangled, demangled, kdef))

print("fa2 kernels:", len(fa2_items))

if fa2_items:
    mangled, demangled, kdef = fa2_items[0]
    print("kernel:", mangled)
    print("demangled:", demangled)
    print("num_instructions:", len(kdef.get_list_instruction()))
    print("has_line_map:", len(kdef.map_address_to_line) > 0)
```

Use this to:
- fetch runtime-loaded FA2 `KernelDefSASS`,
- inspect instruction streams and address-to-line mapping,
- correlate with PC sampling hotspots before editing source.


### Tool Utilization Strategy
You should strategically use above profiling tools based on your current debugging context:
- **Tool 1 (Baseline FLOPs):** Use this to measure the macroscopic throughput (TFLOPS) of the FA2 kernel. This is your ultimate source of truth for validating whether a code change yielded an actual performance improvement.
- **Tool 2 (Select Metrics & Range Profiling):** Leverage this to understand the macro-level hardware characteristics. By analyzing specific hardware counter metrics (guided by SASS/source analysis), determine if the kernel is memory-bound or compute-bound (Roofline model). Use it to isolate overheads across different units/subunits/pipestages (e.g., memory access, compute, or synchronization overheads).
- **Tool 3 (PC Sampling & Stall Reasons):** When you identify a general bottleneck, use this for fine-grained localization. Discover which specific instructions suffer the longest stalls and map these back to potential source code locations to pinpoint the exact bottleneck.
- **Tool 4 (Intra-Kernel Trace):** Utilize this powerful tracing tool to observe thread-block and warp-level execution timelines. This is useful for understanding the double-buffered pipeline and GEMM/softmax overlap behavior in FA2.
- **Tool 5 (Runtime-Control Binary Analysis):** Use this to statically analyze the SASS program. Deeply comprehend the generated instructions at the binary level to understand the compiler's behavior and the actual hardware execution semantics.

But DO NOTED THAT DONT RUN THESE TOOLS CONCURRENTLY!


## Practical Optimization Loop
1. Benchmarking TFLOPs of the original baseline implementation (via `do_flops_fa2.py`).
2. Profile the implementation via tools mentioned above (range_profile/pc_sampling/trace).
3. Form one concrete hypothesis (for example, reduce shared memory bank conflicts in QK GEMM or optimize softmax computation).
4. Change the smallest relevant code region.
5. Rebuild.
6. Re-benchmarking TFLOPs with identical settings.
6. Keep the code change only if end-to-end latency/TFLOPs improves without correctness regressions.
7. Repeat 2-6 until significant TFLOPs improvement is observed.

Note that:
- Please avoid stacking many unvalidated changes in one iteration.
- You're also allowed to write your own python script in /tmp for specific analysing needs.


## Modify FA2 Source and Rebuild
Key files related to FlashAttention-2 include but not limited to:
- `FA_PATH/csrc/flash_attn/src/flash_fwd_kernel.h`
- `FA_PATH/csrc/flash_attn/src/flash_fwd_launch_template.h`
- `FA_PATH/csrc/flash_attn/src/kernel_traits.h`
- `FA_PATH/csrc/flash_attn/src/softmax.h`

Rebuild command:

```bash
cd FA_PATH/
export FLASH_ATTN_CUDA_ARCHS="80"
export FLASH_ATTENTION_DISABLE_HDIM64="TRUE"
export FLASH_ATTENTION_DISABLE_HDIM96="TRUE"
export FLASH_ATTENTION_DISABLE_HDIM192="TRUE"
export FLASH_ATTENTION_DISABLE_HDIM256="TRUE"
export FLASH_ATTENTION_DISABLE_BACKWARD="TRUE"
export FLASH_ATTENTION_DISABLE_FP8="TRUE"
python3 setup.py install
```

Note that this rebuild process could be a bit long.


## Common Pitfalls
- Running profiling scripts without `gwatch profile`. Note that all profiling and runtime-control steps must run under G-Watch runtime injection:
```bash
gwatch profile python3 <script>.py ...
```
Without `gwatch profile`, profiling and runtime kernel-definition APIs may return empty/no-op data.
- Comparing results across different workload shapes or dtypes.
- Using too many metrics at once and increasing replay overhead.
- Trusting one profiler mode alone; always cross-check range + PC (and trace when available).
- Attempting runtime kernel lookup before FA2 kernels are actually launched.


### Deliverables
Upon completing your optimization journey, you must provide a comprehensive final output containing the following 4 sections:

1. **Performance Improvement Report:**
   - Final optimal TFLOPS achieved.
   - Absolute TFLOPS improvement and relative percentage increase compared to the initial baseline.
2. **Code Modification Report:**
   - A detailed breakdown of what specific code changes were made.
   - The estimated or measured percentage of performance gain attributed to each individual modification.
3. **Optimization Process Report:**
   - A complete, step-by-step chronological log of your optimization journey.
   - Detailed documentation of *every* attempt (including the tool used, the finding, the hypothesis, and the result).
   - **Crucial:** You must report *both* successful (positive) and failed (negative) optimization attempts to demonstrate your reasoning process.
4. **Modified FA2 Source Code:**
   - Provide the complete, final optimized source code (or a comprehensive unified `diff`/patch) for all modified FA2 files.
