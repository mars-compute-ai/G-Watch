# Agentic FlashAttention-2 Optimization

This guide walks you through setting up FlashAttention-2 and using the G-Watch agent to automatically optimize its CUDA kernels on Ampere.

## 0. Clone G-Watch

Start by cloning the G-Watch repository and navigating into it:

```bash
git clone https://github.com/mars-compute-ai/G-Watch.git
cd G-Watch
export REPO_PATH=$PWD
```

All subsequent commands assume you are running from the G-Watch root directory.

## 1. Clone and Build FlashAttention-2

Next, clone the FlashAttention repository and check out the known-good commit that this example is tested against:

```bash
cd $REPO_PATH
mkdir workload && cd workload
git clone --recursive https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout d146efff6f3226f465f1b4f089eaefe52c475e9c
```

> **Note:** The flash-attention repo must be cloned under `$REPO_PATH/workload/flash-attention`,
> as this path is hardcoded in the skill.

Next, apply the patch that enables a fast, scoped build for FA-2 with trace instrumentation:

```bash
cd $REPO_PATH/workload/flash-attention
git apply $REPO_PATH/examples/cuda/fa2/fa2_fast_build.patch
```

Now build FA-2. The environment variables below narrow the build scope to keep compilation fast — only the forward-pass kernel with hdim256, BF16, on Ampere (sm80) is compiled:

```bash
cd $REPO_PATH/workload/flash-attention/
export FLASH_ATTN_CUDA_ARCHS="80"
export FLASH_ATTENTION_DISABLE_HDIM64="TRUE"
export FLASH_ATTENTION_DISABLE_HDIM96="TRUE"
export FLASH_ATTENTION_DISABLE_HDIM192="TRUE"
export FLASH_ATTENTION_DISABLE_BACKWARD="TRUE"
export FLASH_ATTENTION_DISABLE_FP8="TRUE"
python3 setup.py install
```

## 2. Verify the Installation

Once the build completes, run the FLOPS benchmark to confirm FA-2 is installed and produces valid results:

```bash
cd $REPO_PATH/examples/cuda/fa2
python3 do_flops_fa2.py
```

## 3. Install Skills

With the workload ready, install the G-Watch skill definitions that teach the agent how to optimize FA-2:

```bash
cd $REPO_PATH/skills
./install_skills.sh
```

## 4. Start the Agent

Everything is set up. Launch the optimization agent by running the following slash command inside your code agent:

```bash
/gwatch-cuda-optimize-fa2
```
