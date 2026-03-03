# Agentic FlashAttention-3 Optimization

This guide walks you through setting up FlashAttention-3 and using the G-Watch agent to automatically optimize its CUDA kernels.

## 0. Clone G-Watch

Start by cloning the G-Watch repository and navigating into it:

```bash
git clone https://github.com/mars-compute-ai/G-Watch.git
cd G-Watch
export REPO_PATH=$PWD
```

All subsequent commands assume you are running from the G-Watch root directory.

## 1. Clone and Build FlashAttention-3

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

Next, apply the patch that integrates PTX instrumentation into the FA-3 built binary:

```bash
cd $REPO_PATH/workload/flash-attention
git apply $REPO_PATH/examples/cuda/fa3/fa3_build_with_ptx.patch
```

Now build FA-3. The environment variables below narrow the build scope to keep compilation fast — only the forward-pass kernel with hdim128, FP16, on Hopper is compiled:

```bash
cd $REPO_PATH/workload/flash-attention/hopper/
export FLASH_ATTENTION_DISABLE_HDIM64="TRUE"
export FLASH_ATTENTION_DISABLE_HDIM96="TRUE"
export FLASH_ATTENTION_DISABLE_HDIM192="TRUE"
export FLASH_ATTENTION_DISABLE_HDIM256="TRUE"
export FLASH_ATTENTION_DISABLE_BACKWARD="TRUE"
export FLASH_ATTENTION_DISABLE_FP8="TRUE"
export FLASH_ATTENTION_DISABLE_SM80="TRUE"
python3 setup.py install
```

## 2. Verify the Installation

Once the build completes, run the FLOPS benchmark to confirm FA-3 is installed and produces valid results:

```bash
cd $REPO_PATH/examples/cuda/fa3
python3 do_flops_fa3.py
```

## 3. Install Skills

With the workload ready, install the G-Watch skill definitions that teach the agent how to optimize FA-3:

```bash
cd $REPO_PATH/skills
./install_skills.sh
```

## 4. Start the Agent Loop

Everything is set up. Launch the optimization agent by running the following slash command inside your code agent:

```bash
/gwatch-cuda-optimize-fa3
```

## 5. Results

We ran the agent through this workflow multiple times and it successfully optimized the FA-3 kernel:

<div align="center">
    <img src="../../../docs/fa3.png" alt="FA-3 optimization results" width="100%" />
</div>
