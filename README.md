# ◮ G-Watch

[![cuda](https://img.shields.io/badge/CUDA-Supported-brightgreen.svg?logo=nvidia)](https://phoenixos.readthedocs.io/en/latest/cuda_gsg/index.html#)
[![rocm](https://img.shields.io/badge/ROCm-Supported-brightgreen.svg?logo=amd)](https://phoenixos.readthedocs.io/en/latest/rocm_gsg/index.html)

<div align="center">
    <img src="https://github.com/mars-compute-ai/G-Watch/blob/main/docs/logo.jpg?raw=true" alt="G-Watch logo" width="350" />
</div>

**G-Watch** is an agentic toolbox for optimizing GPU kernels.
It features rich **Profiling** and **Program Analysis** capabilities on both NVIDIA and AMD GPUs.

## Installation

To get started, run:

```bash
curl -sSL https://raw.githubusercontent.com/mars-compute-ai/G-Watch/main/install.sh | bash
```

## Examples

The following examples demonstrate end-to-end agentic GPU kernel optimization using G-Watch:

- [Agentic FlashAttention-3 Optimization](examples/cuda/fa3/README.md) — Hopper (sm90)
- [Agentic FlashAttention-2 Optimization](examples/cuda/fa2/README.md) — Ampere (sm86)

## Citation
