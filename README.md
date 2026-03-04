# ◮ G-Watch

[![cuda](https://img.shields.io/badge/CUDA-Supported-brightgreen.svg?logo=nvidia)]()
[![rocm](https://img.shields.io/badge/ROCm-Supported-brightgreen.svg?logo=amd)]()

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

- [Agentic FlashAttention-3 Optimization](examples/cuda/fa3) — Hopper (sm90)
- [Agentic FlashAttention-2 Optimization](examples/cuda/fa2) — Ampere (sm86)

## Citation


If you use G-Watch in your research or project, please cite this repository:

```bibtex
@software{gwatch_repo,
  author = {Mars Compute AI},
  title = {G-Watch: An Agentic Toolbox for Optimizing GPU Kernels},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{[https://github.com/mars-compute-ai/G-Watch](https://github.com/mars-compute-ai/G-Watch)}}
}
