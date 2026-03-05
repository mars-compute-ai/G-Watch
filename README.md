# ◮ G-Watch

[![cuda](https://img.shields.io/badge/CUDA-Supported-brightgreen.svg?logo=nvidia)]()
[![rocm](https://img.shields.io/badge/ROCm-Supported-brightgreen.svg?logo=amd)]()

<div align="center">
    <img src="https://github.com/mars-compute-ai/G-Watch/blob/main/docs/logo.jpg?raw=true" alt="G-Watch logo" width="350" />
</div>

**G-Watch** is an agentic toolbox for optimizing GPU kernels.
It features rich **Profiling** and **Program Analysis** capabilities on both NVIDIA and AMD GPUs.

## Installation

1. Clone repository

    ```bash
    # clone repository
    git clone https://github.com/mars-compute-ai/G-Watch
    cd G-Watch/scripts/docker
    ```

2. Create docker container

- For CUDA platform
    ```bash
    # start a CUDA 12.8 docker container with id 1,
    # the container name would be gw_${user_name}_cuda_12_8_1
    bash run_cuda_12_8.sh -s 1

    # NO NEED to run following command, just mark them here
    # close a CUDA 12.8 docker container with id 1,
    bash run_cuda_12_8.sh -c 1

    # enter a CUDA 12.8 docker container with id 1,
    bash run_cuda_12_8.sh -e 1
    ```

- For ROCm platform
    ```bash
    # start a ROCm 7.2 docker container with id 1,
    # the container name would be gw_${user_name}_rocm_7_2_1
    bash run_rocm_7_2.sh -s 1

    # NO NEED to run following command, just mark them here
    # close a ROCm 7.2 docker container with id 1,
    bash run_rocm_7_2.sh -c 1

    # enter a ROCm 7.2 docker container with id 1,
    bash run_rocm_7_2.sh -e 1
    ```

    By running the script for start/enter a container, the repository would be mapped to `/root` inside the container.


3. Install G-Watch inside the container
    ```bash
    # install prerequiries
    apt-get update -o APT::Sandbox::User=root
    apt-get install -y curl git -o APT::Sandbox::User=root

    # install gwatch
    curl -sSL https://raw.githubusercontent.com/mars-compute-ai/G-Watch/main/install.sh | bash
    ```

    Furthermore, you could install your prefer code agent inside your container as well.


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
