#!/bin/bash

# Define a function to run commands with or without sudo
run_cmd() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
    else
        if command -v sudo > /dev/null 2>&1; then
            sudo "$@"
        else
            echo "Error: This script requires root privileges, but sudo is not installed."
            exit 1
        fi
    fi
}

# 1. Detect OS and Install All System Dependencies
if [ -f /etc/debian_version ]; then
    echo "Detected Ubuntu/Debian system. Installing dependencies..."
    run_cmd apt-get -o APT::Sandbox::User=root update
    run_cmd apt-get -o APT::Sandbox::User=root install -y \
        curl python3 python3-pip libelf-dev protobuf-compiler libwebsockets-dev libnuma-dev
    # Update shared library cache
    run_cmd ldconfig

elif [ -f /etc/redhat-release ]; then
    echo "Detected CentOS/RHEL system. Installing dependencies..."
    run_cmd yum install -y \
        curl python3 python3-pip elfutils-libelf-devel protobuf-compiler libwebsockets-devel numactl-devel
    # Update shared library cache
    run_cmd ldconfig
else
    echo "Unsupported OS. Please manually install: curl, libelf, protobuf, libwebsockets, and libnuma."
fi

# 2. Detect GPU Type
if command -v nvidia-smi > /dev/null 2>&1; then
    GPU_TYPE="cuda"
    echo "NVIDIA GPU detected. Preparing to install G-Watch (CUDA)..."
elif command -v rocm-smi > /dev/null 2>&1; then
    GPU_TYPE="rocm"
    echo "AMD GPU detected. Preparing to install G-Watch (ROCm)..."
else
    echo "No GPU driver detected. Please specify installation type (cuda/rocm):"
    read -r GPU_TYPE
fi

# 3. Fetch Latest Release Info from GitHub API
REPO="mars-compute-ai/G-Watch"
API_URL="https://api.github.com/repos/$REPO/releases/latest"

echo "Fetching latest release metadata from GitHub..."
WHL_URL=$(curl -s $API_URL | grep "browser_download_url" | grep "gwatch_${GPU_TYPE}_" | grep "manylinux2014_x86_64.whl" | cut -d '"' -f 4 | head -n 1)

if [ -z "$WHL_URL" ]; then
    echo "Error: Could not find a matching $GPU_TYPE wheel in the latest release."
    exit 1
fi

WHL_FILE=$(basename "$WHL_URL")

# 4. Download with Progress Bar
echo "Downloading $WHL_FILE..."
curl -L --progress-bar "$WHL_URL" -o "$WHL_FILE"

# 5. Install using pip
echo "Installing $WHL_FILE..."
run_cmd python3 -m pip install "$WHL_FILE" --break-system-packages

# 6. Cleanup
if [ $? -eq 0 ]; then
    echo "Installation successful."
    rm "$WHL_FILE"
else
    echo "Installation failed. Check permissions or dependencies."
    exit 1
fi
