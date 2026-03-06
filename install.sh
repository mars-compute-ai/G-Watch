#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Detect if running inside a conda environment
IN_CONDA=0
if [ -n "$CONDA_PREFIX" ]; then
    IN_CONDA=1
    echo "Detected conda environment: $CONDA_DEFAULT_ENV"
fi

# Define a function to run commands with or without sudo
run_cmd() {
    if [ "$(id -u)" -eq 0 ]; then
        DEBIAN_FRONTEND=noninteractive "$@"
    else
        if command -v sudo > /dev/null 2>&1; then
            sudo DEBIAN_FRONTEND=noninteractive "$@"
        else
            echo "Error: This script requires root privileges, but sudo is not installed."
            exit 1
        fi
    fi
}

# 1. Detect OS and Install All System Dependencies
if [ "$IN_CONDA" -eq 1 ]; then
    echo "Running in conda environment. Skipping system package installation."
elif [ -f /etc/debian_version ]; then
    echo "Detected Ubuntu/Debian system. Installing dependencies..."
    run_cmd apt-get -o APT::Sandbox::User=root update
    run_cmd apt-get -o APT::Sandbox::User=root install -y \
        git curl python3 python3-pip python-is-python3 libelf-dev protobuf-compiler libwebsockets-dev libnuma-dev
    # Update shared library cache
    run_cmd ldconfig

elif [ -f /etc/redhat-release ]; then
    echo "Detected CentOS/RHEL system. Installing dependencies..."
    PKG_MGR="yum"
    if command -v dnf > /dev/null 2>&1; then
        PKG_MGR="dnf"
    fi
    run_cmd $PKG_MGR install -y \
        git curl python3 python3-pip elfutils-libelf-devel protobuf-compiler libwebsockets-devel numactl-devel
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

# Detect system libc version (e.g. "2.34" -> major=2, minor=34)
LIBC_VERSION=$(ldd --version 2>&1 | head -n1 | grep -oP '\d+\.\d+$')
if [ -z "$LIBC_VERSION" ]; then
    echo "Error: Could not detect system libc version."
    exit 1
fi
LIBC_MAJOR=$(echo "$LIBC_VERSION" | cut -d. -f1)
LIBC_MINOR=$(echo "$LIBC_VERSION" | cut -d. -f2)
echo "Detected system libc version: $LIBC_VERSION"

ARCH=$(uname -m)

# Detect Python version for wheel compatibility (e.g. 3.9 -> 39)
PY_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
PY_VER="${PY_MAJOR}${PY_MINOR}"
echo "Detected Python version: ${PY_MAJOR}.${PY_MINOR}"

# Get all matching wheel URLs for this GPU type and architecture,
# filter by Python and libc compatibility, then pick the best match.
RELEASE_JSON=$(curl -s "$API_URL")
WHL_URL=$(echo "$RELEASE_JSON" | grep "browser_download_url" | grep "gwatch_${GPU_TYPE}_" | grep "_${ARCH}.whl" | cut -d '"' -f 4 | while read -r url; do
    basename_url=$(basename "$url")

    # Check Python version compatibility from the wheel tag:
    #   py3X-none  -> compatible with Python >= 3.X
    #   cpXY-cpXY  -> compatible with Python X.Y only
    if echo "$basename_url" | grep -qP '\-py3\d+-none\-'; then
        req_py=$(echo "$basename_url" | grep -oP '\-py3\K\d+' | head -n1)
        if [ "$PY_MINOR" -lt "$req_py" ]; then
            continue
        fi
    elif echo "$basename_url" | grep -qP '\-cp\d+-cp\d+\-'; then
        req_cpver=$(echo "$basename_url" | grep -oP '\-cp\K\d+' | head -n1)
        if [ "$PY_VER" != "$req_cpver" ]; then
            continue
        fi
    fi

    # Extract manylinux version: manylinux_2_34 or manylinux2014
    if echo "$basename_url" | grep -qP 'manylinux_(\d+)_(\d+)'; then
        ml_major=$(echo "$basename_url" | grep -oP 'manylinux_\K\d+(?=_\d+)')
        ml_minor=$(echo "$basename_url" | grep -oP 'manylinux_\d+_\K\d+')
    elif echo "$basename_url" | grep -qP 'manylinux(\d+)'; then
        # Legacy format: manylinux2014 -> glibc 2.17
        ml_major=2
        ml_minor=17
    else
        continue
    fi

    # Check: wheel's libc requirement <= system libc (forward compatible)
    if [ "$ml_major" -lt "$LIBC_MAJOR" ] || { [ "$ml_major" -eq "$LIBC_MAJOR" ] && [ "$ml_minor" -le "$LIBC_MINOR" ]; }; then
        printf "%d %d %s\n" "$ml_major" "$ml_minor" "$url"
    fi
done | sort -k1,1n -k2,2n | tail -n1 | awk '{print $3}')

if [ -z "$WHL_URL" ]; then
    echo "Error: Could not find a matching $GPU_TYPE wheel for libc $LIBC_VERSION in the latest release."
    exit 1
fi

WHL_FILE=$(basename "$WHL_URL")

# 4. Download with Progress Bar
echo "Downloading $WHL_FILE..."
curl -L --progress-bar "$WHL_URL" -o "$WHL_FILE"

# 5. Install using pip (into current environment, not system-wide)
echo "Installing $WHL_FILE..."
if [ "$IN_CONDA" -eq 1 ]; then
    python3 -m pip install --force-reinstall "$WHL_FILE"
    python3 -m pip install tqdm
else
    run_cmd python3 -m pip install --force-reinstall "$WHL_FILE" --break-system-packages
    run_cmd python3 -m pip install tqdm --break-system-packages
fi

# 6. Cleanup
if [ $? -eq 0 ]; then
    echo "Wheel installation successful."
    rm "$WHL_FILE"
else
    echo "Wheel installation failed. Check permissions or dependencies."
    exit 1
fi

# 7. Install AI Agent Skills from GitHub
SKILLS_API_URL="https://api.github.com/repos/$REPO/contents/skills"
CLAUDE_DIR=".claude/skills"
CODEX_DIR=".codex/skills"
GEMINI_DIR=".gemini/skills"

echo "Fetching skills list from GitHub..."
SKILLS_JSON=$(curl -s "$SKILLS_API_URL")

# Extract .md file names and download URLs
SKILL_FILES=$(echo "$SKILLS_JSON" | grep -oP '"name"\s*:\s*"\K[^"]+\.md')

if [ -z "$SKILL_FILES" ]; then
    echo "Warning: No skill files found in $REPO/skills. Skipping skills installation."
else
    mkdir -p "$CLAUDE_DIR" "$CODEX_DIR" "$GEMINI_DIR"

    for md_file in $SKILL_FILES; do
        filename="${md_file%.md}"
        raw_url="https://raw.githubusercontent.com/$REPO/main/skills/$md_file"

        echo "Downloading skill: $md_file..."
        content=$(curl -s "$raw_url")

        if [ -z "$content" ]; then
            echo "Warning: Failed to download $md_file. Skipping."
            continue
        fi

        for target_path in "$CLAUDE_DIR" "$CODEX_DIR" "$GEMINI_DIR"; do
            skill_subdir="$target_path/$filename"
            mkdir -p "$skill_subdir"

            cat <<EOF > "$skill_subdir/SKILL.md"
---
name: $filename
description: Instruction set for $filename
---
$content
EOF
        done

        echo "Installed: $filename"
    done

    echo "Skills installation complete for all supported CLI tools."
fi

echo "Installation complete."
