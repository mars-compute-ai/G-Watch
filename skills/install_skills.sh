#!/bin/bash

# Configuration
SOURCE_DIR="skills"
CLAUDE_DIR=".claude/skills"
CODEX_DIR=".codex/skills"
GEMINI_DIR=".gemini/skills"

# Ensure source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' not found."
    exit 1
fi

# Function to sync skills to a target directory
sync_skills() {
    local target_path=$1
    local tool_name=$2

    echo "Syncing skills for $tool_name..."
    
    for file in "$SOURCE_DIR"/*.md; do
        if [ -f "$file" ]; then
            local filename=$(basename "$file" .md)
            local skill_subdir="$target_path/$filename"
            
            mkdir -p "$skill_subdir"
            
            # Standard SKILL.md format with frontmatter for AI agents
            cat <<EOF > "$skill_subdir/SKILL.md"
---
name: $filename
description: Instruction set for $filename
---
$(cat "$file")
EOF
            echo "Installed: $filename to $skill_subdir"
        fi
    done
}

# Create directories and sync
mkdir -p "$CLAUDE_DIR" "$CODEX_DIR" "$GEMINI_DIR"

sync_skills "$CLAUDE_DIR" "Claude Code"
sync_skills "$CODEX_DIR" "Codex"
sync_skills "$GEMINI_DIR" "Gemini CLI"

echo "Installation complete for all supported CLI tools."
