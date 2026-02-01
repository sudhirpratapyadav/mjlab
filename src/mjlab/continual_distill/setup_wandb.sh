#!/bin/bash
# Setup script for project-specific wandb credentials

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

echo "=== WandB Project-Specific Setup ==="
echo ""
echo "This will create a .env file with your wandb credentials."
echo "This file is project-specific and won't affect other users or projects."
echo ""

if [ -f "$ENV_FILE" ]; then
    echo "WARNING: .env file already exists at $ENV_FILE"
    read -p "Do you want to overwrite it? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled."
        exit 0
    fi
fi

echo "1. Get your API key from: https://wandb.ai/authorize"
echo "2. Copy your API key"
echo ""
read -p "Paste your WandB API key here: " API_KEY

if [ -z "$API_KEY" ]; then
    echo "Error: API key cannot be empty"
    exit 1
fi

# Create .env file
cat > "$ENV_FILE" <<EOF
# WandB Configuration - Project-specific credentials
# Generated on $(date)

WANDB_API_KEY=$API_KEY
WANDB_ENTITY=domimagi-iitj
WANDB_PROJECT=continual_rl_mjlab
EOF

echo ""
echo "✓ Success! Created $ENV_FILE"
echo ""
echo "Your training scripts will now use your wandb account automatically."
echo "Other users won't be affected by your credentials."
echo ""
echo "IMPORTANT: Never commit the .env file to git (it's already in .gitignore)"
