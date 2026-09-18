#!/usr/bin/env bash
# Local launch script for FotonVoice Engine
# Launches the compiled Rust/Tauri application binary.

set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BINARY_PATH="$DIR/target/release/fotonvoice-engine"

if [ -f "$BINARY_PATH" ]; then
    # Launch application and pass through any arguments
    exec "$BINARY_PATH" "$@"
else
    echo "Error: fotonvoice-engine binary not found."
    echo "Please build the application first by running:"
    echo "  ./install.sh"
    exit 1
fi
