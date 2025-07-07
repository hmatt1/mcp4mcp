#!/bin/bash
# One-click installer for mcp4mcp

set -e

REPO_URL="https://github.com/hmatt1/mcp4mcp"
INSTALL_DIR="$HOME/.local/bin"

echo "🚀 Installing mcp4mcp..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Install mcp4mcp
echo "📦 Installing mcp4mcp..."
uv pip install git+${REPO_URL}.git

# Create symlink for easy access
mkdir -p $INSTALL_DIR
ln -sf $(which python3) $INSTALL_DIR/mcp4mcp

echo "✅ mcp4mcp installed successfully!"
echo ""
echo "🎉 You can now run:"
echo "  mcp4mcp          - Show help"
echo "  mcp4mcp server   - Start the MCP server"
echo "  mcp4mcp demo     - Run the demo"
echo ""
echo "📚 Documentation: ${REPO_URL}"
