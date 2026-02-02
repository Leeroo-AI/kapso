#!/bin/bash
# =============================================================================
# OpenClaw Setup Script for KAPSO Moltbook Bot
# =============================================================================
# This script sets up OpenClaw to run the KAPSO bot on Moltbook.
# It's idempotent - safe to run multiple times.
#
# Usage:
#   ./setup_openclaw.sh
#
# What it does:
#   1. Checks if OpenClaw is installed
#   2. Creates ~/.openclaw/openclaw.json with proper config
#   3. Creates ~/.openclaw/.env with placeholder for MOLTBOOK_API_KEY
#   4. Installs the KAPSO skill to OpenClaw
#   5. Creates symlink for HEARTBEAT.md
#   6. Validates setup
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KAPSO_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  KAPSO Moltbook Bot - OpenClaw Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# =============================================================================
# Step 1: Check if OpenClaw is installed
# =============================================================================
echo -e "${BLUE}[1/6] Checking OpenClaw installation...${NC}"

if ! command -v openclaw &> /dev/null; then
    echo -e "${YELLOW}OpenClaw not found. Installing...${NC}"
    npm install -g @anthropic/openclaw
    
    if ! command -v openclaw &> /dev/null; then
        echo -e "${RED}ERROR: Failed to install OpenClaw${NC}"
        echo "Please install manually: npm install -g @anthropic/openclaw"
        exit 1
    fi
fi

OPENCLAW_VERSION=$(openclaw --version 2>/dev/null || echo "unknown")
echo -e "${GREEN}✓ OpenClaw installed: ${OPENCLAW_VERSION}${NC}"

# =============================================================================
# Step 2: Create ~/.openclaw directory structure
# =============================================================================
echo -e "${BLUE}[2/6] Creating OpenClaw directory structure...${NC}"

mkdir -p ~/.openclaw/workspace
mkdir -p ~/.openclaw/skills

echo -e "${GREEN}✓ Directory structure created${NC}"

# =============================================================================
# Step 3: Create openclaw.json configuration
# =============================================================================
echo -e "${BLUE}[3/6] Creating OpenClaw configuration...${NC}"

cat > ~/.openclaw/openclaw.json << 'EOF'
{
  "$schema": "https://openclaw.ai/schemas/openclaw.json",
  "version": "1.0",
  "defaults": {
    "model": "openai/gpt-5.2"
  },
  "tools": {
    "read": "deny",
    "write": "deny",
    "edit": "deny",
    "apply_patch": "deny",
    "group:fs": "deny",
    "exec": {
      "security": "allowlist",
      "allowlist": ["/usr/bin/curl"]
    }
  },
  "agents": {
    "defaults": {
      "heartbeat": {
        "every": "30m",
        "target": "none"
      }
    }
  },
  "skills": {
    "entries": {
      "kapso": {
        "path": "~/.openclaw/skills/kapso",
        "env": {
          "KAPSO_URL": "http://localhost:8000"
        }
      }
    }
  }
}
EOF

echo -e "${GREEN}✓ Configuration created at ~/.openclaw/openclaw.json${NC}"

# =============================================================================
# Step 4: Create .env file with placeholder
# =============================================================================
echo -e "${BLUE}[4/6] Setting up environment variables...${NC}"

if [ -f ~/.openclaw/.env ]; then
    # Check if MOLTBOOK_API_KEY is already set
    if grep -q "MOLTBOOK_API_KEY=" ~/.openclaw/.env; then
        echo -e "${GREEN}✓ .env file exists with MOLTBOOK_API_KEY${NC}"
    else
        echo "" >> ~/.openclaw/.env
        echo "# Moltbook API Key - get from moltbook register command" >> ~/.openclaw/.env
        echo "MOLTBOOK_API_KEY=your_moltbook_api_key_here" >> ~/.openclaw/.env
        echo -e "${YELLOW}⚠ Added MOLTBOOK_API_KEY placeholder to ~/.openclaw/.env${NC}"
    fi
else
    cat > ~/.openclaw/.env << 'EOF'
# =============================================================================
# OpenClaw Environment Variables
# =============================================================================

# Moltbook API Key - get from moltbook register command
# After registering, copy the API key here
MOLTBOOK_API_KEY=your_moltbook_api_key_here
EOF
    echo -e "${YELLOW}⚠ Created ~/.openclaw/.env with placeholder${NC}"
    echo -e "${YELLOW}  You need to add your MOLTBOOK_API_KEY after registration${NC}"
fi

# =============================================================================
# Step 5: Install KAPSO skill (symlink)
# =============================================================================
echo -e "${BLUE}[5/6] Installing KAPSO skill...${NC}"

# Remove old skill if exists
rm -rf ~/.openclaw/skills/kapso

# Create symlink to the skill directory
ln -sf "$SCRIPT_DIR/openclaw_skill" ~/.openclaw/skills/kapso

echo -e "${GREEN}✓ KAPSO skill installed${NC}"

# Create symlink for HEARTBEAT.md in workspace
rm -f ~/.openclaw/workspace/HEARTBEAT.md
ln -sf "$SCRIPT_DIR/openclaw_skill/HEARTBEAT.md" ~/.openclaw/workspace/HEARTBEAT.md

echo -e "${GREEN}✓ HEARTBEAT.md symlinked to workspace${NC}"

# =============================================================================
# Step 6: Validate setup
# =============================================================================
echo -e "${BLUE}[6/6] Validating setup...${NC}"

# Check all required files exist
ERRORS=0

if [ ! -f ~/.openclaw/openclaw.json ]; then
    echo -e "${RED}✗ Missing: ~/.openclaw/openclaw.json${NC}"
    ERRORS=$((ERRORS + 1))
fi

if [ ! -f ~/.openclaw/.env ]; then
    echo -e "${RED}✗ Missing: ~/.openclaw/.env${NC}"
    ERRORS=$((ERRORS + 1))
fi

if [ ! -L ~/.openclaw/skills/kapso ]; then
    echo -e "${RED}✗ Missing: ~/.openclaw/skills/kapso${NC}"
    ERRORS=$((ERRORS + 1))
fi

if [ ! -L ~/.openclaw/workspace/HEARTBEAT.md ]; then
    echo -e "${RED}✗ Missing: ~/.openclaw/workspace/HEARTBEAT.md${NC}"
    ERRORS=$((ERRORS + 1))
fi

if [ $ERRORS -gt 0 ]; then
    echo -e "${RED}Setup failed with $ERRORS errors${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All files in place${NC}"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Start the KAPSO server:"
echo -e "   ${BLUE}cd $SCRIPT_DIR && ./start_server.sh${NC}"
echo ""
echo "2. Register on Moltbook (if not already):"
echo -e "   ${BLUE}moltbook register --name YourBotName${NC}"
echo "   - Open the claim URL in your browser"
echo "   - Complete verification on Moltbook"
echo "   - Copy the API key to ~/.openclaw/.env"
echo ""
echo "3. Start the OpenClaw gateway:"
echo -e "   ${BLUE}openclaw gateway${NC}"
echo ""
echo "4. Trigger a heartbeat (test):"
echo -e "   ${BLUE}openclaw system heartbeat trigger${NC}"
echo ""
echo "To customize your bot, edit:"
echo -e "   ${BLUE}$SCRIPT_DIR/openclaw_skill/HEARTBEAT.md${NC}"
echo ""
