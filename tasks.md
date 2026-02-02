# Moltbook Bot Refactoring Tasks

## Overview
Reorganize the Moltbook bot code into a clean, reusable structure that allows users to easily create their own AI/ML expert agents by just modifying the HEARTBEAT.md file.

---

## Task 1: Create Directory Structure
- [x] Create `/home/ubuntu/kapso/moltbook_bot/` directory
- [x] Move these files into it:
  - `kapso_server.py` → `moltbook_bot/server.py`
  - `start_server.sh` → `moltbook_bot/start_server.sh`
  - `check_moltbook_api.sh` → `moltbook_bot/check_moltbook_api.sh`
  - `openclaw_skill/SKILL.md` → `moltbook_bot/openclaw_skill/SKILL.md`
  - `openclaw_skill/HEARTBEAT.md` → `moltbook_bot/openclaw_skill/HEARTBEAT.md`
- [x] Update symlink at `~/.openclaw/workspace/HEARTBEAT.md` to point to new location
- [x] Update any hardcoded paths in the moved files

---

## Task 2: Create OpenClaw Setup Script
- [x] Create `moltbook_bot/setup_openclaw.sh` that:
  - Checks if OpenClaw is installed, installs if not
  - Creates `~/.openclaw/openclaw.json` with proper config (model, tools, heartbeat interval)
  - Creates `~/.openclaw/.env` with placeholder for `MOLTBOOK_API_KEY`
  - Installs the KAPSO skill to OpenClaw
  - Creates symlink for HEARTBEAT.md
  - Validates setup with a test command
- [x] Make it idempotent (safe to run multiple times)

---

## Task 3: Update Server for Flexible Claude Code Backend
- [x] Modify `server.py` to detect API key type:
  ```python
  # Priority order:
  # 1. AWS_BEARER_TOKEN_BEDROCK → use Bedrock
  # 2. ANTHROPIC_API_KEY → use Anthropic API directly
  # 3. Neither → error with helpful message
  ```
- [x] Update `/introduce` endpoint to use detected backend
- [x] Update `/research` endpoint if it uses Claude Code
- [x] Update `/optimize` endpoint if it uses Claude Code
- [x] Add `/health` endpoint info about which backend is active
- [x] Update `start_server.sh.example` with both options documented

---

## Task 4: Create Template start_server.sh
- [x] Create `moltbook_bot/start_server.sh.example` (template, not actual secrets)
- [x] Document both authentication options:
  ```bash
  # Option 1: AWS Bedrock (recommended for production)
  # export AWS_BEARER_TOKEN_BEDROCK="your-token"
  
  # Option 2: Anthropic API (simpler setup)
  # export ANTHROPIC_API_KEY="your-key"
  ```
- [x] Add `.gitignore` entry for `start_server.sh` (actual file with secrets)

---

## Task 5: Write README Guide
- [x] Create `moltbook_bot/README.md` with sections:

### 5.1 Introduction
- What is this? (AI/ML expert bot for Moltbook)
- What can it do? (research, optimize code, answer questions)
- Architecture diagram (HEARTBEAT → OpenClaw → Server → Claude Code)

### 5.2 Quick Start (5 minutes)
1. Clone repo
2. Copy `start_server.sh.example` → `start_server.sh`
3. Add your API key (Bedrock or Anthropic)
4. Run `./setup_openclaw.sh`
5. Run `./start_server.sh`
6. Register on Moltbook:
   - Run `moltbook register --name YourBotName`
   - **IMPORTANT**: Open the claim URL in your browser
   - Complete verification on Moltbook website
   - Come back to terminal - API key will be saved automatically
7. Add Moltbook API key to `~/.openclaw/.env`
8. Start heartbeat: `openclaw system heartbeat trigger`

### 5.3 Customizing Your Bot
- **HEARTBEAT.md** - The only file you need to edit!
- Explain each section of HEARTBEAT.md
- Examples of customization:
  - Change topics (from AI/ML to crypto, gaming, etc.)
  - Change posting frequency
  - Add new tasks
  - Modify response style

### 5.4 API Key Options
- **AWS Bedrock** (production): How to get token, benefits
- **Anthropic API** (development): How to get key, simpler setup

### 5.5 Server Endpoints Reference
- `/health` - Check status
- `/optimize` - Code optimization
- `/research` - Web research
- `/introduce` - Answer questions about your bot

### 5.6 Moltbook Registration
- How to register: `moltbook register --name YourBotName`
- **Claim URL workflow**:
  1. Command outputs a claim URL
  2. Open URL in browser (must be logged into Moltbook)
  3. Click "Claim Agent" on the website
  4. Return to terminal - it auto-detects claim and saves API key
- Where API key is stored: `~/.config/moltbook/credentials.json`
- How to add to OpenClaw: copy to `~/.openclaw/.env` as `MOLTBOOK_API_KEY=...`
- Rate limits: 1 registration per day per IP

### 5.7 Troubleshooting
- Common errors and fixes
- How to check logs
- How to test endpoints manually

### 5.8 Advanced: Building Your Own Expert
- Fork this template
- Replace KAPSO knowledge with your domain
- Customize server endpoints
- Deploy to cloud

---

## Task 6: Update .gitignore
- [x] Add entries:
  ```
  moltbook_bot/start_server.sh
  *.env
  ```
- [x] Ensure secrets are never committed

---

## Task 7: Test the Setup
- [x] Test fresh setup with `setup_openclaw.sh`
- [x] Test server with Anthropic API key
- [x] Test server with Bedrock token
- [x] Test heartbeat trigger
- [x] Verify HEARTBEAT.md symlink works

---

## Task 8: Cleanup
- [x] Remove old files from root after move:
  - `/home/ubuntu/kapso/kapso_server.py`
  - `/home/ubuntu/kapso/start_server.sh`
  - `/home/ubuntu/kapso/check_moltbook_api.sh`
  - `/home/ubuntu/kapso/openclaw_skill/`
- [x] Update any imports or references in main kapso codebase

---

## File Structure After Completion

```
kapso/
├── moltbook_bot/
│   ├── README.md                    # User guide
│   ├── server.py                    # FastAPI server (was kapso_server.py)
│   ├── start_server.sh.example      # Template with placeholders
│   ├── start_server.sh              # Actual file with secrets (gitignored)
│   ├── setup_openclaw.sh            # One-click OpenClaw setup
│   ├── check_moltbook_api.sh        # API health checker
│   └── openclaw_skill/
│       ├── SKILL.md                 # OpenClaw skill definition
│       └── HEARTBEAT.md             # THE FILE USERS CUSTOMIZE
├── src/                             # Main kapso library (unchanged)
├── docs/                            # Documentation (unchanged)
└── ...
```

---

## Notes
- Users only need to edit `HEARTBEAT.md` to customize their bot
- `setup_openclaw.sh` handles all the complex OpenClaw configuration
- Server auto-detects Bedrock vs Anthropic based on environment variables
- Keep backward compatibility with existing kapso library

---

## ✅ ALL TASKS COMPLETED
