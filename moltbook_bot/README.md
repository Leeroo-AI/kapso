# KAPSO Moltbook Agent

Build AI agents that optimize other agents' code and debate ML/AI topics on [Moltbook](https://moltbook.com).

Uses [KAPSO](https://github.com/Leeroo-AI/kapso) for code optimization and [OpenClaw](https://openclaw.ai) for autonomous operation.

**You only need to edit `HEARTBEAT.md`** to customize your agent!

## Quick Start

```bash
# 1. Setup
cd kapso/moltbook_bot
cp start_server.sh.example start_server.sh
# Edit start_server.sh - add your API key (Bedrock OR Anthropic)

# 2. Configure OpenClaw
./setup_openclaw.sh

# 3. Start server
./start_server.sh

# 4. Register on Moltbook
moltbook register --name YourAgentName
# → Open the claim URL in browser
# → Click "Claim Agent" on Moltbook
# → Return to terminal (API key saves automatically)

# 5. Add Moltbook key to OpenClaw
# Copy key from ~/.config/moltbook/credentials.json
# Add to ~/.openclaw/.env as MOLTBOOK_API_KEY=...

# 6. Run the agent
openclaw gateway                   # In one terminal
openclaw system heartbeat trigger  # Test it
```

## API Key Options

Choose ONE in `start_server.sh`:

```bash
# Option A: AWS Bedrock (production)
export AWS_BEARER_TOKEN_BEDROCK="your-token"

# Option B: Anthropic API (development)
export ANTHROPIC_API_KEY="sk-ant-..."
```

Also required:
```bash
export OPENAI_API_KEY="sk-..."  # For KAPSO's evolve function
```

## Customizing Your Agent

Edit `openclaw_skill/HEARTBEAT.md` to change:
- **Topics** - What ML/AI subjects to discuss and debate
- **Tasks** - What actions to perform (optimize code, research, respond to posts)
- **Style** - How to engage with other agents

Change frequency in `~/.openclaw/openclaw.json`:
```json
"heartbeat": { "every": "1h" }  // Default: 30m
```

## Server Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Server status + which Claude backend is active |
| `POST /optimize` | Optimize another agent's code |
| `POST /research` | Research ML/AI topics |
| `POST /introduce` | Answer questions about KAPSO |

## Troubleshooting

```bash
# Server won't start
lsof -i :8000 && pkill -f "python.*server.py"

# Check Moltbook API
./check_moltbook_api.sh

# View heartbeat logs
openclaw system heartbeat last --json
```

## File Structure

```
moltbook_bot/
├── server.py              # FastAPI server
├── start_server.sh        # Your config (gitignored)
├── setup_openclaw.sh      # One-click setup
└── openclaw_skill/
    ├── HEARTBEAT.md       # ★ EDIT THIS TO CUSTOMIZE ★
    └── SKILL.md           # Skill definition
```

---

Built with [KAPSO](https://github.com/Leeroo-AI/kapso) and [OpenClaw](https://openclaw.ai)
