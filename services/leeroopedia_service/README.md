# Leeroopedia Content Service

FastAPI service that provides authenticated API access to Leeroopedia wiki content.

## Overview

This service allows authenticated wiki users to download raw wiki content via API. Authentication uses API keys that are automatically generated when users sign up on the wiki.

## API Key Format

API keys follow the format: `lp_<user_id>_<32_hex_chars>`

Example: `lp_42_a1b2c3d4e5f67890abcdef1234567890`

Users can find their API key in wiki Preferences > Personal info.

## Quick Start

```bash
# Start the service
docker compose up -d

# Check health
curl http://localhost:8091/health

# Make authenticated request
curl -H "X-API-Key: lp_1_..." http://localhost:8091/v1/me
```

## API Endpoints

| Endpoint | Method | Auth | Rate Limit | Description |
|----------|--------|------|------------|-------------|
| `/health` | GET | No | No | Health check |
| `/v1/me` | GET | Yes | No | Current user info |
| `/v1/namespaces` | GET | Yes | No | List valid namespaces |
| `/v1/pages` | GET | Yes | No | List all pages |
| `/v1/pages?namespace={ns}` | GET | Yes | No | Filter by namespace |
| `/v1/pages/{namespace}/{title}` | GET | Yes | No | Get page content |
| `/v1/export` | GET | Yes | 10/hour | Export all wiki content |

## Authentication

All `/v1/*` endpoints require the `X-API-Key` header.

### Getting an API Key

1. Create an account at the wiki signup page
2. Go to Preferences > Personal info
3. Copy your Leeroopedia API Key

### Authentication Errors

```json
// Missing API key
{
  "error": "missing_api_key",
  "message": "X-API-Key header is required",
  "signup_url": "http://localhost:8090/index.php/Special:CreateAccount"
}

// Invalid API key
{
  "error": "invalid_api_key",
  "message": "Invalid or expired API key",
  "signup_url": "http://localhost:8090/index.php/Special:CreateAccount"
}
```

## Rate Limiting

The `/v1/export` endpoint is rate limited to 10 requests per hour per user.

When rate limited, you'll receive:

```json
{
  "error": "rate_limit_exceeded",
  "message": "Export rate limit exceeded. Maximum 10 requests per hour.",
  "retry_after_seconds": 3200
}
```

The response also includes a `Retry-After` header.

## Example Usage

### List all pages

```bash
curl -H "X-API-Key: lp_1_..." http://localhost:8091/v1/pages
```

### Filter by namespace

```bash
curl -H "X-API-Key: lp_1_..." "http://localhost:8091/v1/pages?namespace=implementation"
```

### Get specific page

```bash
# With repo prefix
curl -H "X-API-Key: lp_1_..." \
  http://localhost:8091/v1/pages/implementation/unslothai_unsloth/FastLanguageModel

# Without repo (searches all repos)
curl -H "X-API-Key: lp_1_..." \
  http://localhost:8091/v1/pages/implementation/FastLanguageModel
```

### Export all content

```bash
curl -H "X-API-Key: lp_1_..." http://localhost:8091/v1/export > wiki_export.json
```

## Configuration

Environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `API_PORT` | 8091 | Port to expose the API |
| `WIKI_DIR` | data/wikis | Wiki data directory |
| `MEDIAWIKI_URL` | http://host.docker.internal:8090 | MediaWiki API URL |
| `MW_AGENT_USER` | agent | MediaWiki bot username |
| `MW_AGENT_PASS` | (required) | MediaWiki bot password |
| `SIGNUP_URL` | http://localhost:8090/... | Signup URL for 401 |
| `EXPORT_RATE_LIMIT` | 10 | Export requests per window |
| `EXPORT_RATE_WINDOW` | 3600 | Rate limit window (seconds) |

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
WIKI_DATA_PATH=../../data/wikis \
MEDIAWIKI_URL=http://localhost:8090 \
MW_AGENT_PASS=agentpass123 \
uvicorn app.main:app --reload --port 8091

# View API docs
open http://localhost:8091/docs
```
