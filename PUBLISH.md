# JPStock Agent — Publish & Marketplace Guide

## Step 1: Publish to PyPI

### First time setup

```bash
# Create PyPI account at https://pypi.org/account/register/
# Create API token at https://pypi.org/manage/account/token/

# Install tools
pip install build twine
```

### Build & upload

```bash
# Build
python -m build

# Check
twine check dist/*

# Upload to TestPyPI first (optional)
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*
```

### Verify

```bash
pip install jpstock-agent
jpstock-agent --help
jpstock-agent history 7203
```

---

## Step 2: Test with MCP Inspector

```bash
# Install MCP Inspector
npx @modelcontextprotocol/inspector jpstock-agent serve

# Open browser at http://localhost:5173
# Verify:
#   - All 106 tools appear in the tool list
#   - Test stock_history with symbol "7203"
#   - Test ta_rsi with symbol "7203"
#   - Test company_overview with symbol "6758"
#   - Test auth_tiers (no params needed)
```

---

## Step 3: Submit to MCPize

**URL**: https://mcpize.com/developers

### Requirements
- PyPI package published
- GitHub repo public
- README with usage instructions

### Pricing setup
| Tier | Price | Revenue (85%) |
|------|-------|---------------|
| Free | $0/mo | $0 |
| Pro | $9/mo | $7.65/mo |
| Enterprise | $19/mo | $16.15/mo |

### Submit info
- **Name**: JPStock Agent
- **Description**: MCP server for Japanese & Vietnamese stock market data — 106 AI tools for technical analysis, backtesting, portfolio optimization, ML predictions, options, and more.
- **Category**: Finance / Data
- **Tags**: stock, japan, vietnam, technical-analysis, backtesting, portfolio, ml, options, mcp
- **Install command**: `pip install jpstock-agent`
- **Run command**: `jpstock-agent serve`
- **Pricing model**: Monthly subscription
- **Tiers**: Free ($0), Pro ($9), Enterprise ($19)

---

## Step 4: Submit to MCP Marketplace

**URL**: https://mcp-marketplace.io/docs

### Requirements
- Pass MCP Inspector test
- Security scan (automatic)
- README with clear usage docs

### Submit info
Same as MCPize above. Review usually takes < 24 hours.

---

## Step 5: Claude Code Plugin Marketplace (Optional)

Create a `marketplace.json` for distribution as a Claude Code plugin:

```json
{
  "name": "jpstock-marketplace",
  "description": "Japanese & Vietnamese stock market tools for Claude",
  "plugins": [
    {
      "name": "jpstock-agent",
      "description": "106 MCP tools for stock data, TA, backtesting, ML, options, portfolio",
      "source": {
        "type": "git",
        "url": "https://github.com/fvn-manhtd/jp-stock-agent.git"
      }
    }
  ]
}
```

Host on GitHub and share with:
```
/plugin marketplace add https://raw.githubusercontent.com/fvn-manhtd/jp-stock-agent/main/marketplace.json
```

---

## Docker Deployment (for HTTP/SSE customers)

```bash
# Build
docker build -t jpstock-agent:0.2.0 .

# Run with auth
docker-compose -f docker-compose.yml -f docker-compose.auth.yml up -d

# Generate keys for customers
docker exec jpstock-agent jpstock-agent key-generate --tier pro --owner "customer@example.com"
```

---

## Revenue Tracking

```bash
# Check daily usage
jpstock-agent usage-daily

# Revenue estimate
jpstock-agent usage-revenue

# Per-customer usage
jpstock-agent usage-key jpsk_pro_xxxx
```
