# AI Projects MCP Server

MCP server exposing the Uganda Community Health Bot and supporting tools from the AI engineering portfolio.

## Tools

| Tool | Description |
|------|-------------|
| `ask_health_bot` | Ask the Uganda health bot a question in Luganda, Acholi, Ateso, Runyankole, or English. Emergency detection built in. |
| `route_query` | Classify a prompt by complexity and get the recommended Claude model (Haiku / Sonnet / Opus) with cost estimate. |
| `list_projects` | Browse all 20 AI projects with optional keyword filter. |

## Resources

| URI | Description |
|-----|-------------|
| `projects://all` | Markdown summary of all 20 projects |
| `health://languages` | Supported languages for the health bot |

## Setup

```bash
cd mcp-server
cp ../.env.example .env   # add ANTHROPIC_API_KEY
uv sync
```

## Add to Claude Code

```bash
claude mcp add ai-projects -- uv --directory /path/to/ai-projects/mcp-server run python -m src.server
```

Or add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ai-projects": {
      "command": "uv",
      "args": ["--directory", "/path/to/ai-projects/mcp-server", "run", "python", "-m", "src.server"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

## Run directly (for testing)

```bash
uv run mcp dev src/server.py
```
