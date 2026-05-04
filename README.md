# mini-pi

A minimal coding agent in Python, inspired by [Pi](https://pi.dev).

## Architecture

```
mini_pi/
├── config.py      # Configuration (API key, model, etc.)
├── tools.py       # Core tools: bash, read, write, edit
├── agent.py       # Agent loop: LLM call → tool execution → repeat
├── session.py     # Session management (JSONL persistence)
├── system_prompt  # System prompt builder
│   .py
└── main.py        # REPL entry point
```

## Quick Start

```bash
# Install
pip install -e .

# Set API key
export OPENAI_API_KEY=sk-xxx
# Or use a custom OpenAI-compatible endpoint:
export OPENAI_BASE_URL=http://your-litellm:4000/v1

# Run
mini-pi
```

## What It Does

1. Reads your message
2. Sends to LLM with system prompt + conversation history
3. If LLM wants to call a tool → execute it → feed result back
4. Repeat until LLM gives a final text response
5. Save session as JSONL for continuity

## Tools

| Tool  | Description              |
|-------|--------------------------|
| bash  | Run shell commands       |
| read  | Read file contents       |
| write | Create/overwrite files   |
| edit  | Find & replace in files  |

## Philosophy

Like Pi, this is **minimal by design**:
- 4 core tools, no MCP
- No sub-agents, no plan mode
- Simple system prompt you can customize
- Session persistence in JSONL

Add features by extending `tools.py` or the prompt — not by forking.
