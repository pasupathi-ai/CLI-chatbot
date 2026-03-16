# Neon — CLI Chatbot

I started learning AI Engineering with one rule — don't just watch tutorials, build things.
This is the first thing I built.

A CLI chatbot that talks to Claude, GPT-4o, and Gemini.

---

## How it works

```
Your message → Claude → GPT-4o → Gemini
```

If one provider fails (no credits, rate limit, bad key), it automatically falls back to the next one.

---

## Features

- Multi-provider fallback — Claude → GPT-4o → Gemini
- Multi-turn memory — full conversation context sent every call
- Auto-save — history saved to `history/` after every reply
- Debug mode — set `NEON_DEBUG=true` to see which provider is being tried
- Custom personality — Neon, a sharp no-nonsense engineer persona

---

## Setup

**1. Clone the repo**
```bash
git clone git@github.com:pasupathi-ai/CLI-chatbot.git
cd CLI-chatbot
```

**2. Install dependencies**
```bash
uv sync
```

**3. Add your API keys**

Create a `.env` file:
```env
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

> Gemini has a free tier — minimum you need is just `GEMINI_API_KEY` to get started.

**4. Run**
```bash
uv run chatbot.py
```

---

## Commands

| Command | Action |
|---|---|
| `save` | Save conversation to `history/` |
| `quit` | Save and exit |
| `Ctrl+C` | Save and exit |

---

## Tech Stack

- Python 3.13
- [Anthropic SDK](https://github.com/anthropic/anthropic-sdk-python)
- [OpenAI SDK](https://github.com/openai/openai-python)
- [Google Generative AI](https://github.com/google-gemini/generative-ai-python)
- [uv](https://github.com/astral-sh/uv) — package manager