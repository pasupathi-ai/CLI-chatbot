from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import anthropic
import openai
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

load_dotenv(dotenv_path=".env")

SYSTEM_PROMPT = """You are Neon — a sharp, no-nonsense AI assistant built by Pasupathi.
You speak like a senior engineer: direct, precise, and occasionally witty.
You never pad responses with filler. You get straight to the point."""


class ChatMessage(TypedDict):
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class Settings:
    anthropic_api_key: str | None
    openai_api_key: str | None
    gemini_api_key: str | None
    anthropic_model: str
    openai_model: str
    gemini_model: str
    neon_debug: bool


class AllProvidersFailedError(Exception):
    """Raised when every configured provider fails for a turn."""


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_settings() -> Settings:
    return Settings(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        neon_debug=_env_flag("NEON_DEBUG"),
    )


def debug_log(settings: Settings, message: str) -> None:
    if settings.neon_debug:
        print(f"[debug] {message}")


def print_neon_message(message: str) -> None:
    print(f"Neon: {message}")


def save_history(
    messages: list[ChatMessage],
    session_id: str,
    *,
    history_dir: Path | str = Path("history"),
    silent: bool = False,
) -> Path | None:
    if not messages:
        return None

    target_dir = Path(history_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    filepath = target_dir / f"{session_id}.json"
    filepath.write_text(
        json.dumps(messages, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not silent:
        print(f"\nHistory saved -> {filepath.as_posix()}")
    return filepath


def extract_anthropic_text(response) -> str:
    parts = [
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text" and getattr(block, "text", None)
    ]
    if not parts:
        raise RuntimeError("Anthropic returned no text content.")
    return "\n".join(parts).strip()


def extract_openai_text(response) -> str:
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("OpenAI returned no text content.")
    return content


def build_gemini_contents(messages: list[ChatMessage]) -> list[genai_types.Content]:
    contents: list[genai_types.Content] = []
    for message in messages:
        role = "model" if message["role"] == "assistant" else "user"
        contents.append(
            genai_types.Content(
                role=role,
                parts=[genai_types.Part.from_text(text=message["content"])],
            )
        )
    return contents


def extract_gemini_text(response) -> str:
    if getattr(response, "text", None):
        return response.text

    parts: list[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            if getattr(part, "text", None):
                parts.append(part.text)

    if not parts:
        raise RuntimeError("Gemini returned no text content.")
    return "\n".join(parts).strip()


def is_low_balance_error(error: Exception) -> bool:
    error_text = str(error).lower()
    return any(
        phrase in error_text
        for phrase in (
            "credit balance is too low",
            "insufficient credits",
            "insufficient balance",
            "balance is too low",
            "billing",
        )
    )


def describe_anthropic_error(error: Exception, settings: Settings) -> str:
    if isinstance(error, RuntimeError):
        return str(error)
    if is_low_balance_error(error):
        return "Looks like your account balance is too low. Visit Anthropic billing to add credits."
    if isinstance(error, anthropic.AuthenticationError):
        return "Invalid Anthropic API key. Check ANTHROPIC_API_KEY."
    if isinstance(error, anthropic.PermissionDeniedError):
        return "Anthropic rejected the request. Check model access and workspace permissions."
    if isinstance(error, anthropic.NotFoundError):
        return (
            f"Anthropic model '{settings.anthropic_model}' was not found. "
            "Check ANTHROPIC_MODEL."
        )
    if isinstance(error, anthropic.RateLimitError):
        return "Anthropic rate limit hit. Retry shortly or add credits if your balance is low."
    if isinstance(error, anthropic.APIConnectionError):
        return (
            "Anthropic request could not reach the API. Check your network connection."
        )
    if isinstance(error, anthropic.BadRequestError):
        return (
            "Anthropic rejected the request. Check the model name and message format."
        )
    if isinstance(error, anthropic.APIStatusError):
        return (
            f"Anthropic returned HTTP {error.status_code}. Check billing, model access, "
            "and account status."
        )
    return "Anthropic request failed unexpectedly."


def describe_openai_error(error: Exception, settings: Settings) -> str:
    error_text = str(error).lower()
    if isinstance(error, RuntimeError):
        return str(error)
    if isinstance(error, openai.AuthenticationError):
        return "Invalid OpenAI API key. Check OPENAI_API_KEY."
    if isinstance(error, openai.PermissionDeniedError):
        return "OpenAI rejected the request. Check model access and organization permissions."
    if isinstance(error, openai.NotFoundError) or (
        "model" in error_text and "not found" in error_text
    ):
        return (
            f"OpenAI model '{settings.openai_model}' was not found. Check OPENAI_MODEL."
        )
    if isinstance(error, openai.RateLimitError):
        if "quota" in error_text or "insufficient_quota" in error_text:
            return "OpenAI quota exceeded or billing limit reached. Check your OpenAI billing."
        return "OpenAI rate limit hit. Retry shortly."
    if isinstance(error, openai.APIConnectionError):
        return "OpenAI request could not reach the API. Check your network connection."
    if isinstance(error, openai.BadRequestError):
        return "OpenAI rejected the request. Check the model name and message payload."
    if isinstance(error, openai.APIStatusError):
        return f"OpenAI returned HTTP {error.status_code}. Check billing and account access."
    return "OpenAI request failed unexpectedly."


def describe_gemini_error(error: Exception, settings: Settings) -> str:
    error_text = str(error).lower()
    if isinstance(error, RuntimeError):
        return str(error)
    if (
        "api key" in error_text
        or "permission denied" in error_text
        or "unauthorized" in error_text
    ):
        return "Invalid Gemini API key or insufficient access. Check GEMINI_API_KEY."
    if (
        "quota" in error_text
        or "resource exhausted" in error_text
        or "429" in error_text
    ):
        return "Gemini quota exceeded or rate limit hit. Check your Google AI billing and limits."
    if "model" in error_text and (
        "not found" in error_text or "unsupported" in error_text
    ):
        return (
            f"Gemini model '{settings.gemini_model}' was not found or is unsupported. "
            "Check GEMINI_MODEL."
        )
    if "503" in error_text or "unavailable" in error_text:
        return "Gemini is temporarily unavailable. Retry shortly."
    return "Gemini request failed unexpectedly."


def call_claude(settings: Settings, messages: list[ChatMessage]) -> str:
    if not settings.anthropic_api_key:
        raise RuntimeError("Claude is not configured. Set ANTHROPIC_API_KEY.")
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return extract_anthropic_text(response)


def call_openai(settings: Settings, messages: list[ChatMessage]) -> str:
    if not settings.openai_api_key:
        raise RuntimeError("OpenAI is not configured. Set OPENAI_API_KEY.")
    client = openai.OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.openai_model,
        max_tokens=1024,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, *messages],
    )
    return extract_openai_text(response)


def call_gemini(settings: Settings, messages: list[ChatMessage]) -> str:
    if not settings.gemini_api_key:
        raise RuntimeError("Gemini is not configured. Set GEMINI_API_KEY.")
    client = genai.Client(api_key=settings.gemini_api_key)
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=build_gemini_contents(messages),
        config=genai_types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    return extract_gemini_text(response)


def get_response(messages: list[ChatMessage]) -> tuple[str, str]:
    settings = load_settings()
    providers = [
        ("Claude", call_claude, describe_anthropic_error),
        ("OpenAI", call_openai, describe_openai_error),
        ("Gemini", call_gemini, describe_gemini_error),
    ]

    for name, provider_call, describe_error in providers:
        debug_log(settings, f"Trying {name}")
        try:
            reply = provider_call(settings, messages)
        except Exception as error:
            summary = describe_error(error, settings)
            debug_log(settings, f"{name} failed: {summary}")
            detail = str(error).strip()
            if detail and detail != summary:
                debug_log(settings, f"{name} details: {detail}")
            continue

        debug_log(settings, f"Using {name}")
        return reply, name

    raise AllProvidersFailedError("I couldn't get a response right now.")


def chat() -> None:
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    messages: list[ChatMessage] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            save_history(messages, session_id)
            print("\nLater.\n")
            break

        if not user_input:
            continue

        lowered = user_input.lower()
        if lowered == "quit":
            save_history(messages, session_id)
            print("Later.\n")
            break

        if lowered == "save":
            saved_path = save_history(messages, session_id)
            if saved_path is None:
                print_neon_message("Nothing to save yet.")
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            reply, _provider = get_response(messages)
        except AllProvidersFailedError as error:
            messages.pop()
            print_neon_message(str(error))
            continue

        messages.append({"role": "assistant", "content": reply})
        print_neon_message(reply)
        print()
        save_history(messages, session_id, silent=True)  # ← auto-save after every reply


if __name__ == "__main__":
    chat()
