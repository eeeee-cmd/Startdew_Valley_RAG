"""
llm.py — LLM client for qwen3-30b-a3b-fp8 with reasoning enabled.
         Uses the OpenAI client pattern from class labs.

Configuration via environment variables (or pass directly to LLMClient):
    LLM_BASE_URL   — OpenAI-compatible base URL
                     e.g. "https://rsm-8430-lab2.bjlkeng.io/v1"  (lab server)
                          "http://localhost:8080/v1"              (local vLLM)
    LLM_API_KEY    — API key / student ID (e.g. "A12345678")
    LLM_MODEL      — model name (default: qwen3-30b-a3b-fp8)
    LLM_REASONING  — enable chain-of-thought (default: true)

Qwen3 reasoning
---------------
When reasoning is enabled the model returns a <think>...</think> block
before its final answer. Depending on your server version this arrives as:
  (a) A list of content blocks — one type="thinking", one type="text"  (vLLM ≥ 0.8.4)
  (b) A plain string with <think>…</think> tags embedded inline
Both are handled transparently; callers always receive a clean LLMResponse.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

# ── Config (matches lab OpenAI client pattern) ────────────────────────────────

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://rsm-8430-finalproject.bjlkeng.io/v1")
LLM_API_KEY   = os.getenv("LLM_API_KEY",   "")         # set to your STUDENT_ID
LLM_MODEL     = os.getenv("LLM_MODEL",     "qwen3-30b-a3b-fp8")
LLM_REASONING = os.getenv("LLM_REASONING", "true").lower() != "false"

MAX_TOKENS            = 2048
MAX_REASONING_TOKENS  = 1024   # thinking budget; cap keeps latency reasonable


# ── Response dataclass ────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    answer:        str
    reasoning:     Optional[str]   # chain-of-thought content (None if reasoning off)
    model:         str
    input_tokens:  int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ── Client ─────────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Wrapper around OpenAI client pointed at the lab (or local vLLM) endpoint.

    Mirrors the lab pattern:
        client = OpenAI(base_url="...", api_key=STUDENT_ID)
    but adds reasoning support for Qwen3 and clean response parsing.
    """

    def __init__(
        self,
        base_url:  str  = LLM_BASE_URL,
        api_key:   str  = LLM_API_KEY,
        model:     str  = LLM_MODEL,
        reasoning: bool = LLM_REASONING,
    ):
        if not api_key:
            raise ValueError(
                "LLM_API_KEY is not set. "
                "Export it as an env var or pass api_key= directly.\n"
                "  export LLM_API_KEY=<your-student-id>"
            )
        self.model     = model
        self.reasoning = reasoning
        self._client   = OpenAI(base_url=base_url, api_key=api_key)

    # ── Public ────────────────────────────────────────────────────────────────

    def complete(
        self,
        messages:    list[dict],
        system:      Optional[str] = None,
        max_tokens:  int   = MAX_TOKENS,
        temperature: float = 0.6,        # Qwen3 recommended for reasoning tasks
    ) -> LLMResponse:
        """
        Send a chat completion.

        Args:
            messages:    List of {"role": "user"/"assistant", "content": "..."}.
                         Do NOT include a system message — use the system param.
            system:      Optional system prompt (prepended as a system message).
            max_tokens:  Total token budget for the response (reasoning + answer).
            temperature: Sampling temperature.
        """
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        # Reasoning extra_body — standard across vLLM / SGLang / Together / Fireworks
        extra: dict = {}
        if self.reasoning:
            extra["enable_thinking"]        = True
            extra["thinking_budget_tokens"] = MAX_REASONING_TOKENS

        response = self._client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra or None,
        )

        return self._parse(response)

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse(self, response) -> LLMResponse:
        choice  = response.choices[0]
        message = choice.message
        reasoning: Optional[str] = None
        answer: str              = ""

        # Format A: structured content blocks (vLLM ≥ 0.8.4)
        if isinstance(message.content, list):
            text_parts: list[str] = []
            for block in message.content:
                btype = getattr(block, "type", None)
                btext = getattr(block, "text", "") or ""
                if btype in ("thinking", "reasoning"):
                    reasoning = btext.strip()
                else:
                    text_parts.append(btext)
            answer = "".join(text_parts).strip()

        # Format B: plain string with optional <think>…</think> tags
        else:
            raw = message.content or ""
            reasoning, answer = _split_think_tags(raw)

        usage = response.usage
        return LLMResponse(
            answer=answer,
            reasoning=reasoning,
            model=response.model,
            input_tokens=  usage.prompt_tokens     if usage else 0,
            output_tokens= usage.completion_tokens if usage else 0,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)

def _split_think_tags(text: str) -> tuple[Optional[str], str]:
    match = _THINK_RE.search(text)
    if not match:
        return None, text.strip()
    return match.group(1).strip(), _THINK_RE.sub("", text).strip()


# ── Module-level singleton ────────────────────────────────────────────────────

_default_client: Optional[LLMClient] = None

def get_llm_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client
