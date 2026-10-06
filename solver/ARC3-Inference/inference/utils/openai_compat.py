"""Helpers for provider-specific OpenAI-compatible requests."""
from __future__ import annotations

import json

from typing import Any


def normalize_provider(value: str | None) -> str:
    provider = str(value or "").strip().lower()
    if provider in {"", "openai", "openai-compatible", "compat"}:
        return "vllm"
    if provider in {"openrouter", "router"}:
        return "openrouter"
    return provider


def build_headers(
    *,
    provider: str,
    api_key: str,
    referer: str = "",
    title: str = "",
) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    normalized = normalize_provider(provider)
    if normalized == "openrouter":
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title
    return headers


def build_chat_payload(
    *,
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int | None,
    temperature: float,
    top_p: float,
    top_k: int,
    thinking: bool,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    seed: int | None = None,
    priority: int | None = None,
    stream: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": bool(stream),
        "temperature": temperature,
        "top_p": top_p,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if priority is not None:
        # scheduling hint, ignored by servers that do not implement it
        payload["priority"] = int(priority)
    if tools:
        payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

    normalized = normalize_provider(provider)
    if stream:
        payload["stream_options"] = {"include_usage": True}
    if normalized == "openrouter":
        import os
        if not payload["stream"] and os.environ.get(
            "OPENROUTER_STREAM", ""
        ).strip().lower() not in ("0", "false", "no", "off"):
            payload["stream"] = True
        if payload["stream"]:
            payload["stream_options"] = {"include_usage": True}
        payload["reasoning"] = {"enabled": bool(thinking)}
        _order = os.environ.get("OPENROUTER_PROVIDER_ORDER", "").strip()
        if _order:
            payload["provider"] = {
                "order": [s.strip() for s in _order.split(",") if s.strip()],
                "allow_fallbacks": os.environ.get(
                    "OPENROUTER_ALLOW_FALLBACKS", ""
                ).strip().lower() in ("1", "true", "yes"),
            }
    if normalized == "vllm":
        if top_k > 0:
            payload["top_k"] = top_k
        payload["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
        if seed is not None and seed >= 0:
            payload["seed"] = seed

    return payload


def assemble_streamed_chat_response(lines) -> dict[str, Any]:
    """Consume an SSE chat-completion stream and rebuild the non-streaming
    response shape: {"choices": [{"message", "finish_reason"}], "usage"}.

    Tolerates keep-alive comments (": ..."), event: lines, usage-only chunks,
    and mid-stream error events. If an error arrives and no assistant content
    was produced, returns {"error": ..., "choices": []} so callers surface it.
    """
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls_by_index: dict[int, dict[str, Any]] = {}
    finish_reason = ""
    usage: dict[str, Any] | None = None
    error: Any = None
    provider_name = [""]

    for raw_line in lines:
        if raw_line is None:
            continue
        line = raw_line.decode("utf-8", "replace") if isinstance(raw_line, bytes) else str(raw_line)
        line = line.strip()
        if not line or line.startswith(":") or line.startswith("event:"):
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        if line == "[DONE]":
            break
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(chunk, dict):
            continue
        if chunk.get("error"):
            error = chunk["error"]
            continue
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        if chunk.get("provider") and not provider_name[0]:
            provider_name[0] = str(chunk["provider"])
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            if choice.get("finish_reason"):
                finish_reason = str(choice["finish_reason"])
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                continue
            if isinstance(delta.get("content"), str):
                content_parts.append(delta["content"])
            if isinstance(delta.get("reasoning"), str):
                reasoning_parts.append(delta["reasoning"])
            elif isinstance(delta.get("reasoning_content"), str):
                reasoning_parts.append(delta["reasoning_content"])
            elif isinstance(delta.get("reasoning_details"), list):
                for detail in delta["reasoning_details"]:
                    if isinstance(detail, dict) and isinstance(detail.get("text"), str):
                        reasoning_parts.append(detail["text"])
            for fragment in delta.get("tool_calls") or []:
                if not isinstance(fragment, dict):
                    continue
                index = int(fragment.get("index", 0) or 0)
                entry = tool_calls_by_index.setdefault(
                    index,
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if fragment.get("id"):
                    entry["id"] = str(fragment["id"])
                if fragment.get("type"):
                    entry["type"] = str(fragment["type"])
                function = fragment.get("function") or {}
                if isinstance(function, dict):
                    if function.get("name"):
                        entry["function"]["name"] = str(function["name"])
                    if isinstance(function.get("arguments"), str):
                        entry["function"]["arguments"] += function["arguments"]

    produced = bool(content_parts or reasoning_parts or tool_calls_by_index)
    if error is not None and not produced:
        return {"error": error, "choices": []}

    message: dict[str, Any] = {"role": "assistant", "content": "".join(content_parts)}
    if reasoning_parts:
        message["reasoning"] = "".join(reasoning_parts)
    if tool_calls_by_index:
        message["tool_calls"] = [tool_calls_by_index[i] for i in sorted(tool_calls_by_index)]
    result: dict[str, Any] = {
        "choices": [{"message": message, "finish_reason": finish_reason}],
    }
    if usage is not None:
        result["usage"] = usage
    if error is not None:
        result["error"] = error
    if provider_name[0]:
        result["provider"] = provider_name[0]
    return result
