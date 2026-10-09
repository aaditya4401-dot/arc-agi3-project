"""Direct OpenAI-compatible tool-calling analyzer for ARC puzzle runs."""
from __future__ import annotations

import base64
import heapq
import threading
import json
import logging
import os
import random
import re
import time
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse, urlunparse

import requests

from inference.agent.noop_repeat_guard import NoopRepeatGuard, action_signature
from inference.agent.priority_scheduler import (
    ProgressPace, PrioritySnapshot, parse_total_levels, priority_value,
)
from inference.utils.animation import (
    _transient_cells as _animation_transient_cells,
    build_animation_frames,
    build_animation_view,
    describe_animation,
)
from inference.agent.action_names import reset_exposed, to_engine_action, to_model_action, undo_exposure_mode
from inference.utils.retained_functions import function_signature
from inference.agent.prompts import (
    apply_level_transfer_system_guidance,
    LEVEL_START_USER_PROMPT,
    COMPACT_TOOL_SESSION_ADDENDUM,
    PERSISTENT_FUNCTIONS_LINE,
    EPHEMERAL_FUNCTIONS_LINE,
    SUMMARY_REQUEST_PROMPT,
    PREFER_TOOL_CALLS_LINE,
    GAME_OVERVIEW_ADDENDUM,
    WORLD_MODEL_ADDENDUM,
    WORLD_MODEL_FREE_ADDENDUM,
    FRAME_DIFF_HINT_ADDENDUM,
    STEP_VERIFICATION_ADDENDUM,
    PYTHON_ADDENDUM_HEAD,
    PYTHON_ADDENDUM_TAIL,
    GAMEPLAY_CHANGED_ADDENDUM,
    STRUCTURED_RUNTIME_STATE_ADDENDUM,
    ANIMATION_ADDENDUM,
    ANIMATION_ADDENDUM_TIMELINE,
    MULTIMODAL_CONTEXT_ADDENDUM,
    TOOL_CALL_FORMAT_GUIDANCE,
    VISUAL_GAME_ADDENDUM,
    ACTION_INFO_ADDENDUM,
    UNDO_INFO_ADDENDUM,
    RESET_INFO_ADDENDUM,
)

from inference.agent.vision_context import (
    animation_composite_part,
    animation_death_image_mode,
    animation_image_mode,
    animation_peak_part,
    current_grid_image_enabled,
    current_grid_image_part,
)

from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.agent.runtime_state import Frame, HistoryEntry, RUNTIME_STATE_FILENAME, load_runtime_state
from inference.utils.openai_compat import (
    assemble_streamed_chat_response,
    build_chat_payload,
    build_headers,
)

log = logging.getLogger(__name__)

_LOCAL_ANALYZER_MODEL_ID = os.environ.get("LOCAL_ANALYZER_MODEL_ID", "")
_LOCAL_ANALYZER_BASE_URL = os.environ.get("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:1234/v1")
_DEFAULT_ANALYZER_MODEL = os.environ.get(
    "INFERENCE_ANALYZER_MODEL",
    _LOCAL_ANALYZER_MODEL_ID,
)
_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*<function=([^>\n]+)>\s*(.*?)\s*</function>\s*</tool_call>",
    flags=re.DOTALL | re.IGNORECASE,
)
_TOOL_CALL_PARAMETER_RE = re.compile(
    r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
    flags=re.DOTALL | re.IGNORECASE,
)
_THINK_TAG_RE = re.compile(r"</?think>", flags=re.IGNORECASE)


def _get_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _level_transfer_guidance_enabled() -> bool:
    """Opt-in level-transfer prompts; set before constructing the agent.

    ARC3_LEVEL_TRANSFER_GUIDANCE defaults off, preserving the original prompts.
    """
    return _get_env_bool("ARC3_LEVEL_TRANSFER_GUIDANCE", False)


def _contains_tool_call_markup(*chunks: str) -> bool:
    for chunk in chunks:
        lowered = chunk.lower()
        if "<tool_call" in lowered or "<function=" in lowered:
            return True
    return False


def _strip_tool_call_markup(text: str) -> str:
    if not text.strip():
        return ""
    stripped = _TOOL_CALL_BLOCK_RE.sub("", text)
    return stripped.strip()


def _recover_tool_calls_from_markup(*chunks: str) -> list[dict[str, Any]]:
    recovered: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for chunk in chunks:
        if not chunk.strip():
            continue
        for match in _TOOL_CALL_BLOCK_RE.finditer(chunk):
            tool_name = str(match.group(1) or "").strip()
            if not tool_name:
                continue
            raw_body = str(match.group(2) or "")
            arguments = {
                str(parameter_name).strip(): value
                for parameter_name, value in _TOOL_CALL_PARAMETER_RE.findall(raw_body)
                if str(parameter_name).strip()
            }
            cache_key = (
                tool_name,
                json.dumps(arguments, ensure_ascii=True, sort_keys=True),
            )
            if cache_key in seen:
                continue
            seen.add(cache_key)
            recovered.append(
                {
                    "id": f"markup-call-{len(recovered) + 1}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(arguments, ensure_ascii=True),
                    },
                }
            )
    return recovered


def _get_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


_LOCAL_ANALYZER_MAX_OUTPUT = _get_env_int("LOCAL_ANALYZER_MAX_OUTPUT", 0)
_LOCAL_ANALYZER_CONTEXT_WINDOW = _get_env_int("LOCAL_ANALYZER_CONTEXT_WINDOW", 32768)
_LOCAL_ANALYZER_TIMEOUT = _get_env_float("LOCAL_ANALYZER_TIMEOUT", 0.0)
_LOCAL_ANALYZER_TOOL_STEPS = _get_env_int("LOCAL_ANALYZER_TOOL_STEPS", 12)
_LOCAL_ANALYZER_TOOL_TIMEOUT = _get_env_int("LOCAL_ANALYZER_TOOL_TIMEOUT", 30)
_LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS = _get_env_int("LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS", 1024)
_LOCAL_ANALYZER_YIELD_SECONDS = _get_env_float("LOCAL_ANALYZER_YIELD_SECONDS", 0.0)
# A wall-clock budget is really a token budget with the throughput baked in:
# the same 45 seconds bought 2,140 generated tokens on one backend and 3,829 on
# a faster one, so every serving change silently retuned it. Measured in
# tokens the setting means the same thing whatever the hardware does. Zero
# disables, as with the seconds; if both are set, whichever comes first ends
# the turn.
_LOCAL_ANALYZER_YIELD_TOKENS = _get_env_int("LOCAL_ANALYZER_YIELD_TOKENS", 0)
_LOCAL_ANALYZER_ENABLE_THINKING = _get_env_bool("LOCAL_ANALYZER_ENABLE_THINKING", True)
_LOCAL_ANALYZER_TEMPERATURE = _get_env_float("LOCAL_ANALYZER_TEMPERATURE", 0.6)
_LOCAL_ANALYZER_TOP_P = _get_env_float("LOCAL_ANALYZER_TOP_P", 0.95)
_LOCAL_ANALYZER_TOP_K = _get_env_int("LOCAL_ANALYZER_TOP_K", 20)
_LOCAL_ANALYZER_SEED = _get_env_int("LOCAL_ANALYZER_SEED", -1)
_REQUEST_SAFETY_MARGIN_TOKENS = 512
_CONTEXT_OVERFLOW_RETRY_TRIM_TOKENS = 512
def _context_drain_floor_mode() -> bool:
    """Whether the drain stops ABOVE the target rather than below it.

    History is dropped in whole blocks, so trimming "until under the target"
    normally overshoots by most of a block. With a 40k budget and a 12k drain
    the band is nominally 28k-40k, but simulation with ~6k blocks gives an
    effective 14k-38k: the average sits ~3k below the nominal midpoint and the
    minimum is far below the target.

    Floor mode drops the mandatory blocks to get under budget, then stops
    before a drop would cross below the target - so the retained content keeps
    a real floor and the band means what it says. The cost is trim frequency:
    the overshoot is what lengthens the interval between trims, so removing it
    trims about half again as often, which shortens the stable prefix and cuts
    the prefix-cache hit rate. Off by default for that reason; the same average
    depth is available by raising the budget, which does not touch trim
    frequency."""
    return _get_env_bool("ARC3_CONTEXT_DRAIN_FLOOR", False)


def _context_drain_tokens() -> int:
    """Hysteresis for context trimming, in tokens.

    Without it, trimming drops the oldest exchange the moment the estimate
    exceeds the budget and stops the instant it fits - so once history reaches
    the ceiling, nearly every turn evicts a little. Each eviction is cheap in
    tokens but total in cache cost: prefix caching keys on the token sequence
    from position zero, so removing anything from the FRONT shifts every
    surviving token and invalidates the whole cached prefix, not just the part
    removed. Frequent small evictions are therefore the worst case.

    With a drain set, trimming still TRIGGERS at the budget but keeps dropping
    until it is `drain` tokens below it, so evictions become rare and large:
    the same tokens discarded overall, a fraction of the invalidations. The
    cost is average history depth - the window sits below the budget most of
    the time rather than pressed against it.

    Size it against tokens per turn: at ~4-5k per turn including the opener,
    8-12k buys two or three clean turns between evictions. 0 (default) keeps
    the original drop-until-it-fits behavior."""
    return max(0, _get_env_int("ARC3_CONTEXT_DRAIN_TOKENS", 0))


def _history_turn_drain() -> int:
    """Hysteresis for the assistant-turn cap, in assistant messages.

    The token trimmer got hysteresis in ARC3_CONTEXT_DRAIN_TOKENS, but the
    turn cap runs AFTER it and had none: once history sits at the cap, every
    commit shaves back to exactly N, so the prefix is invalidated on every
    single turn. With a 46k window and a cap of 30 assistant messages the cap
    is the binding limit, which is why a token drain alone changed nothing.

    Set this and the cap triggers at N but drains to N - drain, so evictions
    become periodic instead of continuous."""
    return max(0, _get_env_int("ARC3_HISTORY_TURN_DRAIN", 0))


def _history_drain_coalesce() -> bool:
    """Take the deep turn cut whenever the token trimmer already dropped.

    Invalidation is all-or-nothing: removing anything from the front of
    history shifts every surviving token, so a commit that has already
    dropped blocks for token reasons has already paid the full cache cost.
    Dropping turns in that same commit is therefore free, whereas letting the
    cap fire on its own a turn later costs a second invalidation. This makes
    the two mechanisms evict together rather than alternately."""
    return _get_env_bool("ARC3_HISTORY_DRAIN_COALESCE", False)


def _persistent_history_assistant_turns() -> int:
    """Cap on assistant messages kept in persistent history.

    Counts ASSISTANT MESSAGES, not analyzer turns: a turn that inspects, then
    computes, then acts spends three of them, so 30 is closer to ten or
    fifteen turns of real history than to thirty. The cap is independent of
    the token budget, so with a large context window it can be the binding
    constraint while thousands of tokens sit unused - which is what the
    default 30 does at 98k, having been chosen when the window was 32k.

    0 or negative disables the cap entirely, leaving the token budget as the
    only limit."""
    return _get_env_int("ARC3_HISTORY_ASSISTANT_TURNS", 30)
_WM_NUDGE_TURNS = _get_env_int("ARC3_WM_NUDGE_TURNS", 0)
_WM_WIPE_ON_GAME_OVER = _get_env_int("ARC3_WM_WIPE_ON_GAME_OVER", 1)
_AUTO_FRAME_DIFF = _get_env_bool("ARC3_AUTO_FRAME_DIFF", False)
_AUTO_FRAME_DIFF_BUDGET = _get_env_int("ARC3_AUTO_FRAME_DIFF_BUDGET", 300)
_AUTO_FRAME_DIFF_MAX_GROUP = 40
_DIFF_IMAGE = _get_env_bool("ARC3_DIFF_IMAGE", False)
_RESPONSE_META_MAX_CHARS = 4000

_PYTHON_TOOL_DESCRIPTION = (
    "Run one ephemeral Python snippet against preloaded ASCII game state. Available globals: "
    "`current_frame`, `previous_frame`, `history`, `transitions`, `last_transition`, "
    "`valid_actions`, `last_action_call_result`, "
    "and `action(actions)` for executing one or more real environment actions. "
    "`current_frame` and each `history[*].frame` expose only `.ascii`, `.segmentation`, `.step`, `.level`, and `.shape` (a `(rows, cols)` tuple); "
    "`history[-1].frame` is the current post-action frame, not the previous frame. "
    "For before/after diffs, compare `previous_frame` to `current_frame` or use `last_transition.before_frame` and `.after_frame`. "
    "For MOUSE, pass `row` and `col` integer fields; legacy x/y fields are rejected. "
    "The raw numeric grid is not available. Use `.segmentation` as the primary view; use `.ascii` only to read a small, specific region. "
    "Use `print(...)` for compact output or assign final data to `result`."
)

def _normalize_valid_actions(valid_actions: list[str] | None) -> list[str]:
    names: list[str] = []
    for value in valid_actions or []:
        engine_name = to_engine_action(value)
        name = to_model_action(engine_name or value)
        if name and name not in names:
            names.append(name)
    return names


def _format_valid_action_line(valid_actions: list[str] | None) -> str:
    names = _normalize_valid_actions(valid_actions)
    if not names:
        return "unknown"
    return ", ".join(names)


def _terminal_action_reason(result: dict[str, Any]) -> str | None:
    if result.get("run_complete"):
        return "run_complete"
    if result.get("game_over"):
        return "game_over"
    if result.get("level_completed"):
        return "level_completed"
    if result.get("done"):
        return "done"
    return None


def _terminal_action_stop_detail(reason: str | None) -> str:
    if reason == "run_complete":
        return "No further actions were executed because the run is already complete."
    if reason == "game_over":
        return (
            "No further actions were executed because the previous action reached GAME_OVER "
            "(attempt failed; the run is NOT finished). The runner will auto-reset the level "
            "before your next turn. Cause is either the action itself (hazard) or a depleted "
            "step/time budget (usually a shrinking bar at the grid border)."
        )
    if reason == "level_completed":
        return (
            "No further actions were executed because the previous action completed a level; "
            "re-ground on the new scene before acting again."
        )
    if reason == "done":
        return "No further actions were executed because the environment reported done."
    return "No further actions were executed because the previous action reached a terminal state."


def _display_action_number(action_num: int) -> int:
    return max(1, int(action_num) + 1)


def _stale_summary_lines(summary: dict[str, Any]) -> list[str]:
    """Rendering for a step summary carried over from an earlier exchange of
    the same turn. States that explicitly instead of re-narrating the old
    outcome as if it had just happened. Two ways a turn can reach here:
    a snippet ran and executed nothing (default wording), or the exchange
    produced no tool call at all (stale_reason='no_tool_call') - in which case
    claiming a snippet ran would be false."""
    if summary.get("stale_reason") == "no_tool_call":
        lines = [
            "Your previous exchange this turn produced no tool call and "
            "executed no actions."
        ]
    else:
        lines = [
            "Your last python snippet executed NO actions (it only computed or "
            "inspected state - that is fine)."
        ]
    executed_actions = summary.get("executed_actions")
    rendered = ""
    if isinstance(executed_actions, list):
        rendered = ", ".join(
            str(name).strip() for name in executed_actions[:10] if str(name).strip()
        )
    if summary.get("game_over"):
        fatal = str(summary.get("fatal_action") or "")
        lines.append(
            "Reminder of the still-pending outcome from earlier this turn: the "
            f"executed sequence ({rendered or 'see earlier message'}) ended in "
            "GAME OVER"
            + (f" after '{fatal}'" if fatal else "")
            + "; the level was auto-reset and you are planning the fresh attempt "
            "now. Do NOT re-submit that sequence - the full diagnosis "
            "instructions were given in an earlier message this turn."
        )
    elif summary.get("level_transition"):
        lines.append(
            "Reminder: the earlier sequence this turn completed the level; you "
            "are now on the new level."
        )
    elif rendered:
        lines.append(
            "For reference, the most recent executed sequence earlier this turn "
            f"was: {rendered}."
        )
    return lines


def _state_identity_border() -> int:
    """Border cropped when deciding whether two boards are THE SAME STATE.

    Distinct from ARC3_NOOP_GUARD_BORDER, which asks a different question. The
    batch stop asks "did this action accomplish anything meaningful", and there
    cropping the HUD is right: a timer ticking down is not progress. The repeat
    guards and the death ledger ask "have I been here before", and there
    cropping is a liability - two boards with identical interiors but different
    HUD are treated as one state, so a mechanic gated on anything the HUD
    encodes (a countdown, a phase, a collected count) can be mis-recorded. The
    resulting block is permanent and self-sealing: the action never executes,
    so no evidence can arrive to correct it.

    Defaults to 0 - exact whole-frame matching, which is safe by construction
    and matches the reference implementation. Raise it to trade that guarantee
    for more matches."""
    return max(0, _get_env_int("ARC3_STATE_IDENTITY_BORDER", 0))


def _interior_state_hash(grid: list[list[int]]) -> str:
    """Hash identifying whether two boards are the same state, for the repeat
    guards and the death ledger. Crops ARC3_STATE_IDENTITY_BORDER cells, which
    defaults to 0 - the whole frame, HUD included. Small grids hash whole."""
    import hashlib

    border = _state_identity_border()
    rows = len(grid)
    cols = max((len(row) for row in grid), default=0)
    if rows > 2 * border and cols > 2 * border:
        core = [tuple(row[border:cols - border]) for row in grid[border:rows - border]]
    else:
        core = [tuple(row) for row in grid]
    return hashlib.sha1(repr(core).encode("utf-8")).hexdigest()


_LEAKED_TOOLCALL_RE = None


def _extract_leaked_tool_calls(message: dict[str, Any]) -> dict[str, Any]:
    """Fallback for servers without a tool-call parser for the model's
    dialect: when a response carries no structured tool_calls but the content
    contains Nemotron-style <TOOLCALL>[{...}]</TOOLCALL> markup, parse it and
    synthesize the OpenAI-shaped tool_calls the harness expects. Malformed
    payloads leave the message untouched. Disable with
    ARC3_TOOLCALL_TEXT_FALLBACK=0."""
    global _LEAKED_TOOLCALL_RE
    if not _get_env_bool("ARC3_TOOLCALL_TEXT_FALLBACK", True):
        return message
    if not isinstance(message, dict) or message.get("tool_calls"):
        return message
    content = message.get("content")
    if not isinstance(content, str) or "<TOOLCALL>" not in content:
        return message
    import json as _json
    import re as _re
    import uuid as _uuid

    if _LEAKED_TOOLCALL_RE is None:
        _LEAKED_TOOLCALL_RE = _re.compile(
            r"<TOOLCALL>\s*(\[.*?\])\s*</TOOLCALL>", _re.DOTALL
        )
    match = _LEAKED_TOOLCALL_RE.search(content)
    if match is None:
        return message
    try:
        parsed = _json.loads(match.group(1))
    except _json.JSONDecodeError:
        return message
    if not isinstance(parsed, list):
        parsed = [parsed]
    calls: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        arguments = item.get("arguments", {})
        if not isinstance(arguments, str):
            arguments = _json.dumps(arguments, ensure_ascii=False)
        calls.append(
            {
                "id": f"call_{_uuid.uuid4().hex}",
                "type": "function",
                "function": {
                    "name": str(item["name"]),
                    "arguments": arguments,
                },
            }
        )
    if not calls:
        return message
    remainder = (content[: match.start()] + content[match.end():]).strip()
    updated = dict(message)
    updated["tool_calls"] = calls
    updated["content"] = remainder or None
    return updated


def _env_int(name: str, default: int) -> int:
    """A numeric override that cannot take the run down on a typo."""
    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


_RETRYABLE_HTTP_STATUSES = {429, 500, 502, 503, 504, 520, 522, 524}
# Establishing a TCP connection is fast or it is not happening: a server that
# has exited refuses instantly, and one that is unreachable will not become
# reachable inside the read budget. Separate from the read timeout so a dead
# endpoint fails in seconds rather than waiting out the generation budget.
_CONNECT_TIMEOUT_SECONDS = 10.0


def _http_retry_delay_seconds(
    attempt: int,
    retry_after: str | None,
    base_seconds: float,
    max_seconds: float = 60.0,
) -> float:
    """Exponential backoff with jitter, or the server's Retry-After when it
    parses as a number and asks for longer. Capped at 60s per wait.

    Retry-After raises the wait, never lowers it. A provider under load can
    answer `Retry-After: 1` to every request - observed at 100 consecutive
    retries roughly 1.3s apart - and honouring that literally means hammering
    an endpoint that has already said no, which neither clears the rate limit
    nor helps the run. The configured base is the floor; the server can still
    ask for more.
    """
    backoff = min(max_seconds, base_seconds * (2 ** attempt)) + random.uniform(0.0, 1.0)
    if retry_after:
        try:
            # max_seconds caps the exponential, not an explicit request from the
            # server: a Retry-After of 30 against a 5s ceiling still waits 30,
            # because the endpoint said so and hammering it earlier helps no one
            return max(backoff, min(60.0, float(retry_after)))
        except ValueError:
            pass
    return backoff


def _post_with_retries(
    post_fn,
    *,
    retries: int,
    base_seconds: float,
    max_seconds: float = 60.0,
    initial_seconds: float = 0.0,
    sleep=time.sleep,
    monotonic=time.monotonic,
):
    """Retry transient HTTP failures (rate limits, gateway errors) in place,
    BEFORE they escalate to turn abort + history rollback. Returns the final
    response either way; non-retryable statuses return immediately and flow
    into the existing error handling. Rate-limit retries here preserve the
    turn's partial tool work, which the abort path deliberately discards."""
    # A negative count means keep trying. Zero keeps its obvious meaning - do
    # not retry - so a typo disables retries rather than hanging the run.
    unlimited = retries < 0
    budget = max(0, retries)
    # A floor on total retry time, used once. On Kaggle the benchmark is
    # released before the server has finished loading, so the first request of
    # every game meets a refused connection and has to wait it out - minutes,
    # not the seconds the ordinary budget allows. Expressed as time rather than
    # a count because the count that covers a 15-minute load would be absurd at
    # any sensible delay, and would then apply to every later failure too.
    started_at = monotonic()

    def _budget_label(attempt_index: int) -> str:
        """What the log should call the limit. Only reached outside a grace
        period now - inside one the retry lines are suppressed entirely, since
        "1/3" describes a budget that is not the one in force."""
        if unlimited:
            return "unlimited"
        if attempt_index >= budget:
            return f"{initial_seconds:.0f}s grace"
        return str(budget)

    def _exhausted(attempt_index: int) -> bool:
        if unlimited:
            return False
        if attempt_index < budget:
            return False
        return (monotonic() - started_at) >= initial_seconds

    def _announce_grace(reason: str) -> None:
        """Said once, on the first failure, so the grace is visible from the
        start. Without it the log shows three ordinary-looking retries before
        the "Ns grace" label appears, and a reader watching a cold start cannot
        tell a long wait from a broken one."""
        nonlocal announced
        if announced or initial_seconds <= 0:
            return
        announced = True
        log.warning(
            "first analyzer request failed (%s); retrying for up to %.0fs while "
            "the endpoint comes up, then falling back to %d retries per request",
            reason, initial_seconds, budget,
        )

    announced = False
    response = None
    attempt = -1
    while True:
        attempt += 1
        try:
            response = post_fn()
        except requests.exceptions.RequestException as exc:
            # The server was never reached - refused, DNS, a socket error, or a
            # connect timeout. Without this the exception leaves the turn
            # entirely and the analyzer-level retry starts it over, discarding
            # whatever tool work it had already done; retrying the request in
            # place keeps it. A ReadTimeout is deliberately not caught: the
            # response body stalled, which has its own path that can yield with
            # history preserved.
            if _is_read_timeout(exc) or _exhausted(attempt):
                raise
            _announce_grace(type(exc).__name__)
            delay = _http_retry_delay_seconds(attempt, None, base_seconds, max_seconds)
            # Silent while the grace is carrying the loop: at 14 games polling a
            # cold endpoint every 5s for 15 minutes this line alone would be
            # ~2,500 entries, burying everything else. The announcement above
            # already said a wait is in progress, and the outcome is logged when
            # the loop ends either way.
            if initial_seconds <= 0:
                log.warning(
                    "analyzer endpoint unreachable (%s); retry %d/%s in %.1fs",
                    type(exc).__name__, attempt + 1, _budget_label(attempt), delay,
                )
            sleep(delay)
            continue
        status = getattr(response, "status_code", 200)
        if status not in _RETRYABLE_HTTP_STATUSES or _exhausted(attempt):
            return response
        _announce_grace(f"HTTP {status}")
        retry_after = None
        headers = getattr(response, "headers", None)
        if headers is not None:
            retry_after = headers.get("Retry-After")
        delay = _http_retry_delay_seconds(attempt, retry_after, base_seconds, max_seconds)
        if initial_seconds <= 0:
            log.warning(
                "analyzer endpoint returned HTTP %s; retry %d/%s in %.1fs",
                status, attempt + 1, _budget_label(attempt), delay,
            )
        sleep(delay)
    return response


_CONTROL_MESSAGE_KEY = "_arc3_control"

_YIELD_RESUME_PROMPT_WITH_TOOLS = (
    "You yielded control on the turn time budget. Your tool results above are "
    "still valid - do NOT restart your analysis or re-inspect the board from "
    "scratch. Continue from them and call `action(actions)` with your best "
    "next action or short batch."
)

_YIELD_RESUME_PROMPT_WITH_TOOLS_NEUTRAL = (
    "You yielded control on the turn time budget. This is a scheduling "
    "interruption, not a signal that your analysis is complete. Your tool "
    "results above are still valid - do NOT restart or re-inspect the board "
    "from scratch. Pick up where you left off: keep investigating if you still "
    "need to, and call `action(actions)` once your evidence supports a choice."
)


def _yield_resume_tone() -> str:
    """Wording for the commonest resumption message.

    'commit' (default) is the original text, which reads as though the model
    has finished: it asks for the best next action. At a 60s budget that fires
    after very little deliberation, so it can push a commitment the evidence
    does not yet support.

    'neutral' keeps the load-bearing part - do not restart, your results are
    valid - while naming the interruption for what it is and leaving the
    decision to continue investigating open. The anti-stall backstops are
    untouched: the no-tool-call nudge, the tool-step-exhaustion prompt and the
    timeout prompt all still push to act, and they fire on exhausted work
    rather than on elapsed time.

    'minimal' goes further and does not mention the yield at all. A yield that
    executed nothing and a snippet that only inspected are the same situation
    from the model's side - nothing happened, carry on - and the scheduling
    reason is harness detail it cannot act on. The wording echoes the mid-turn
    stale opener so a resumption reads as a continuation rather than as its own
    kind of event, keeping only the part that is load-bearing: the earlier
    results are still good.

    Read the arm as actions performed against tokens generated: the change
    should raise tokens per action if it is working as intended, and the
    question is whether the actions get better.
    """
    tone = os.environ.get("ARC3_YIELD_RESUME_TONE", "").strip().lower()
    return tone if tone in ("neutral", "minimal") else "commit"

_YIELD_RESUME_PROMPT_NO_TOOLS = (
    "You spent the entire turn budget reasoning without calling a tool. Stop "
    "deliberating and call the `python` tool NOW: inspect "
    "`current_frame.segmentation`, then call `action(actions)` with the best "
    "available action. Acting on a partial model beats another turn of "
    "analysis."
)

_YIELD_RESUME_PROMPT_WITH_TOOLS_MINIMAL = (
    # Says nothing about yielding. From the model's side a yield with nothing
    # executed and an inspect-only snippet are the same situation - nothing
    # happened, carry on - and the scheduling reason is harness detail it
    # cannot act on. Deliberately echoes the mid-turn stale opener it already
    # sees often, so a resumption reads as a continuation rather than as a
    # distinct kind of message. Kept: the earlier results are still good, so
    # do not start over.
    "Your last python snippet executed no actions (it only computed or "
    "inspected state - that is fine). Your tool results above are still "
    "valid: continue from them rather than re-inspecting the board from "
    "scratch."
)

_YIELD_RESUME_PROMPT_NO_TOOLS_MINIMAL = (
    # The with-tools wording would be false here: there was no snippet at all,
    # so there are no results to continue from and the useful instruction is
    # where to start instead.
    "Your previous exchange produced no tool call and executed no actions. "
    "Start with the `python` tool: inspect `current_frame.segmentation`, then "
    "call `action(actions)` when your evidence supports a choice."
)

_YIELD_RESUME_PROMPT_NO_TOOLS_NEUTRAL = (
    # The same edit as the with-tools pair: keep what is load-bearing - that
    # nothing executed, and the concrete first step - and drop the pressure to
    # commit. This fires when the yield budget expired on a turn that never
    # called a tool, so the model really did spend the turn reasoning; what is
    # unwarranted is telling it that acting on a partial model is better, when
    # the budget expiring says nothing about whether its analysis was close.
    "You yielded control on the turn time budget without calling a tool. This "
    "is a scheduling interruption, not a signal that your analysis is complete "
    "- but nothing has been executed and the board is unchanged. Start with "
    "the `python` tool: inspect `current_frame.segmentation`, then call "
    "`action(actions)` once your evidence supports a choice."
)

_DEAD_REASONING_STUB = "(previous attempt produced no tool call)"

_DEGENERATE_CONTENT_STUB = (
    "(previous reply was highly repetitive and has been omitted; do not "
    "continue that pattern - write concise plain prose)"
)

_STATE_ONLY_CAPTION = (
    # Leads with the assurance rather than with "Current board state":
    # _build_user_message appends its own "Current grid image:" immediately
    # below, so opening with a second caption put two labels for one picture
    # next to each other. The load-bearing part is that the opener has not
    # gone stale, not that a board follows.
    "Nothing has been executed since the turn opener above, so the step, "
    "level and valid actions stated there still hold."
)


def _yield_resume_prompt_mode() -> str:
    """How a resumed turn re-opens.

    'full'       (default) rebuild the complete opener - 2-3k tokens restating
                 state the retained opener already carries.
    'short'      a brief continuation sentence and nothing else.
    'state_only' the continuation sentence plus a one-line caption and the
                 CURRENT board image(s). Measurements showed 'short' holding up
                 while images were attached and degrading without them, which
                 is consistent with the omitted text (batch summary, state
                 line, diffs, ledger) being what the image was covering for.
                 This mode makes that explicit: drop the boilerplate, keep the
                 state, in the cheapest form available.
    """
    mode = os.environ.get("ARC3_YIELD_RESUME_PROMPT", "").strip().lower()
    return mode if mode in ("short", "state_only") else "full"


_RESUME_PROMPT_TIMEOUT = (
    "The previous request to the model timed out and was abandoned. Your tool "
    "results above are still valid - do NOT restart your analysis. Continue "
    "from them and call `action(actions)` with your best next action, keeping "
    "your reasoning short so the next request completes."
)

_RESUME_PROMPT_TOOL_STEPS_EXHAUSTED = (
    "You used every available tool step this turn without calling "
    "`action(actions)`. Your tool results above are still valid - do NOT begin "
    "a new investigation. Act NOW with the best option you already have."
)


def _yield_resume_prompt(reason: str) -> str:
    """Continuation text for a resumed turn, chosen by why it ended.

    'timeout'      - the request exceeded the analyzer read timeout: the work
                     already done this turn is intact, the model just needs to
                     wrap up quickly.
    'tool_steps'   - the step budget ran out mid-investigation: results exist,
                     the missing thing is the action.
    'yield_tools'  - the time budget ran out with tool work done: same premise,
                     softer framing.
    'yield_bare'   - the turn produced only reasoning, so there is nothing
                     above to continue from and the useful instruction is to
                     stop deliberating.
    """
    if reason == "timeout":
        return _RESUME_PROMPT_TIMEOUT
    if reason == "tool_steps":
        return _RESUME_PROMPT_TOOL_STEPS_EXHAUSTED
    tone = _yield_resume_tone()
    if reason == "yield_tools":
        if tone == "minimal":
            return _YIELD_RESUME_PROMPT_WITH_TOOLS_MINIMAL
        return (
            _YIELD_RESUME_PROMPT_WITH_TOOLS_NEUTRAL
            if tone == "neutral"
            else _YIELD_RESUME_PROMPT_WITH_TOOLS
        )
    if tone == "minimal":
        return _YIELD_RESUME_PROMPT_NO_TOOLS_MINIMAL
    return (
        _YIELD_RESUME_PROMPT_NO_TOOLS_NEUTRAL
        if tone == "neutral"
        else _YIELD_RESUME_PROMPT_NO_TOOLS
    )


def _fatal_lookahead_enabled() -> bool:
    """Block a batch that replays a recorded fatal path BEFORE spending any of
    its actions, rather than running the harmless prefix and refusing only the
    final move.

    Default off keeps the per-position behaviour: the prefix executes and the
    model ends up at the brink, which costs those actions but leaves it
    somewhere it can still choose differently. With lookahead on, nothing is
    spent and the model must resubmit a shorter batch to reach that point."""
    return _get_env_bool("ARC3_DEATH_GUARD_LOOKAHEAD", False)


def _yield_on_timeout_enabled() -> bool:
    """Treat an analyzer read timeout as a yield instead of a failed turn.

    The default path clears preserve_history, so the whole invocation's tool
    calls and results are rolled back and the retry re-derives them from
    scratch - on Kaggle that is a ~900s round trip discarded. Nothing can have
    executed when a timeout fires (an executing dispatch breaks the loop before
    another request is issued), so preserving the exchanges is safe: history
    ends on a tool message and the resumed invocation appends a user message,
    exactly the shape every yielded turn already produces.

    Scoped to read timeouts only. Connection errors and HTTP failures keep the
    rollback path, and context-overflow handling is untouched - preserving an
    over-long context would just reproduce the failure."""
    return _get_env_bool("ARC3_YIELD_ON_TIMEOUT", True)


def _is_read_timeout(exc: BaseException) -> bool:
    """requests raises ReadTimeout for a stalled response body; ConnectTimeout
    (a sibling of the same base) means the server was never reached, so it is
    deliberately excluded."""
    return isinstance(exc, requests.exceptions.ReadTimeout)


def _stub_dead_reasoning_enabled() -> bool:
    """Replace reasoning-only assistant replies (no tool calls, no content)
    with a short stub IN-TURN. Without this the dead trace is re-sent on every
    later request of the same turn - observed at ~35k characters, truncated
    mid-sentence by finish_reason=length, i.e. a broken thought replayed as
    context."""
    return _get_env_bool("ARC3_STUB_DEAD_REASONING", False)


_HISTORY_IMAGE_PLACEHOLDER = "[grid image omitted from history]"


def _history_image_keep() -> int | None:
    """How many of the most recent image-bearing history MESSAGES keep their
    images. None (unset) preserves today's behavior - images accumulate for as
    long as their message is retained, ~400 tokens each at
    MULTIMODAL_UPSCALE=10. 0 strips every image from history, so only the
    current turn is visual. Counting per message rather than per image keeps a
    multi-image turn (game-over visuals) together in one slot."""
    raw = os.environ.get("ARC3_HISTORY_IMAGE_KEEP", "").strip()
    if not raw:
        return None
    try:
        return max(0, int(raw))
    except ValueError:
        return None


def _resume_prompt_images_enabled() -> bool:
    """Whether a resumption message re-attaches the grid image. A resumption
    only happens when nothing executed, so the board is identical to the one
    the retained opener already showed; skipping it saves an image per
    resumption and stops resumptions from occupying history retention slots.
    Default True = today's behavior."""
    return _get_env_bool("ARC3_RESUME_PROMPT_IMAGES", True)


def _message_has_image(message: dict[str, Any]) -> bool:
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(part, dict) and part.get("type") == "image_url"
        for part in content
    )


def _strip_message_images(message: dict[str, Any]) -> dict[str, Any]:
    """Replace image parts with a short text marker. The marker keeps the
    message well formed (a message whose only part was an image would
    otherwise be empty) and tells the model a visual was there rather than
    silently showing a turn that never had one."""
    content = message.get("content")
    if not isinstance(content, list):
        return message
    parts: list[Any] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "image_url":
            parts.append({"type": "text", "text": _HISTORY_IMAGE_PLACEHOLDER})
        else:
            parts.append(part)
    return {**message, "content": parts}


def _apply_history_image_window(
    messages: list[dict[str, Any]], keep: int
) -> list[dict[str, Any]]:
    """Keep images on the `keep` most recent image-bearing messages, strip the
    rest. Walks backwards so recency wins; originals are never mutated."""
    seen = 0
    result: list[dict[str, Any]] = []
    for message in reversed(messages):
        if _message_has_image(message):
            seen += 1
            if seen > keep:
                message = _strip_message_images(message)
        result.append(message)
    result.reverse()
    return result


# Tag for the pair of messages a rolling summary is made of: the request that
# asks for it and the reply that is it. Deliberately absent from
# _CONTROL_KIND_ENV, so no pruning knob can remove them - the trimmer's
# summary-aware drain navigates by these, and losing one would silently return
# it to ordinary block dropping.
_SUMMARY_CONTROL_KIND = "summary"
# Below this a reply that also called a tool is a preamble rather than a
# summary. Real ones measured 4,385 to 6,248 characters; an introduction to a
# tool call measured nine.
_SUMMARY_MIN_USABLE_CHARS = 500


def _summary_interval_tokens() -> int:
    """New context tokens between summary attempts, successful or not. 0 disables.

    Count newly appended prompts, assistant messages (including reasoning), tool
    results and images, excluding the summary exchange itself. Re-sending or
    evicting existing history does not change this counter.

    The model is asked for a summary on a schedule rather than when it feels
    like one. Measured across a run where the prompt asked for a note every
    turn: one game in fourteen complied consistently, and the rest wrote
    something on roughly half their replies - so a mechanism that depends on
    the model choosing to summarise does not get summaries.

    Each summary is generated from live context, which keeps both the KV prefix
    and the recurrent state intact - a summary built from a truncated prefix
    would force a full prefill, and recurrent state cannot be rewound the way a
    KV prefix can be reused.

    The summary is appended when it is made, so every later turn is generated
    with it already present. Nothing is ever inserted underneath a message that
    was produced without it.
    """
    return max(0, _get_env_int("ARC3_SUMMARY_INTERVAL_TOKENS", 0))


def _drain_stop_at_summaries() -> bool:
    """Whether the drain may stop early on a summary.

    The drain normally evicts from the budget down to budget-minus-drain in one
    go. With this on it stops sooner when the head is already a summary request
    and the request is back under budget - so the retained history starts with a
    digest of everything dropped, without evicting a token more than the
    ordinary drain would.

    The earlier design made summaries the ONLY boundary, which could not
    overshoot in the good case but removed two intervals at once whenever a
    summary had failed - 38% of a 42k window at an 8k interval. Here the drain
    target still bounds the drop, so the worst case is exactly today's
    behaviour and the summary is an early exit rather than the sole exit.

    The trade is coverage: when no summary falls inside the drain range the drop
    stops mid-span with the previous summary already gone, and that stretch has
    nothing standing in for it. So ARC3_SUMMARY_INTERVAL_TOKENS wants to be
    comfortably SMALLER than ARC3_CONTEXT_DRAIN_TOKENS - several candidates per
    drop - which is the opposite of sizing the interval to replace the drain.
    """
    return _get_env_bool("ARC3_DRAIN_STOP_AT_SUMMARIES", False)


def _hide_followup_summaries() -> bool:
    """Send only the first summary in history, hiding the rest.

    Each summary covers everything before it, so a later one restates an
    earlier one's material - measured at 1 or 2 carried per request, 4,496
    characters each, so roughly 6% of a 42k window at worst.

    Applied to the wire and to the estimate, never to the stored history: the
    hidden ones stay so that when eviction removes the first, the next becomes
    the first and appears. Dropping them at commit would delete what it meant
    to hide.

    The cost is the thing the rolling design otherwise avoids. A hidden summary
    is absent while the turns after it are generated and becomes their head
    later, so those messages end up under something that was not there when
    they were written - which is exactly the rearrangement the append-only
    scheme was built to prevent.
    """
    return _get_env_bool("ARC3_HIDE_FOLLOWUP_SUMMARIES", False)


def _apply_summary_visibility(
    messages: list[dict[str, Any]], *, evicted: bool
) -> list[dict[str, Any]]:
    """Hide completed summaries, preserving a pending final summary request.

    Before the first eviction no completed summaries are shown. What they
    cover is still in context verbatim, so a summary there is pure duplication.
    Hiding them all means appending one does not change the prompt, leaving the fill
    phase a perfect append with a prefix that never moves. The first summary
    appears when eviction starts, which invalidates the prefix anyway.
    """
    if not _hide_followup_summaries():
        return messages

    # The request currently asking for a summary has no reply yet. It must
    # reach the model even when completed summaries are hidden. Split it off
    # without mutating the input; only stored pairs participate in visibility.
    pending_request: list[dict[str, Any]] = []
    history = messages
    if (
        messages
        and _message_is_summary(messages[-1])
        and str(messages[-1].get("role", "")).strip() == "user"
    ):
        pending_request = messages[-1:]
        history = messages[:-1]
    if not evicted:
        return [m for m in history if not _message_is_summary(m)] + pending_request
    kept: list[dict[str, Any]] = []
    seen_pair = False
    in_first_pair = False
    for message in history:
        if not _message_is_summary(message):
            kept.append(message)
            continue
        is_request = str(message.get("role", "")).strip() == "user"
        if is_request:
            # the request opens a pair; the first one and its reply stay
            in_first_pair = not seen_pair
            seen_pair = True
        if in_first_pair:
            kept.append(message)
    return kept + pending_request


def _summary_replaces_history() -> bool:
    """Whether a successful summary discards the history it summarised.

    The opposite end of the design from drain sizing: instead of evicting a
    fixed span and hoping a summary falls inside it, the summary IS the
    history. Everything above it goes the moment it lands, leaving system
    prompt, request, summary, and whatever the turn appends after.

    Safe in the sense that matters - nothing is dropped unless a usable summary
    exists, so a failed or rejected one leaves ordinary eviction in charge and
    the next turn tries again. But it is the more aggressive option, not the
    safer one: a large drain still leaves a turn or two of verbatim history,
    where this leaves none.

    Worth trying because the summaries turn out to be good. One measured at
    7,107 characters carried the board layout, the movement lattice, which
    actions did nothing and why, an unresolved two-branch interpretation of the
    last result, and the next probes to run - which is most of what the evicted
    turns were holding. And the raw frames are not in the history anyway:
    current_frame, previous_frame, history and frame_diff are runtime globals
    rebuilt every turn, so what eviction destroys is the interpretation, which
    is precisely what the summary keeps.

    Costs: the prefix is invalidated on every summary rather than on every
    eviction, and the turn immediately after has no verbatim record of the turn
    before it beyond what the opener carries.
    """
    return _get_env_bool("ARC3_SUMMARY_REPLACES_HISTORY", False)


def _summary_turn_context() -> bool:
    """Whether the summary request is told what just happened.

    The request goes out at turn start, BEFORE this turn's opener exists, so
    the newest thing in context is the previous turn's tool results. A level-up
    is visible there only as a `level_completed` flag inside a JSON payload -
    so a summary written at that moment can describe the level just finished as
    if it were still current, which is exactly the material a later turn needs
    to be right about.

    What goes in: what executed, whether the board changed, why a sequence
    stopped early, whether the level changed, and the step and level now.

    What stays out: the valid actions, which are state for the next decision
    and go stale the moment the summary outlives the turn; the tool inventory
    and the world-model demands, which are instructions that compete with
    "write the summary"; and the animation line, which is a pointer to go and
    read an accessor that will describe a different action by the time anyone
    reads the summary.
    """
    return _get_env_bool("ARC3_SUMMARY_TURN_CONTEXT", False)


def _summary_enable_thinking() -> bool:
    """Whether the summary request lets the model think first.

    OFF by default. Thinking about a summary is not obviously worth it - the
    material is all in front of the model - and it is expensive twice over:
    observed at 18,303 characters of reasoning that reached the output limit
    with no summary at all, and it doubles the generation for a request that
    already runs once per interval.

    The template must be able to suppress it without moving the prefix. In the
    Qwen3-family templates the `enable_thinking` conditional wraps BOTH the
    empty think block at the tail AND the "Reasoning effort is set to ..."
    sentence at the top of the system message, so the stock template diverges
    from the first tokens and every summary prefills the whole context -
    measured as prefix reuse falling from 75% to 45%. Forcing that conditional
    true (`{%- if true %}`) keeps the sentence unconditional and leaves only
    the tail block varying, which restores a byte-identical prefix. Check that
    before assuming this default is free.
    """
    return _get_env_bool("ARC3_SUMMARY_ENABLE_THINKING", False)


def _summary_max_gen_tokens(default: int | None) -> int:
    """Output limit: the normal cap, or 8192 when normal output is unlimited.

    The summary is asked for everything worth keeping, so it legitimately wants
    more room than a turn reply - and it is one request per interval, so a
    larger ceiling costs little. Raising this keeps the thinking;
    ARC3_SUMMARY_ENABLE_THINKING=0 removes it.
    """
    fallback = default if default is not None and default > 0 else 8192
    return max(1, _get_env_int("ARC3_SUMMARY_MAX_GEN", fallback))


def _message_is_summary(message: dict[str, Any]) -> bool:
    return message.get(_CONTROL_MESSAGE_KEY) == _SUMMARY_CONTROL_KIND


_CONTROL_KIND_ENV = {
    "nudge": "ARC3_PRUNE_NUDGES",
    "resume": "ARC3_PRUNE_RESUME_PROMPTS",
    "stub": "ARC3_PRUNE_DEAD_STUBS",
}


def _prune_control_context_enabled() -> bool:
    """Master switch: drop every kind of tagged control message when
    committing turn messages to persistent history. The per-kind knobs
    (ARC3_PRUNE_NUDGES / _RESUME_PROMPTS / _DEAD_STUBS) turn individual kinds
    on without it, so an arm can attribute each effect separately."""
    return _get_env_bool("ARC3_PRUNE_CONTROL_CONTEXT", False)


def _prune_control_kind_enabled(kind: str) -> bool:
    """A kind is pruned if the master switch is on OR its own knob is set.

    - 'nudge'  : the "you have not acted yet" corrective (a few hundred tokens)
    - 'resume' : the message appended when re-entering a non-executing turn
                 (a full opener in `full` mode, the short continuation in
                 `short` mode) - a restatement of unchanged state
    - 'stub'   : the placeholder that ARC3_STUB_DEAD_REASONING leaves behind
    """
    if _prune_control_context_enabled():
        return True
    env_name = _CONTROL_KIND_ENV.get(kind)
    return bool(env_name) and _get_env_bool(env_name, False)


def _any_control_pruning_enabled() -> bool:
    return _prune_control_context_enabled() or any(
        _get_env_bool(name, False) for name in _CONTROL_KIND_ENV.values()
    )


def _mark_control_message(message: dict[str, Any], kind: str = "resume") -> dict[str, Any]:
    message[_CONTROL_MESSAGE_KEY] = kind
    return message


def _strip_control_keys(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Wire-safe copy: the private tag never leaves the process."""
    cleaned: list[dict[str, Any]] = []
    for message in messages:
        if _CONTROL_MESSAGE_KEY in message:
            copy = dict(message)
            copy.pop(_CONTROL_MESSAGE_KEY, None)
            cleaned.append(copy)
        else:
            cleaned.append(message)
    return cleaned


def _prune_control_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove tagged control messages, then strip trailing user messages.

    The trailing-user strip is alternation repair, not a preference: pruning a
    spin-out turn (opener + dead reasoning, no tool calls) would otherwise
    leave the opener dangling while the next invocation appends another user
    message. A turn with no tool calls executed nothing, so the next opener
    regenerates identical state."""
    kept = [
        m
        for m in messages
        if not (
            m.get(_CONTROL_MESSAGE_KEY)
            and _prune_control_kind_enabled(str(m.get(_CONTROL_MESSAGE_KEY)))
        )
    ]
    while kept and str(kept[-1].get("role", "")).strip() == "user":
        kept.pop()
    return kept


def _reasoning_history_keys() -> tuple[str, ...]:
    """Message key(s) under which prior reasoning is sent back to the server.

    Backends disagree: OpenRouter returns and reads `reasoning`, while
    llama.cpp returns `reasoning_content` and its chat template reads ONLY
    that - so reasoning stored under `reasoning` is silently discarded there
    (measured: 69 prompt tokens vs 3670 for the same text under the two keys).
    A dropped reasoning block is invisible except as prompt growth that falls
    short of the tokens generated.

    Default `reasoning` preserves existing behaviour. Set to
    `reasoning_content` for llama.cpp, or to a comma list to send both - which
    is backend-agnostic but risks a template that renders both keys duplicating
    the text."""
    raw = os.environ.get("ARC3_REASONING_HISTORY_KEY", "").strip()
    keys = tuple(k.strip() for k in raw.split(",") if k.strip())
    return keys or ("reasoning",)


def _level_inventory_enabled() -> bool:
    """The START-vs-START object inventory shown on the first turn of a new
    level.

    It has its own switch because it answers a different question from the
    turn-to-turn diff: not "what did my last actions do" but "what is new
    about this level", with `appeared` types flagged as likely new mechanics.
    Previously it rode on the auto-diff machinery and was suppressed only when
    BOTH ARC3_AUTO_FRAME_DIFF and ARC3_DIFF_IMAGE were off - a coupling that
    was invisible from the knob names."""
    return _get_env_bool("ARC3_LEVEL_INVENTORY", False)


def _resume_diff_image_enabled() -> bool:
    """Re-attach the current turn's diff image to resumption messages.

    The diff image is produced only while building a full turn opener, so a
    resumption (short/state_only) carries the current grid but no diff. That
    is normally harmless - a resumption happens when nothing executed, so the
    diff would be identical to the one the turn's original opener already
    showed. It stops being harmless when ARC3_HISTORY_IMAGE_KEEP has stripped
    that opener's images: the model has then lost the diff entirely, and the
    resumption is its only chance to see what changed.

    Costs roughly one image per resumption. Off by default."""
    return _get_env_bool("ARC3_RESUME_DIFF_IMAGE", False)


def _reasoning_effort_ladder() -> list[str]:
    """Reduced reasoning-effort levels to step down through after a truncated
    generation, e.g. "medium,low".

    A reply cut off at the output ceiling is wasted twice over: the tokens are
    spent, and if the cut lands inside a tool call the turn produces nothing
    callable. Rather than accommodating long generations with a bigger ceiling,
    this asks the model for less deliberation until it completes a turn -
    stepping one rung down per truncation and resetting once an action
    executes, which is the evidence that the turn recovered.

    Empty (default) disables the behaviour entirely."""
    raw = os.environ.get("ARC3_REASONING_EFFORT_LADDER", "").strip()
    return [level.strip() for level in raw.split(",") if level.strip()]


def _preserve_thinking_kwarg() -> bool | None:
    """Value for the `preserve_thinking` chat-template kwarg, or None to omit.

    The payload builder sets chat_template_kwargs wholesale for vLLM, which
    replaces rather than merges - so a server-side
    --default-chat-template-kwargs carrying preserve_thinking can be silently
    dropped by a request. Defaults to on so the request asserts it rather than
    depending on server-side merge semantics; set to an empty string to omit
    the key entirely and leave the server default in charge."""
    raw = os.environ.get("ARC3_PRESERVE_THINKING", "1").strip().lower()
    if not raw:
        return None
    return raw in ("1", "true", "yes", "on")


def _openrouter_provider_prefs(provider: str) -> dict[str, Any] | None:
    """OpenRouter request-level provider pinning. ARC3_OPENROUTER_PROVIDER
    holds a provider name (or a comma-separated preference order); routing is
    then restricted to that list with fallbacks disabled, so every request in
    a run is served by a known engine. Under :nitro-style routing, providers
    differ in media handling (some silently drop video/image parts),
    quantization, and sampling behavior - a silent confound for any controlled
    comparison. Unset or non-OpenRouter: no change to the request."""
    if str(provider or "").strip().lower() != "openrouter":
        return None
    raw = os.environ.get("ARC3_OPENROUTER_PROVIDER", "").strip()
    if not raw:
        return None
    order = [name.strip() for name in raw.split(",") if name.strip()]
    if not order:
        return None
    return {"order": order, "allow_fallbacks": False}


_SLIM_MEMORY_SECTIONS = ("World model", "Plan")
_MEDIUM_MEMORY_SECTIONS = ("World model", "Plan", "Action model")

_MEMORY_SECTION_GUIDANCE_FULL = (
                "If you include assistant text before a tool call, use it to update your persistent memory sections. Section meanings: `World model:` = durable RULES of the environment that will still be true many actions from now (mechanics, what objects/colors mean, movement rules, win conditions) — never current positions, events, or what just happened. `Goal model:` = what winning requires. `Action model:` = what each action does mechanically. `Recent findings:` = this turn's observations and events (positions, action results, surprises). `Open questions:` = unresolved hypotheses. `Plan:` = intended next steps. `Cross-level notes:` = knowledge that carries across levels. Example — GOOD: `World model: Red cells are walls; each move shifts the player 2 cells; yellow keys open doors of the same color.` BAD: `World model: Batch stopped after 2 DOWNs, player now at (4,2).` (that belongs in `Recent findings:`). IMPORTANT: each labeled section you write REPLACES the stored version shown under 'Working world model carried from earlier turns' — write the COMPLETE updated section; any fact you omit is permanently lost."
)

_MEMORY_SECTION_MEANINGS = {
    "World model": (
        "durable RULES of the environment that will still be true many actions from now "
        "(mechanics, what objects/colors mean, movement rules, win conditions) - never "
        "current positions or what just happened"
    ),
    "Goal model": "what winning requires",
    "Action model": "what each action does mechanically",
    "Recent findings": "this turn's observations",
    "Open questions": "unresolved hypotheses",
    "Plan": "intended next steps",
    "Cross-level notes": "knowledge that carries across levels",
}


def _advertised_section_labels() -> tuple[str, ...]:
    """The sections the model should be TOLD about: exactly those the current
    mode carries.

    Advertising all seven while a mode drops five invites the model to write
    into sections it will never see again - the write is parsed, stored, and
    silently discarded from the prompt. It also confounds a slim/medium arm,
    because a poor result could mean either that fewer sections hurt or that
    output was wasted on sections that went nowhere."""
    labels = _memory_section_labels()
    if labels is None:
        return tuple(_MEMORY_SECTION_MEANINGS)
    # canonical order, so the advertisement matches the order the carried block
    # renders them in
    chosen = set(labels)
    return tuple(name for name in _MEMORY_SECTION_MEANINGS if name in chosen)


def _section_list_phrase(labels: tuple[str, ...], conjunction: str = "and") -> str:
    """Render the labels as prose. The conjunction is a parameter because the
    two call sites word it differently: the opener lists what is available
    ("and"), the no-tool-call followup offers alternatives ("or")."""
    quoted = [f"`{name}:`" for name in labels]
    if len(quoted) == 1:
        return quoted[0]
    if len(quoted) == 2:
        # no serial comma before a two-item conjunction
        return f" {conjunction} ".join(quoted)
    return ", ".join(quoted[:-1]) + f", {conjunction} " + quoted[-1]


def _memory_section_guidance_upstream() -> str:
    return (
        "If you include assistant text before a tool call, keep it short and use it to "
        "update the world model. Helpful optional prefixes are "
        f"{_section_list_phrase(_advertised_section_labels())}."
    )


def _memory_section_guidance_empty() -> str:
    labels = _advertised_section_labels()
    meanings = " ".join(
        f"`{name}:` = {_MEMORY_SECTION_MEANINGS[name]}." for name in labels
    )
    return (
        "You have no stored memory sections yet. If you include assistant text before a "
        f"tool call, use it to start them: {meanings}"
    )


_MEMORY_SECTION_GUIDANCE_UPSTREAM = (
    "If you include assistant text before a tool call, keep it short and use it to update "
    "the world model. Helpful optional prefixes are `World model:`, `Goal model:`, "
    "`Action model:`, `Recent findings:`, `Open questions:`, `Plan:`, and "
    "`Cross-level notes:`."
)


def _wm_section_explanation_mode() -> str:
    """Which wording explains the memory sections in the turn opener.

    'tufa' (also spelled 'upstream', the default) is the terse original: one
    sentence naming the prefixes, framed
    as optional. 'full' (default) is ours - a glossary of what each section
    means, a GOOD/BAD example, and the warning that a written section REPLACES
    the stored one so an omitted fact is lost. Roughly 1100 characters against
    250, every turn, and it asks for complete rewrites rather than short
    updates - so it is a candidate for longer generations as well as better
    memory."""
    mode = os.environ.get("ARC3_WM_SECTION_EXPLANATION", "tufa").strip().lower()
    return "full" if mode == "full" else "upstream"


_MEMORY_SECTION_GUIDANCE_EMPTY = (
    "You have no stored memory sections yet. If you include assistant text before a tool "
    "call, use it to start them: `World model:` = durable RULES of the environment that will "
    "still be true many actions from now (mechanics, what objects/colors mean, movement "
    "rules, win conditions) - never current positions or what just happened. `Goal model:` = "
    "what winning requires. `Action model:` = what each action does mechanically. `Recent "
    "findings:` = this turn's observations. `Open questions:` = unresolved hypotheses. "
    "`Plan:` = intended next steps. `Cross-level notes:` = knowledge that carries across "
    "levels."
)

_MEMORY_SECTION_GUIDANCE_SLIM = (
                "If you include assistant text before a tool call, use it to update your persistent memory sections. Section meanings: `World model:` = durable RULES of the environment that will still be true many actions from now (mechanics, what objects/colors mean, movement rules, win conditions) \u2014 never current positions, events, or what just happened. `Plan:` = intended next steps. Example \u2014 GOOD: `World model: Red cells are walls; each move shifts the player 2 cells; yellow keys open doors of the same color.` BAD: `World model: Batch stopped after 2 DOWNs, player now at (4,2).` (that is a passing event, not a rule). IMPORTANT: each labeled section you write REPLACES the stored version shown under 'Working world model carried from earlier turns' \u2014 write the COMPLETE updated section; any fact you omit is permanently lost."
)


def _wm_revise_nudges_enabled() -> bool:
    """The sticky 'LEVEL CHANGED / ATTEMPT RESET - revise your sections' hint.

    Built for an observed failure (the champion drifting without revising),
    but a later model maintained its sections diligently with no prompting at
    all, and the mechanism has needed two corrections since. Turning it off
    changes ONLY the pressure: sections are still parsed, stored and
    re-injected, so a null result means the nagging was not earning its
    prompt weight."""
    return not _memory_sections_disabled() and _get_env_bool(
        "ARC3_WM_REVISE_NUDGES", False
    )


def _wm_rebuild_nudge_enabled() -> bool:
    """The post-wipe "Rebuild them NOW: write fresh World model:, Goal model:,
    Action model:" demand, which repeats every turn until the model complies.

    An unmodified run wipes the sections silently - the carried block simply
    vanishes and reappears when the model next writes something. This nudge
    applies its strongest pressure exactly when the model has least to say: it
    has just lost its notes and has not yet explored the new level. Default off
    for that reason; the wipe itself is unaffected."""
    return not _memory_sections_disabled() and _get_env_bool(
        "ARC3_WM_REBUILD_NUDGE", False
    )


def _memory_sections_disabled() -> bool:
    """ARC3_MEMORY_SECTIONS=off removes the persistent-memory mechanic entirely:
    no carried block, no section guidance, no nudges, no parsing, and the
    world-model lines drop out of the system prompt too. The model is left to
    reason from history alone, which is what an unmodified run does before any
    of this machinery existed."""
    return os.environ.get("ARC3_MEMORY_SECTIONS", "").strip().lower() == "off"


def _memory_section_labels() -> tuple[str, ...] | None:
    """Which memory sections are advertised and carried, or None for all.

    'slim' keeps World model and Plan - the two the model actually maintains.
    Measured over 25 games: World model 351 writes, Plan 255, Open questions
    25, Recent findings 11, Action model 10, Goal model 9, Cross-level notes
    ZERO. The dropped sections are also the stale ones: a section written once
    and never revised keeps being re-injected as current, which was observed
    holding an answered question open for a whole game.

    'medium' additionally keeps Action model. It is written in a third of
    games and describes MECHANICS rather than transient state, so it ages far
    better than `Recent findings:` or `Open questions:` - on a game whose
    difficulty is working out what each control does, it is the section least
    worth dropping."""
    mode = os.environ.get("ARC3_MEMORY_SECTIONS", "").strip().lower()
    if mode == "slim":
        return _SLIM_MEMORY_SECTIONS
    if mode == "medium":
        return _MEDIUM_MEMORY_SECTIONS
    return None


def _message_display_text(message: dict[str, Any]) -> str:
    """Transcript rendering of a built user message. Content-parts messages
    carry their text in parts; images are noted rather than dumped."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    pieces: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "image_url":
            pieces.append("[image attached]")
        else:
            text = str(part.get("text") or "").strip()
            if text:
                pieces.append(text)
    return "\n\n".join(pieces)


def _wm_revise_hint(reason: str, already_shown: bool) -> str:
    """Text for the sticky revise-nudge. Full event text on first showing;
    accurate reminder wording on repeats (the event is no longer 'just now',
    and the adjacent step summary may correctly say the level is unchanged)."""
    if reason == "level":
        if not already_shown:
            return (
                "LEVEL CHANGED - your memory sections were KEPT, not wiped. Learned "
                "mechanics, action semantics, and the general goal usually carry over "
                "to the new level; the map/layout is new, and levels often introduce "
                "new elements or mechanics. On this first turn, scan the grid for "
                "unfamiliar elements (compare objects against your World/Action "
                "model), then revise your `World model:` and `Plan:` sections - "
                "update the map, keep what still holds, and note anything new."
            )
        return (
            "Reminder: a level change occurred earlier and you have STILL not "
            "revised your `World model:` or `Plan:` sections since. Do it now - "
            "update the map for this level, keep what still holds, and note "
            "anything new."
        )
    if reason == "death":
        if not already_shown:
            return (
                "ATTEMPT RESET - your memory sections were KEPT, not wiped. The "
                "level is unchanged, so your map and mechanics knowledge still "
                "hold. Amend your sections NOW with what this death taught you "
                "(the fatal action or sequence and its cause), and revise your "
                "`Plan:` to avoid repeating it."
            )
        return (
            "Reminder: this attempt was reset by an earlier death and you have "
            "STILL not amended your memory sections with what it taught you. Do "
            "it now - record the fatal action or sequence and its cause, and "
            "revise your `Plan:` to avoid repeating it."
        )
    return ""


def _suppress_board_dumps(text: str) -> tuple[str, int]:
    """Collapse runs of full-width board rows in tool output. Enforces the
    existing 'never print full boards' policy mechanically for models that
    ignore it: keeps the first rows of each run as orientation, replaces the
    rest with a stub, and appends a compute-don't-count nudge. Small crops
    (fewer than 6 rows) pass through untouched. ARC3_BOARD_DUMP_GUARD=0
    disables."""
    if not _get_env_bool("ARC3_BOARD_DUMP_GUARD", False):
        return text, 0

    def _board_like(line: str) -> bool:
        stripped = line.strip()
        return (
            len(stripped) >= 24
            and stripped.isalnum()
            and len(set(stripped)) <= 10
        )

    lines = text.split("\n")
    out: list[str] = []
    suppressed = 0
    i = 0
    while i < len(lines):
        if _board_like(lines[i]):
            j = i
            while j < len(lines) and _board_like(lines[j]):
                j += 1
            run = j - i
            if run >= 6:
                out.extend(lines[i:i + 2])
                out.append(f"[... {run - 2} full-width board rows suppressed ...]")
                suppressed += run - 2
            else:
                out.extend(lines[i:j])
            i = j
        else:
            out.append(lines[i])
            i += 1
    if suppressed:
        out.append(
            "[BOARD PRINT GUARD] Printing full boards is disallowed and "
            "character counting is unreliable for you. Compute over the grid "
            "in python (segmentation nodes, slicing, run-length checks, BFS) "
            "and print only coordinates, counts, and small crops."
        )
    return "\n".join(out), suppressed


def _memory_section_max_chars() -> int:
    """Cap on a single stored memory section, in characters.

    These sections are re-injected into EVERY later turn opener, so an
    oversized one is paid for repeatedly and grows the prompt without bound.
    The extraction path used to store them uncapped, which let a degenerate
    75k-character `World model:` into every subsequent prompt until the
    request exceeded the model's context entirely. Legitimate sections run a
    few hundred characters, so the default leaves ample room while making the
    runaway case impossible. 0 disables the cap (the old behaviour)."""
    return max(0, _get_env_int("ARC3_MEMORY_SECTION_MAX_CHARS", 8000))


def _degeneracy_ratio_threshold() -> float:
    """Compression ratio below which text is treated as degenerate.

    Measured on representative text: varied prose 0.20-0.28, a coordinate
    list 0.28, an ASCII board map 0.016, one sentence repeated 300 times
    0.008, the field mojibake spiral 0.0016. The default sits below the board
    map so a pasted grid is not falsely rejected, while still catching genuine
    repetition loops by a wide margin. Rejecting on compressibility catches
    those generally rather than one encoding artifact, and does not penalise
    legitimate unicode. 0 disables the check."""
    try:
        return max(0.0, float(os.environ.get("ARC3_DEGENERACY_RATIO", "0.01")))
    except ValueError:
        return 0.01


def _degeneracy_min_chars() -> int:
    """Length floor for the degeneracy check: short strings compress badly for
    uninteresting reasons, so testing them would produce false positives."""
    return max(1, _get_env_int("ARC3_DEGENERACY_MIN_CHARS", 500))


def _is_degenerate_text(value: str) -> bool:
    threshold = _degeneracy_ratio_threshold()
    if threshold <= 0:
        return False
    text = str(value or "")
    if len(text) < _degeneracy_min_chars():
        return False
    try:
        import zlib

        encoded = text.encode("utf-8", "replace")
        return len(zlib.compress(encoded, 6)) / max(1, len(encoded)) < threshold
    except Exception:
        return False


def _normalize_summary_text(value: Any, *, max_chars: int | None = 280) -> str:
    text = " ".join(str(value or "").split())
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars].rstrip()}... [{omitted} chars omitted]"


_HEADER_PARENTHETICAL_RE = re.compile(r"\([^)]*\)")
# Only SHORT decorations are stripped. A long parenthetical or emphasised aside
# means the line is a sentence rather than a header, and removing it could turn
# prose into a spurious match - so the header is left intact and simply fails to
# match, which loses the section rather than mis-capturing it.
_HEADER_DECORATION_MAX_CHARS = 30


def _normalize_section_header(head: str) -> str:
    """Reduce a candidate section header to its bare label.

    Models decorate these constantly. Observed in a single game: `Revised
    world model`, `World model update`, `World model updates`, `World model
    revision`, `Updated world model`, `World model (revised)`, `World model
    (L4)`, `Action model (from run 1)`, `Plan (19 moves)`, and `Revised world
    model - **important finding**`. Under exact matching, thirteen of
    twenty-seven memory writes in that game were silently discarded, including
    the one that correctly derived the level's kill rule.
    """
    def drop_short(match: "re.Match[str]") -> str:
        return " " if len(match.group(0)) <= _HEADER_DECORATION_MAX_CHARS else match.group(0)

    text = _HEADER_PARENTHETICAL_RE.sub(drop_short, head)
    # an emphasised trailing aside is dropped on the same terms; the em-dash is
    # deliberately NOT a cut point - it would discard text AFTER the label, which
    # can change what the header means, and it earned nothing in practice
    if "**" in text:
        head_part, _, rest = text.partition("**")
        if len(rest) <= _HEADER_DECORATION_MAX_CHARS:
            text = head_part
    text = text.replace("*", " ").replace("_", " ").replace("#", " ")
    # dangling dashes left behind by a removed decoration are punctuation, not
    # content: stripping them discards nothing, unlike cutting AT a dash
    return " ".join(text.split()).strip(" \u2014\u2013-")


def _tolerant_section_headers_enabled() -> bool:
    """Accept decorated section headers (`Updated world model:`, `Plan (19
    moves):`) rather than only the exact label.

    Doubled memory capture on a measured game (12 sections -> 24), which also
    doubles the carried world-model block the model is asked to rewrite in
    full each turn - so it is a candidate for longer generations, not only for
    better memory. Off restores exact matching."""
    return _get_env_bool("ARC3_TOLERANT_SECTION_HEADERS", False)


def _match_section_label(head: str, labels: list[str]) -> str | None:
    """The canonical label a header refers to, or None.

    Accepts the bare label, or the label with at most ONE extra word before or
    after it. One word rather than more is deliberate: it covers every
    decoration seen in practice while keeping prose from matching. Labels are
    tried longest-first so `Cross-level notes` is not shadowed by `Plan`."""
    tolerant = _tolerant_section_headers_enabled()
    # with the knob off, compare the RAW head: normalising would still strip
    # parentheticals and emphasis, which is itself part of the tolerance
    normalized = (
        _normalize_section_header(head).lower() if tolerant else " ".join(head.split()).lower()
    )
    if not normalized:
        return None
    if tolerant:
        # A trailing parenthetical is decoration, not part of the label. The
        # model versions and annotates its own sections - "World model
        # (revised)", "(major revision)", "World model v34 (level 3)" - and
        # those are the same section wearing a hat. Measured on one run: the
        # one-extra-word rule alone caught 89% of section writes, stripping a
        # trailing parenthetical first took it to 98%.
        #
        # Stripping rather than widening the word allowance to two: a
        # parenthetical is a bounded construct, whereas two free words start
        # matching prose. The 60-character cap in _extract_labeled_blocks is
        # applied to the UNDECORATED head, so a long decoration is rejected
        # before it can be stripped down to a match.
        without_paren = re.sub(r"\s*\([^)]*\)\s*$", "", normalized).strip()
        if without_paren:
            normalized = without_paren
    for label in sorted(labels, key=len, reverse=True):
        lowered = label.lower()
        if normalized == lowered:
            return label
        if not tolerant:
            continue
        prefix = lowered + " "
        if normalized.startswith(prefix) and len(normalized[len(prefix):].split()) == 1:
            return label
        suffix = " " + lowered
        if normalized.endswith(suffix) and len(normalized[: -len(suffix)].split()) == 1:
            return label
    return None


def _extract_labeled_blocks(content: str, labels: list[str]) -> dict[str, str]:
    extracted: dict[str, list[str]] = {label: [] for label in labels}
    current_label: str | None = None

    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        candidate = stripped
        while candidate.startswith(("-", "*")):
            candidate = candidate[1:].lstrip()

        matched_label: str | None = None
        inline_value = ""
        if ":" in candidate:
            head, _, rest = candidate.partition(":")
            # a header is a short line-initial phrase, not a sentence that
            # happens to contain a colon
            if len(head) <= 60:
                matched_label = _match_section_label(head, labels)
                if matched_label is not None:
                    inline_value = rest.strip()

        if matched_label is not None:
            current_label = matched_label
            if inline_value:
                extracted[current_label].append(inline_value)
            continue

        if current_label is not None and stripped:
            extracted[current_label].append(stripped)

    return {
        label: _normalize_summary_text(
            "\n".join(lines).strip(), max_chars=_memory_section_max_chars() or None
        )
        for label, lines in extracted.items()
        if "\n".join(lines).strip()
    }


def _opener_commit_hint() -> bool:
    """Ask the turn opener for a commitment, the way a resumption already does.

    Every piece of commitment pressure in the harness lives in the resumption
    path - "call action(actions) with your best next action or short batch",
    "Stop deliberating", "Act NOW". The fresh opener says nothing about acting
    at all. Measured across two runs, sequences following a resumption batch
    6.73 and 6.91 actions; those following a fresh opener batch 5.43 and 3.91,
    and the opener figure is what collapsed when resumptions became rarer.

    Deliberately asymmetric: probing stays available but has to be justified,
    while batching is the default. An earlier prompt that told the model to
    "re-issue actions one at a time" produced a 200-iteration single-step loop,
    so the wording avoids prescribing a size while naming the batch.
    """
    return _get_env_bool("ARC3_OPENER_COMMIT_HINT", False)


def _new_noop_wording() -> bool:
    """Whether the patch-110 rewordings are used.

    They replaced "changed nothing" with "no change inside the board area"
    across every guard message. Off restores the wording of the run that scored
    4.05, so the rewordings can be judged apart from the outcome line they
    shipped with."""
    return _get_env_bool("ARC3_NEW_CHANGED_PROMPTS", False)


def _animation_timeline_enabled() -> bool:
    """Whether the diff timeline is offered alongside the raw frames.

    The timeline is a cheap first look, but a summary that reads coherently
    invites stopping at it: on one level it produced a confident and wrong
    mechanic that the model only corrected after going to the frames. Off, the
    model computes its own comparison from `animation_frames` - a few lines it
    writes readily - at the cost of a snippet before it learns anything, which
    is wasted on animations that turn out to be irrelevant.

    Only an experiment settles which costs more.
    """
    return _get_env_bool("ARC3_ANIMATION_TIMELINE", True)


def _dedupe_multicall_line() -> bool:
    """Drop the duplicated multi-call permission and its stale caveat.

    Upstream states "you may call action() more than once" twice per opener,
    about thirty lines apart, the second time without the caveat - so the last
    word the model gets is the permission stripped of its qualifier.

    The caveat itself is now enforced three ways: the batch loop breaks on a
    terminal result and skips the rest, the per-snippet latch refuses the next
    action() call, and an executed action ends the turn so the next snippet
    always opens on a fresh board. On is the change; off keeps upstream.
    """
    return _get_env_bool("ARC3_DEDUPE_MULTICALL_LINE", False)


def _prefer_tool_calls() -> bool:
    """State the cost asymmetry: tool calls are free, actions are not.

    The prompt already says not to ration tool calls, but only ever compares
    them with other tool calls - never with reasoning, which is the trade the
    model actually makes when it writes twelve thousand characters of
    hypothesis before its first snippet. Observed: a turn spent 18,012
    characters reasoning, hit the output cap, and executed nothing.
    """
    return _get_env_bool("ARC3_PREFER_TOOL_CALLS", False)


# Three bands, a million apart, so no calculated value can cross between them.
# The maximum _game_priority is about 1,325.
#
#   ~2,000,000   never trimmed: the original prefix is whole and this game has
#                never paid a prefill, so it is the cheapest of all to continue
#   ~1,000,000   warm again: trimmed once, prefix rebuilt. Ordered within the
#                band by the estimate at that trim, so a game that was doing
#                well sits above one that was stalling
#      0..1,325  awaiting prefill: the trim invalidated the prefix and the next
#                request has to pay for it anyway, which is the one moment
#                yielding the slot is free
#
# The old scheme sent a flat 999999 while the prefix held, so every game tied at
# startup and the server broke ties by arrival. Subtracting the queue position
# gives a deterministic order instead.
# ---------------------------------------------------------------- TEMPORARY
# Diagnostic for "the backend never shows more than 13 concurrent requests with
# 14 games". Records where each game thread is and prints a census once a
# second, so the missing one can be located rather than guessed at.
# Remove once the cause is known.
_GATE_STATS = {"enqueued": 0, "blocked": 0, "blocked_ms": 0, "handover_ms": 0}
_DIAG_LOCK = threading.Lock()
_DIAG_STATE: dict[str, str] = {}
_DIAG_LAST = [0.0]


def _diag(agent: Any, where: str) -> None:
    if not _get_env_bool("ARC3_DIAG_CONCURRENCY", False):
        return
    name = getattr(agent, "_diag_name", "?")
    now = time.monotonic()
    with _DIAG_LOCK:
        _DIAG_STATE[name] = where
        if now - _DIAG_LAST[0] < 1.0:
            return
        _DIAG_LAST[0] = now
        counts: dict[str, int] = {}
        for value in _DIAG_STATE.values():
            counts[value] = counts.get(value, 0) + 1
        total = len(_DIAG_STATE)
        inflight = counts.get("request", 0)
        summary = "  ".join(
            f"{k}={v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])
        )
        log.warning(
            "DIAG %d games, %d in flight | %s | gate: %d handovers, %d blocked, "
            "%dms blocked, %dms total",
            total, inflight, summary,
            _GATE_STATS["enqueued"], _GATE_STATS["blocked"],
            _GATE_STATS["blocked_ms"], _GATE_STATS["handover_ms"],
        )
        if inflight < total:
            others = sorted(
                f"{g}:{w}" for g, w in _DIAG_STATE.items() if w != "request"
            )
            log.warning("DIAG   elsewhere: %s", ", ".join(others))


def _priority_refresh_enabled() -> bool:
    return any(_get_env_bool(name, False) for name in (
        "ARC3_PRIORITY_REFRESH_QUEUE", "ARC3_PRIORITY_PACE", "ARC3_PRIORITY_TAIL_FADE",
    ))


def _priority_action_count(summary: dict[str, Any]) -> int:
    return int(summary.get("end_action_num") or 0)


@lru_cache(maxsize=8)
def _load_pace_reference(path: str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text())
    values = [data["fallback_tokens"], *data["by_level"].values()]
    if not all(isinstance(v, (int, float)) and 0 < v < float("inf") for v in values):
        raise ValueError("Pace reference costs must be finite positive numbers")
    return data


def _pace_reference_tokens(level: int) -> float:
    path = os.environ.get("ARC3_PRIORITY_PACE_REFERENCE", "").strip()
    if not path:
        path = str(Path(__file__).with_name("priority_pace_reference.json"))
    reference = _load_pace_reference(path)
    return float(reference["by_level"].get(str(level), reference["fallback_tokens"]))


class _PriorityGate:
    """Harness-side admission control, for when the server ignores `priority`.

    The request field is a hint a backend may drop; this enforces the same
    ordering locally by limiting how many games hold a slot at once and
    choosing who gets a freed one.

    A game keeps its slot while its prefix is intact, because interrupting it
    there throws away a cache entry about to be reused. When the trimmer drops
    a message the prefix is invalid anyway and the next request must be
    prefilled from the divergence point - the one moment when yielding is free -
    so that is where the handover happens.

    Release and re-enqueue are one critical section. Releasing first would leave
    a window in which a lower-priority waiter takes the slot the caller was
    about to compete for.
    """

    def __init__(self, slots: int) -> None:
        self._cond = threading.Condition()
        self._free = max(1, int(slots))
        self._waiting: list[tuple[int, int]] = []
        self._seq = 0
        self._admitted: set[int] = set()
        self._snapshots: dict[int, PrioritySnapshot] = {}
        self._clock_start: float | None = None
        self._clock_deadline: float | None = None
        self._tail_fade_announced = False

    def configure_clock(self, started_at: float, deadline: float | None) -> None:
        """One shared monotonic clock, configured before initial admission.

        Earliest session start/deadline wins; appropriate for simultaneously
        launched sessions. No clock is inferred from the last action payload.
        """
        if _get_env_bool("ARC3_PRIORITY_TAIL_FADE", False) and deadline is None:
            raise ValueError("ARC3_PRIORITY_TAIL_FADE requires a finite runtime or soft deadline")
        with self._cond:
            self._clock_start = min(self._clock_start, started_at) if self._clock_start is not None else started_at
            if deadline is not None:
                self._clock_deadline = min(self._clock_deadline, deadline) if self._clock_deadline is not None else deadline

    def _snapshot_priority(self, snapshot: PrioritySnapshot, now: float) -> int:
        elapsed = max(0.0, now - self._clock_start) if self._clock_start is not None else 0.0
        endgame = _endgame_start_minutes() > 0 and elapsed >= 60 * _endgame_start_minutes()
        tail_fraction = None
        if _get_env_bool("ARC3_PRIORITY_TAIL_FADE", False):
            if self._clock_deadline is None or self._clock_start is None:
                raise ValueError("ARC3_PRIORITY_TAIL_FADE requires a finite runtime or soft deadline")
            fraction_setting = os.environ.get("ARC3_PRIORITY_TAIL_FADE_FRACTION", "").strip()
            if fraction_setting:
                try:
                    fraction = float(fraction_setting)
                except ValueError:
                    raise ValueError("ARC3_PRIORITY_TAIL_FADE_FRACTION must be > 0 and <= 1") from None
                if not 0.0 < fraction <= 1.0:
                    raise ValueError("ARC3_PRIORITY_TAIL_FADE_FRACTION must be > 0 and <= 1")
                # Use the full shared run duration, not the shrinking time left.
                window = fraction * max(0.0, self._clock_deadline - self._clock_start)
            else:
                window = max(1.0, 60 * _get_env_float("ARC3_PRIORITY_TAIL_FADE_MINUTES", 90.0))
            remaining = max(0.0, self._clock_deadline - now)
            tail_fraction = remaining / window if window > 0.0 else 0.0
            if remaining <= window:
                # Snapshot scoring also happens outside _pump's lock. Announce
                # once per shared gate, even when several games trim together.
                with self._cond:
                    if not self._tail_fade_announced:
                        self._tail_fade_announced = True
                        log.warning(
                            "tail fade phase after %.1f minutes: %.1f minutes "
                            "remaining; fading tail B to zero over the final "
                            "%.1f minutes, retaining hazard C",
                            elapsed / 60.0, remaining / 60.0, window / 60.0,
                        )
        return priority_value(
            snapshot, endgame=endgame, tail_fraction=tail_fraction,
            human_actions=_priority_human_actions(), action_scale=_priority_action_scale(),
            token_scale=_priority_token_scale(), action_weight=_priority_action_weight(),
            tail_base=_priority_tail_base(), tail_cap=_priority_tail_cap(),
            **_priority_level_options(),
        )

    def acquire(self, priority: int) -> None:
        with self._cond:
            self._enqueue_and_wait(priority)

    def release(self) -> None:
        with self._cond:
            self._free += 1
            self._pump()

    def handover(self, priority: int, snapshot: PrioritySnapshot | None = None) -> None:  # noqa: D401
        """Give up the slot and immediately compete for it again.

        The caller may win it straight back, which is the right outcome when
        nothing else is waiting or everything else is further along.
        """
        _t0 = time.monotonic()
        with self._cond:
            self._free += 1
            self._enqueue_and_wait(priority, snapshot)
        _GATE_STATS["handover_ms"] += int(1000 * (time.monotonic() - _t0))

    def _enqueue_and_wait(self, priority: int, snapshot: PrioritySnapshot | None = None) -> None:
        # TEMPORARY: with slots == games nobody should ever block here
        _GATE_STATS["enqueued"] += 1
        self._seq += 1
        token = self._seq
        if snapshot is not None:
            self._snapshots[token] = snapshot
        # negated so the heap pops the HIGHEST priority; the sequence number
        # breaks ties by arrival
        heapq.heappush(self._waiting, (-int(priority), token))
        self._pump()
        if token not in self._admitted:
            _GATE_STATS["blocked"] += 1
            t0 = time.monotonic()
            while token not in self._admitted:
                self._cond.wait()
            _GATE_STATS["blocked_ms"] += int(1000 * (time.monotonic() - t0))
        self._admitted.discard(token)

    def _pump(self) -> None:
        if _priority_refresh_enabled() and self._free > 0:
            now = time.monotonic()
            self._waiting = [
                (-self._snapshot_priority(self._snapshots[token], now), token)
                if token in self._snapshots else (priority, token)
                for priority, token in self._waiting
            ]
            heapq.heapify(self._waiting)
        woke = False
        while self._free > 0 and self._waiting:
            _, token = heapq.heappop(self._waiting)
            self._snapshots.pop(token, None)
            self._admitted.add(token)
            self._free -= 1
            woke = True
        if woke:
            self._cond.notify_all()


def _max_active_streams() -> int:
    """Games allowed to hold a slot at once. 0 disables the gate entirely.

    Below the game count this staggers the start - a slot only frees when a
    game's context is first trimmed, which measurements put 30 to 50 turns in -
    so early games run alone for a while and late ones begin much later. That is
    the point: a prefix is reused for as long as it lasts, instead of fourteen
    games each progressing at a fourteenth of the rate.
    """
    return max(0, _get_env_int("ARC3_MAX_ACTIVE_STREAMS", 0))


_PRIORITY_GATE: _PriorityGate | None = None
_PRIORITY_GATE_LOCK = threading.Lock()


def _priority_gate() -> _PriorityGate | None:
    """Process-wide, built on first use. Games run as threads in one process,
    so a module-level gate needs no plumbing and nothing to pickle."""
    global _PRIORITY_GATE
    slots = _max_active_streams()
    if slots <= 0:
        return None
    with _PRIORITY_GATE_LOCK:
        if _PRIORITY_GATE is None:
            _PRIORITY_GATE = _PriorityGate(slots)
            log.warning("priority gate active: %d concurrent streams", slots)
        return _PRIORITY_GATE


_PRIORITY_BAND = 1_000_000
_PRIORITY_UNTRIMMED_BASE = 2_000_000


def _priority_level_options() -> dict[str, Any]:
    # Keep boolean lookup settings backward compatible. The explicit
    # "remaining" mode selects the 8/7/5/0 tail, already at its final scale.
    lookup = os.environ.get("ARC3_PRIORITY_TAIL_LOOKUP", "").strip().lower()
    return {
        "normalize_score": _get_env_bool("ARC3_PRIORITY_SCORE_NORMALIZATION", False),
        "tail_lookup": (
            "remaining" if lookup == "remaining"
            else _get_env_bool("ARC3_PRIORITY_TAIL_LOOKUP", False)
        ),
        "tail_efficiency": _get_env_float("ARC3_PRIORITY_TAIL_EFFICIENCY", 0.8),
    }


def _priority_action_scale() -> float:
    return max(1.0, _get_env_float("ARC3_PRIORITY_ACTION_SCALE", 115.0))


def _priority_token_scale() -> float:
    return max(1.0, _get_env_float("ARC3_PRIORITY_TOKEN_SCALE", 62000.0))


def _priority_action_weight() -> float:
    """How much of `c` comes from the action curve rather than the token one.

    A blend rather than max(actions, tokens): max sits below both curves, so it
    demotes harder than either measure alone warrants. Replaying four runs
    against their own level-up costs, max was the weakest of every combination
    tried, while the weight itself barely mattered - 0.25 to 1.0 all landed
    within a few percent of each other.
    """
    return min(1.0, max(0.0, _get_env_float("ARC3_PRIORITY_ACTION_WEIGHT", 0.25)))


def _priority_tail_cap() -> float:
    """Ceiling on `b`, the value of every level after the current one.

    min(cap, level + 1) rather than a constant. Chaining the measured
    completion rates makes `b` nearly flat, which is why a constant looked
    right - but that chain prices expected future score, and what the scheduler
    needs is the term that lets depth outrank freshness. Replayed, a flat tail
    was the worst option tested at every constrained budget; the cap matters
    more than the slope, and a lower cap is better because it leaves more of
    the ordering to `a`.
    """
    return max(0.0, _get_env_float("ARC3_PRIORITY_TAIL_CAP", 4.0))


def _priority_tail_base() -> float:
    """Value of `b` at level 1, before the per-level rise.

    b = min(cap, base + level - 1). Base 2 with cap 4 gives the fitted form,
    2 3 4 4 4...; base 6 with cap 6 gives a flat tail, which is what this
    replaced and is worth being able to get back for a comparison.
    """
    return max(0.0, _get_env_float("ARC3_PRIORITY_TAIL_BASE", 2.0))


def _priority_human_actions() -> float:
    """Denominator offset in the efficiency term.

    The benchmark scores a level by min(1, (human/used)^2). The level is not
    finished yet, so the term belongs at the expected completion point rather
    than the current count - hence an offset instead of a bare division.
    """
    return max(1.0, _get_env_float("ARC3_PRIORITY_HUMAN_ACTIONS", 25.0))


def _endgame_start_minutes() -> float:
    """Wall-clock minutes from a game's start before it switches phase.

    0 disables the endgame phase entirely.
    """
    return max(0.0, _get_env_float("ARC3_ENDGAME_START_MINUTES", 0.0))


# Endgame priorities sit in the upper half of whichever warmth band the game is
# in: the cache argument for the bands is orthogonal to the phase, so a game
# about to pay a prefill still yields to one that has a warm prefix. The widest
# value either phase can produce is under 2,000 even at level 15, so half a
# band is ample.
_PRIORITY_ENDGAME_OFFSET = 500_000


def _game_priority(
    level: int,
    actions_on_level: int,
    tokens_on_level: int = 0,
    endgame: bool = False,
    total_levels: int | None = None,
) -> int:
    """Expected remaining score for a game on `level` after this much spend.

    Two phases. In the main one, (a + b) * c:

      a  what finishing THIS level banks - the score becomes `level` levels
         completed, discounted by the benchmark's efficiency term
      b  every level after it, min(cap, base + level - 1)
      c  whether this level still yields anything, as a blend of two measured
         decay curves

    In the endgame there is no time to reach a further level, so the tail and
    the hazard stop meaning anything and what remains is simply what completing
    this level would bank. The 0.25 floor lets efficiency separate games at the
    same depth without letting a shallower one overtake a deeper one.

    Replaying four runs against their own level-up costs, the endgame switch was
    worth 8 to 12% of total score at constrained budgets and nothing at all once
    the budget covered what the games actually spent - which is where real runs
    sit, so this matters only if the competition budget is tighter.
    """
    level = max(1, int(level))
    actions = max(0, int(actions_on_level))
    tokens = max(0, int(tokens_on_level))
    options = _priority_level_options()
    if options["normalize_score"] or options["tail_lookup"]:
        # Same scoring as the refreshed queue, retaining this legacy path's
        # phase offset. With both flags off, execute the original path below.
        value = priority_value(
            PrioritySnapshot(level, actions, tokens, total_levels=total_levels),
            endgame=endgame, human_actions=_priority_human_actions(),
            action_scale=_priority_action_scale(), token_scale=_priority_token_scale(),
            action_weight=_priority_action_weight(), tail_base=_priority_tail_base(),
            tail_cap=_priority_tail_cap(), **options,
        )
        return value + (_PRIORITY_ENDGAME_OFFSET if endgame else 0)
    human = _priority_human_actions()
    efficiency = min(1.0, (human / (human + actions)) ** 2)

    if endgame:
        value = level * (0.25 + efficiency)
        return _PRIORITY_ENDGAME_OFFSET + max(1, int(value * 100))

    a = level * efficiency
    b = min(_priority_tail_cap(), _priority_tail_base() + level - 1.0)
    weight = _priority_action_weight()
    by_actions = actions / _priority_action_scale()
    by_tokens = tokens / _priority_token_scale()
    c = (
        weight * 0.5 ** (by_actions * by_actions)
        + (1.0 - weight) * 0.5 ** (by_tokens * by_tokens)
    )
    return max(1, int((a + b) * c * 100))


def _persistent_functions_scope() -> str:
    """How long a kept helper lives: 'levelup' (default), 'turn', or 'game'.

    'turn' clears at every analyzer invocation that is not a resumption, so a
    turn that inspects, acts and yields loses its toolkit even though the
    investigation continues. That is conservative but costly: the helpers the
    keepability check lets through are pure transforms taking the board as an
    argument, and those do not care that an action happened.

    'levelup' keeps them until the level changes, which is where a helper can
    genuinely go wrong - not by holding stale data, which the check already
    prevents, but by encoding an assumption in its code, like a glyph parser
    with row offsets from the previous layout.

    'game' keeps helpers across turns and level transitions. The model is
    reminded to recheck level-specific assumptions. New game sessions still
    start with an empty pool; snippet variables remain ephemeral in every scope.
    """
    scope = os.environ.get("ARC3_PERSISTENT_FUNCTIONS_SCOPE", "").strip().lower()
    return scope if scope in ("turn", "levelup", "game") else "levelup"


def _repeat_state_guard_enabled() -> bool:
    """Refuse an action this snippet has already run from this exact board.

    Buggy loop code can spend a whole life bar getting nowhere: a cursor that
    wraps turns RIGHT into a five-state cycle, and `while pos < j: LEFT` walks
    away from its target forever. The terminal latch does stop these, but only
    once the game ends - 20 to 30 actions later.

    Keyed on (board, action) rather than on repetition counts, because the
    loops mostly alternate: LEFT, RIGHT, LEFT, RIGHT defeats a same-action rule
    but revisits its first state on the third action.

    Per snippet, and no override slot. A repeat ACROSS snippets is the model
    choosing, having seen the board and the results; a repeat inside one is
    code executing a plan fixed before any of it ran, which is the only case
    that can be a bug the model has not had a chance to notice. It is also not
    reading the refusal - its code is running - so writing a new snippet is the
    response, and that clears the map.
    """
    return _get_env_bool("ARC3_REPEAT_STATE_GUARD", False)


def _guards_active(level: Any) -> bool:
    """Whether the per-action guards apply at this level.

    Level 1 pays the smallest share of the score - one part in N - and it is
    where the model knows least, so it probes hardest: repeating an action to
    see whether the repeat differs, retrying what killed it to learn what
    killed it. The guards read that as waste. ARC3_GUARDS_FROM_LEVEL=2 lets
    level 1 run unguarded.

    Covers the stale block and both repeat guards. NOT the batch no-op stop,
    which costs one action of a planned sequence rather than refusing a
    deliberate probe, and not the terminal latch, which stops a snippet acting
    against a board that no longer exists.
    """
    try:
        floor = max(1, int(os.environ.get("ARC3_GUARDS_FROM_LEVEL", "1") or 1))
    except ValueError:
        floor = 1
    try:
        return int(level) >= floor
    except (TypeError, ValueError):
        return True  # unknown level: guard rather than not


def _persistent_functions() -> bool:
    """Retain eligible function sources across calls; off by default.

    Scope is controlled by ARC3_PERSISTENT_FUNCTIONS_SCOPE. State and variables
    are never serialized. Definitions are validated and recreated in each fresh
    sandbox, against its current harness globals.
    """
    return _get_env_bool("ARC3_PERSISTENT_FUNCTIONS", False)


def _animation_enabled() -> bool:
    """Whether intermediate animation frames are captured and offered.

    Off by default: this is new information for the model rather than a fix to
    something broken, so it needs to be measurable on its own."""
    return _get_env_bool("ARC3_ANIMATION", False)


def _report_gameplay_changed() -> bool:
    """Tell the model WHICH kind of change an action produced.

    The harness computes two flags - `board_changed` over the whole frame and
    `gameplay_changed` over the interior only - but the turn opener reports the
    first and the system prompt names only the first. On a game with a
    remaining-steps bar the whole frame differs after every action, so the
    opener says a change occurred every time and hedges: "verify that it
    affected gameplay objects rather than only HUD elements". Meanwhile the
    per-action trace, built from the second flag, can say the same action
    changed nothing - so the two lines of the same message contradict each
    other, and on a single-action turn only the misleading one appears.

    On (default) the opener reports the three states that actually occur and
    the system prompt explains both flags. The wording stays geometric -
    "cells inside the board area" rather than "gameplay objects" - because the
    test is a border crop, not a semantic one."""
    return _get_env_bool("ARC3_REPORT_GAMEPLAY_CHANGED", False)


def _explain_gameplay_changed() -> bool:
    """Whether the system prompt explains the two change flags.

    Split from the per-turn line because the two halves have very different
    reach and very different arguments. The sentence is paid once and states
    something the model cannot work out for itself: `last_action_call_result`
    exposes both flags and nothing says what distinguishes them, so a game with
    a timer bar makes `board_changed` true after every action.

    The opener line is paid on every executing turn - on one measured run,
    thousands of times against thirteen batch-guard firings - and only
    summarises a comparison the model can make with frame_diff. It is also the
    half with evidence against it: the one run that carried it scored 2.30 on
    competition where the run without it scored 4.05, though that run changed
    guard wording at the same time.

    Defaults to following ARC3_REPORT_GAMEPLAY_CHANGED so the pair behaves as
    before unless it is set explicitly.
    """
    raw = os.environ.get("ARC3_EXPLAIN_GAMEPLAY_CHANGED", "").strip()
    if not raw:
        return _report_gameplay_changed()
    return raw.lower() not in ("0", "false", "no", "off")


def _alias_respects_stored() -> bool:
    """Whether `Hypothesis:` may replace a world model that already exists.

    The alias condition checks only THIS reply's extraction: a reply carrying a
    hypothesis and no `World model:` has its hypothesis stored as the world
    model, overwriting whatever was there. That is right on a first turn and
    destructive later - `Hypothesis` connotes untested, so a speculation
    silently replaces tested content. Measured on 25 games: 7 such overwrites
    against 351 world-model writes with tolerant headers on, 16 against 311 in
    an unpatched run.

    On (default) the aliases apply only when the STORED value is also empty.
    Off restores the original behaviour."""
    return _get_env_bool("ARC3_ALIAS_RESPECTS_STORED", True)


def _extract_scientist_note(
    content: str, stored: dict[str, Any] | None = None
) -> dict[str, str]:
    if not content.strip():
        return {}
    extracted = _extract_labeled_blocks(
        content,
        [
            "World model",
            "Goal model",
            "Action model",
            "Recent findings",
            "Open questions",
            "Plan",
            "Cross-level notes",
            "Hypothesis",
            "History check",
            "Next test",
        ],
    )
    result = {
        "world_model": extracted.get("World model", ""),
        "goal_model": extracted.get("Goal model", ""),
        "action_model": extracted.get("Action model", ""),
        "recent_findings": extracted.get("Recent findings", ""),
        "open_questions": extracted.get("Open questions", ""),
        "current_plan": extracted.get("Plan", ""),
        "cross_level_notes": extracted.get("Cross-level notes", ""),
    }
    respect_stored = _alias_respects_stored()
    existing = stored or {}

    def apply_alias(key: str, label: str) -> None:
        if result[key]:
            return
        if respect_stored and str(existing.get(key) or "").strip():
            return
        result[key] = extracted.get(label, "")

    apply_alias("world_model", "Hypothesis")
    apply_alias("recent_findings", "History check")
    apply_alias("current_plan", "Next test")
    return result


def _empty_world_model() -> dict[str, str]:
    return {
        "world_model": "",
        "goal_model": "",
        "action_model": "",
        "recent_findings": "",
        "open_questions": "",
        "current_plan": "",
        "cross_level_notes": "",
    }


def _request_tool_choice(tools: list[dict[str, Any]] | None) -> str | None:
    """Tool-choice sent with tool-bearing requests. LOCAL_ANALYZER_TOOL_CHOICE
    overrides the default "auto": set "omit"/"none" to drop the field for
    servers whose validation rejects "auto" without a tool-call parser (tools
    are still sent, so chat templates still render tool instructions); any
    other value (e.g. "required") passes through verbatim."""
    if not tools:
        return None
    choice = os.environ.get("LOCAL_ANALYZER_TOOL_CHOICE", "auto").strip().lower()
    if choice in ("", "omit", "none"):
        return None
    return choice


def _trim_log_text(text: str, *, max_chars: int = _RESPONSE_META_MAX_CHARS) -> str:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    omitted = len(stripped) - max_chars
    return f"{stripped[:max_chars].rstrip()}\n... [truncated {omitted} chars]"


def _format_model_response_meta(
    *,
    finish_reason: str,
    served_by: str,
    reasoning: str,
    content: str,
    tool_calls: list[dict[str, Any]],
    tool_call_markup_in_text: bool,
    recovered_tool_calls_from_markup: bool,
    malformed_argument_errors: list[str],
) -> str:
    lines = [
        f"finish_reason: {finish_reason or '(empty)'}",
        f"served_by: {served_by or '(unknown)'}",
        f"tool_call_count: {len(tool_calls)}",
        f"content_chars: {len(content)}",
        f"reasoning_chars: {len(reasoning)}",
        f"tool_call_markup_in_text: {'yes' if tool_call_markup_in_text else 'no'}",
        f"tool_calls_recovered_from_markup: {'yes' if recovered_tool_calls_from_markup else 'no'}",
    ]
    if malformed_argument_errors:
        lines.append("tool_call_argument_issues:")
        lines.extend(f"- {issue}" for issue in malformed_argument_errors)
    if tool_calls:
        lines.append("raw_tool_calls:")
        lines.append(_trim_log_text(json.dumps(tool_calls, indent=2, ensure_ascii=True)))
    return "\n".join(lines)


def _estimate_diff_tokens(text: str) -> int:
    return len(text) // 3


def _diff_entry_line(bucket: str, entry: dict[str, Any]) -> str:
    color = entry.get("color")
    pixels = entry.get("pixels")
    if bucket == "moved":
        line = f"moved: color {color} {pixels}px {entry.get('from')}->{entry.get('to')}"
        if entry.get("rotated_by"):
            line += f" rotated_by {entry['rotated_by']}"
        return line
    if bucket == "rotated":
        return (
            f"rotated: color {color} {pixels}px by {entry.get('rotated_by')} "
            f"at {entry.get('at')}"
        )
    if bucket == "changed_color":
        return (
            f"recolored at {entry.get('at')}: color "
            f"{entry.get('before_color')}->{entry.get('after_color')} "
            f"({entry.get('pixels')}px)"
        )
    if bucket == "resized":
        return (
            f"resized: color {entry.get('color')} "
            f"{entry.get('before_pixels')}->{entry.get('after_pixels')}px "
            f"at {entry.get('at_before')}"
        )
    return f"{bucket}: color {color} {pixels}px at {entry.get('at')}"


_DIFF_BUCKETS = ("moved", "rotated", "changed_color", "resized", "appeared", "disappeared")


def _render_auto_frame_diff(
    diff: dict[str, Any], budget_tokens: int, total_cells: int | None = None
) -> list[str]:
    """Render a frame diff as prompt lines, degrading to stay under budget:
    full listing -> aggregated by (bucket, color, pixels) -> counts only.
    Resized entries covering most of the grid (the background reshaping as
    collateral of other objects moving) are elided as noise; the callable
    frame_diff() keeps them."""
    if total_cells:
        resized = diff.get("resized") or []
        kept = [e for e in resized if (e.get("before_pixels") or 0) * 2 < total_cells]
        if len(kept) != len(resized):
            diff = dict(diff)
            diff["resized"] = kept
    count = diff.get("changed_cell_count")
    if count == -1:
        return [f"grid shape changed ({diff.get('before_shape')} -> {diff.get('after_shape')}); per-object diff skipped."]
    header = f"{count} cells changed."
    # rung 1: full listing
    full = [header]
    for bucket in _DIFF_BUCKETS:
        for entry in diff.get(bucket) or []:
            full.append("- " + _diff_entry_line(bucket, entry))
    if _estimate_diff_tokens("\n".join(full)) <= budget_tokens:
        return full
    # rung 2: aggregate identical entries, largest objects first, capped
    lines = [header]
    for bucket in _DIFF_BUCKETS:
        entries = list(diff.get(bucket) or [])
        if not entries:
            continue
        entries.sort(key=lambda e: -(e.get("pixels") or e.get("after_pixels") or 0))
        groups: dict[tuple, int] = {}
        for entry in entries:
            key = (entry.get("color"), entry.get("pixels"))
            groups[key] = groups.get(key, 0) + 1
        shown = 0
        for entry in entries:
            if shown >= 3:
                break
            key = (entry.get("color"), entry.get("pixels"))
            if groups.get(key, 0) > 3:
                continue
            lines.append("- " + _diff_entry_line(bucket, entry))
            shown += 1
        swarm = {k: n for k, n in groups.items() if n > 3}
        for (color, pixels), n in sorted(swarm.items(), key=lambda kv: -kv[1]):
            lines.append(f"- {bucket}: {n}x color {color} {pixels}px (positions elided)")
        remaining = len(entries) - shown - sum(swarm.values())
        if remaining > 0:
            lines.append(f"- {bucket}: +{remaining} more")
    lines.append("(call frame_diff() in python for the full object list)")
    if _estimate_diff_tokens("\n".join(lines)) <= budget_tokens:
        return lines
    # rung 3: counts only, cannot be large
    parts = [
        f"{len(diff.get(bucket) or [])} {bucket}"
        for bucket in _DIFF_BUCKETS
        if diff.get(bucket)
    ]
    return [
        header + " " + ", ".join(parts) + " (details omitted; call frame_diff() in python)."
    ]


def _build_system_prompt(*, tool_output_tokens: int) -> str:
    prompt = "You are a coding agent solving a grid-based puzzle game."
    # ARC3_SYSTEM_PROMPT_PREFIX (read at call time): prepended verbatim as the
    # first line(s) of the system prompt. Intended for model-specific control
    # directives that live in the system prompt, e.g. Nemotron's "/think"
    # reasoning toggle. Empty/unset leaves the prompt unchanged.
    _prefix = os.environ.get("ARC3_SYSTEM_PROMPT_PREFIX", "").strip()
    if _prefix:
        prompt = f"{_prefix}\n\n{prompt}"
    prompt += GAME_OVERVIEW_ADDENDUM
    prompt += STRUCTURED_RUNTIME_STATE_ADDENDUM
    if _explain_gameplay_changed():
        prompt += GAMEPLAY_CHANGED_ADDENDUM.format(
            border=f"{_get_env_int('ARC3_NOOP_GUARD_BORDER', 4)} cells"
        )
    if _animation_enabled():
        prompt += ANIMATION_ADDENDUM
        if _animation_timeline_enabled():
            prompt += ANIMATION_ADDENDUM_TIMELINE
    if current_grid_image_enabled():
        prompt += MULTIMODAL_CONTEXT_ADDENDUM
    if _AUTO_FRAME_DIFF:
        prompt += (
            " Each turn's message includes an 'Auto frame diff' section summarizing "
            "object-level changes since your previous analyzer turn (it spans all "
            "actions executed in between, not just the last one). Treat it as ground "
            "truth for what changed, consult it before acting, and call `frame_diff()` "
            "in python when you need per-action or full object detail."
        )
    if _DIFF_IMAGE and current_grid_image_enabled():
        prompt += (
            " Alongside the current grid image you receive a diff image: cells that "
            "changed since your previous analyzer turn are shown in their new true "
            "color; unchanged cells are dimmed to a dark navy that is not a game "
            "color. Use it to locate change at a glance."
        )
    prompt += VISUAL_GAME_ADDENDUM
    if _get_env_bool("ARC3_ACTION_INFO", False):
        # The action-info section replaces the earlier coordinate-only guidance.
        prompt = prompt.replace(
            "- For `MOUSE`, pass `row` and `col` integer arguments. `row` is vertical position, `col` is horizontal position.\n",
            "",
        )
        prompt += ACTION_INFO_ADDENDUM
        if undo_exposure_mode() == "on":
            prompt += UNDO_INFO_ADDENDUM
        if reset_exposed():
            prompt += RESET_INFO_ADDENDUM
    prompt += (
        PYTHON_ADDENDUM_HEAD.replace(
            "- Every `python` tool call starts fresh. Re-import modules or re-define any custom utility logic you need.\n",
            "- Python variables reset between tool calls. Re-import modules as needed; "
            "eligible functions are retained as described in the tool session rules below.\n",
        ) if _persistent_functions() else PYTHON_ADDENDUM_HEAD
    )
    prompt += (
        WORLD_MODEL_FREE_ADDENDUM if _memory_sections_disabled() else WORLD_MODEL_ADDENDUM
    )
    prompt += PYTHON_ADDENDUM_TAIL
    if _get_env_bool("ARC3_FRAME_DIFF_HINT", False):
        # documents `frame_diff(before, after)`. The function stays callable
        # either way - this only controls whether the model is told about it,
        # so the arm measures whether knowing costs more than it returns.
        prompt += FRAME_DIFF_HINT_ADDENDUM
    if _get_env_bool("ARC3_STEP_VERIFICATION_HINT", False):
        # General checks during action sequences and targeted probes when search fails.
        prompt += STEP_VERIFICATION_ADDENDUM
    # only describe the guards that are actually armed: prompt weight spent on a
    # mechanism that cannot fire is weight the model reads and cannot use, and an
    # exception it is told about but never sees is worse than silence
    # The guards are no longer described up front. Every claim the addenda made -
    # not executed, nothing spent, the batch is checked throughout, how to
    # override - is repeated in the refusal message itself, which arrives with
    # the concrete action attached and only when it is relevant. Describing a
    # mechanism the model may never meet is prompt weight spent on nothing.
    prompt += COMPACT_TOOL_SESSION_ADDENDUM.format(
        tool_output_tokens=tool_output_tokens,
        prefer_tool_calls=PREFER_TOOL_CALLS_LINE if _prefer_tool_calls() else "",
        persistent_functions=(
            PERSISTENT_FUNCTIONS_LINE.format(lifetime=(
                "for later calls throughout this game, including across level changes; "
                "they are cleared when a new game starts"
                if _persistent_functions_scope() == "game" else
                "for later calls on the same level; they are cleared when the level changes"
                if _persistent_functions_scope() == "levelup" else
                "for later calls in this analyzer turn, including yield resumptions; "
                "they are cleared at the start of the next turn"
            ), import_guidance=(
                "Supported explicit top-level imports used by retained functions are restored "
                "with them; imports inside a function also work."
                if _get_env_bool("ARC3_PERSISTENT_FUNCTIONS_IMPORTS", False) else
                "Put needed imports inside the function."
            )) if _persistent_functions() else EPHEMERAL_FUNCTIONS_LINE
        ),
    )
    if _level_transfer_guidance_enabled():
        prompt = apply_level_transfer_system_guidance(prompt)
    return prompt


@dataclass(frozen=True)
class AnalyzerModelConfig:
    provider: str
    base_url: str
    model_id: str


@dataclass(frozen=True)
class AnalyzerTurnResult:
    step_executed: bool
    retryable_failure: bool = False
    reasoning: str = ""
    yielded_control: bool = False


@dataclass(frozen=True)
class _ToolDispatchResult:
    content: str
    step_executed: bool = False
    # True when an executed action changed the board outside the HUD border, or
    # ended the level or the attempt. "Executed" alone is too weak a signal for
    # the reasoning-effort ladder: bumping a wall executes and achieves nothing,
    # and resetting on it lets the model oscillate between truncating at full
    # effort and acting inertly at reduced effort, never settling.
    made_progress: bool = False


@dataclass(frozen=True)
class _AsciiFrameView:
    ascii: str
    step: int
    level: int
    shape: tuple[int, int]

    def __str__(self) -> str:
        rows, cols = self.shape
        return f"AsciiFrameView(level={self.level}, step={self.step}, shape={rows}x{cols})"

    __repr__ = __str__


@dataclass(frozen=True)
class _AsciiHistoryEntryView:
    action: str
    frame: _AsciiFrameView
    result: dict[str, Any]

    def __str__(self) -> str:
        return f"AsciiHistoryEntryView(action={self.action!r}, frame={self.frame})"

    __repr__ = __str__


def _to_ascii_frame_view(frame: Frame | None) -> _AsciiFrameView | None:
    if frame is None:
        return None
    return _AsciiFrameView(
        ascii=frame.ascii,
        step=frame.step,
        level=frame.level,
        shape=frame.shape,
    )


def _to_ascii_history_views(history_entries: list[HistoryEntry]) -> list[_AsciiHistoryEntryView]:
    views: list[_AsciiHistoryEntryView] = []
    for entry in history_entries:
        frame_view = _to_ascii_frame_view(entry.frame)
        if frame_view is None:
            continue
        views.append(_AsciiHistoryEntryView(
            action=entry.action, frame=frame_view,
            result=dict(getattr(entry, "result", {}) or {}),
        ))
    return views


def _ascii_frame_view_payload(frame: Frame | None) -> dict[str, Any] | None:
    view = _to_ascii_frame_view(frame)
    if view is None:
        return None
    return {
        "ascii": view.ascii,
        "step": view.step,
        "level": view.level,
        "shape": [int(view.shape[0]), int(view.shape[1])],
        "grid": [list(row) for row in frame.grid],
    }


def _ascii_history_view_payload(history_entries: list[HistoryEntry]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for entry in history_entries:
        frame_payload = _ascii_frame_view_payload(entry.frame)
        if frame_payload is None:
            continue
        payload.append({
            "action": entry.action, "frame": frame_payload,
            "result": dict(getattr(entry, "result", {}) or {}),
        })
    return payload


_VISION_PATCH_PIXELS = 32          # Qwen3.6: 16px patches, 2x2 merge
_VISION_SENTINEL_TOKENS = 2        # vision start/end markers
_IMAGE_TOKENS_FALLBACK = 402       # 64-cell board at MULTIMODAL_UPSCALE=10
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _text_token_chars() -> int:
    """Characters per token for the text heuristic - the SEED value only.

    Measured against a server-reported prompt_tokens on a real request: 25,189
    characters of system prompt plus opener tokenised to 5,892, i.e. about 4.3
    characters per token. The 3 default therefore over-counts text by roughly
    40%, so the trimmer evicts history the budget could have held. This value
    is used until the first response supplies a measurement; after that
    ARC3_CALIBRATE_TEXT_TOKENS drives the divisor from the server's own
    figures."""
    return max(1, _get_env_int("ARC3_TEXT_TOKEN_CHARS", 3))


def _calibrate_text_tokens() -> bool:
    """Derive the characters-per-token divisor from the server's reported
    prompt_tokens rather than a fixed guess.

    Every response carries prompt_tokens for the request that produced it - an
    exact count including the chat template. Subtracting the vision cost of the
    request's images leaves the text, and dividing by the text characters gives
    the divisor for the next estimate. No extra requests, no tokenizer, and it
    tracks content: code-heavy turns tokenise differently from prose.

    The last measurement is used rather than an average: history carries
    forward, so the prompt just measured is mostly the prompt about to be
    built, and averaging would blend in content already evicted."""
    return _get_env_bool("ARC3_CALIBRATE_TEXT_TOKENS", True)


# Asymmetric on purpose. Over-counting trims history and degrades one game;
# under-counting overflows the context, and an overflowing request holds a
# slot and can pressure the others. Measured across 22 games: 2.75-3.69,
# mean 3.12. The 3.3 ceiling therefore binds on the sparsest games by
# design - a deliberate haircut - while the floor is set low enough that
# only an arithmetic failure reaches it, and such a game is likely lost
# anyway.
def _text_token_chars_min() -> float:
    """Floor for the calibrated divisor. Low enough that only an arithmetic
    failure reaches it, and a game in that state is likely lost anyway."""
    # a divisor of zero would make every estimate infinite and trim the whole
    # history on the first request
    return max(0.1, _get_env_float("ARC3_TEXT_TOKEN_CHARS_MIN", 1.0))


def _text_token_chars_max() -> float:
    """Ceiling for the calibrated divisor.

    Exposed because 3.3 is a fitted constant, not a property of the tokeniser:
    it came from 22 games measuring 2.75-3.69, so it binds on the sparsest
    boards by design. A run whose prompts are denser than that sample - a
    different game set, a longer system prompt, another model - is being
    trimmed harder than its own measurements justify, and raising this is how
    that is tested.
    """
    # never below the floor: an inverted pair would clamp every measurement to
    # the ceiling and silently invert the guard's meaning
    return max(_text_token_chars_min(),
               _get_env_float("ARC3_TEXT_TOKEN_CHARS_MAX", 3.3))


def _image_token_estimate_enabled() -> bool:
    """Count image parts by their real vision-token cost instead of by the
    length of their base64 payload. Without this an image is charged roughly
    one token per three base64 characters - thousands for a board render whose
    true cost is a few hundred - so enabling images silently evicts real
    history."""
    return _get_env_bool("ARC3_IMAGE_TOKEN_ESTIMATE", True)


def _png_dimensions(data_url: str) -> tuple[int, int] | None:
    """Width/height from a base64 PNG data URL, read from the IHDR header
    without decoding the image. Only the first 24 bytes are needed."""
    marker = "base64,"
    index = data_url.find(marker)
    if index < 0:
        return None
    head = data_url[index + len(marker):index + len(marker) + 32]
    if len(head) < 32:
        return None
    try:
        raw = base64.b64decode(head, validate=True)
    except Exception:
        return None
    if len(raw) < 24 or not raw.startswith(_PNG_SIGNATURE) or raw[12:16] != b"IHDR":
        return None
    width = int.from_bytes(raw[16:20], "big")
    height = int.from_bytes(raw[20:24], "big")
    if width <= 0 or height <= 0:
        return None
    return width, height


def _image_part_tokens(part: dict[str, Any]) -> int:
    """Vision tokens for one image content part: one token per merged patch
    plus the two sentinel tokens. ARC3_IMAGE_TOKENS_FLAT overrides the
    computation (useful for non-Qwen backends) and is also the fallback when
    the header cannot be read."""
    flat = _get_env_int("ARC3_IMAGE_TOKENS_FLAT", 0)
    if flat > 0:
        return flat
    url = ""
    image_url = part.get("image_url")
    if isinstance(image_url, dict):
        url = str(image_url.get("url") or "")
    dims = _png_dimensions(url) if url else None
    if dims is None:
        return _IMAGE_TOKENS_FALLBACK
    width, height = dims
    cells = -(-width // _VISION_PATCH_PIXELS) * -(-height // _VISION_PATCH_PIXELS)
    return cells + _VISION_SENTINEL_TOKENS


def _split_messages_for_estimate(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Return (messages with image payloads replaced by a short placeholder,
    total vision tokens). Originals are never mutated."""
    image_tokens = 0
    rewritten: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            rewritten.append(message)
            continue
        parts: list[Any] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                image_tokens += _image_part_tokens(part)
                parts.append({"type": "image_url", "image_url": {"url": "<image>"}})
            else:
                parts.append(part)
        rewritten.append({**message, "content": parts})
    return rewritten, image_tokens


def _estimate_tokens(value: Any, chars_per_token: float | None = None) -> int:
    try:
        # ensure_ascii=False is deliberate: with escaping ON, every non-ASCII
        # character renders as a six-character \uXXXX sequence, so any prompt
        # containing arrows, box-drawing glyphs or a multiplication sign is
        # counted at up to 6x its real size. The trimmer decides when to evict
        # in these units, so the inflation makes it discard history it did not
        # need to - and the error grows with the share of non-ASCII text.
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        rendered = str(value)
    chars = chars_per_token or _text_token_chars()
    return max(1, int(len(rendered) / chars) + (1 if len(rendered) % chars else 0))


def _host_accessible_base_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    hostname = (parsed.hostname or "").strip().lower()
    if hostname != "host.docker.internal":
        return base_url
    netloc = "127.0.0.1"
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    return urlunparse(parsed._replace(netloc=netloc))


def _resolve_analyzer_model(model: str) -> AnalyzerModelConfig:
    requested = (model or "").strip()
    lowered = requested.lower()
    if lowered in {"local", "local-qwen", "qwen-local", "qwen"}:
        configured_base_url = os.environ.get("LOCAL_ANALYZER_BASE_URL", _LOCAL_ANALYZER_BASE_URL).strip()
        if not configured_base_url:
            raise ValueError("LOCAL_ANALYZER_BASE_URL must be set for the local analyzer preset.")

        provider = os.environ.get("LOCAL_ANALYZER_PROVIDER", os.environ.get("OPENAI_PROVIDER", "vllm")).strip().lower()
        if not provider:
            provider = "vllm"
        model_id = os.environ.get("LOCAL_ANALYZER_MODEL_ID", "").strip() or _LOCAL_ANALYZER_MODEL_ID.strip()
        if not model_id:
            raise ValueError("LOCAL_ANALYZER_MODEL_ID must be set for the local analyzer preset.")
        return AnalyzerModelConfig(
            provider=provider,
            base_url=_host_accessible_base_url(configured_base_url),
            model_id=model_id,
        )

    if not requested:
        requested = _LOCAL_ANALYZER_MODEL_ID.strip()
    if not requested:
        raise ValueError(
            "Analyzer model id is required. Set analyzer.model_id in config, pass --model, "
            "or set LOCAL_ANALYZER_MODEL_ID / INFERENCE_ANALYZER_MODEL."
        )

    provider = os.environ.get("OPENAI_PROVIDER", os.environ.get("LOCAL_ANALYZER_PROVIDER", "vllm")).strip().lower()
    if not provider:
        provider = "vllm"
    base_url = _host_accessible_base_url(
        os.environ.get("OPENAI_BASE_URL", os.environ.get("LOCAL_ANALYZER_BASE_URL", _LOCAL_ANALYZER_BASE_URL)).strip()
    )
    if not base_url:
        raise ValueError("OPENAI_BASE_URL or LOCAL_ANALYZER_BASE_URL must be set for direct model ids.")
    return AnalyzerModelConfig(provider=provider, base_url=base_url, model_id=requested)


def _append_transcript_section(log_path: Path, label: str, content: str) -> None:
    rendered_content = content.strip()
    if not rendered_content:
        return
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{label}]\n")
        f.write(rendered_content)
        f.write("\n\n")


def _render_transcript_section(label: str, content: str) -> str:
    rendered_content = content.strip()
    if not rendered_content:
        return ""
    return f"[{label}]\n{rendered_content}\n\n"


def _json_like_payload(value: Any) -> Any | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or stripped[0] not in "{[":
        return None
    try:
        return json.loads(stripped)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _render_scalar_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True)


def _render_human_readable_lines(value: Any, *, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        if not value:
            return [f"{prefix}{{}}"]
        lines: list[str] = []
        for key, item in value.items():
            key_text = str(key)
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key_text}:")
                lines.extend(_render_human_readable_lines(item, indent=indent + 2))
                continue
            if isinstance(item, str) and "\n" in item:
                multiline = item.splitlines() or [""]
                lines.append(f"{prefix}{key_text}: |")
                lines.extend(f"{prefix}  {line}" for line in multiline)
                continue
            lines.append(f"{prefix}{key_text}: {_render_scalar_value(item)}")
        return lines
    if isinstance(value, list):
        if not value:
            return [f"{prefix}[]"]
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_render_human_readable_lines(item, indent=indent + 2))
                continue
            if isinstance(item, str) and "\n" in item:
                multiline = item.splitlines() or [""]
                lines.append(f"{prefix}- |")
                lines.extend(f"{prefix}  {line}" for line in multiline)
                continue
            lines.append(f"{prefix}- {_render_scalar_value(item)}")
        return lines
    if isinstance(value, str):
        if "\n" in value:
            multiline = value.splitlines() or [""]
            return [f"{prefix}|", *(f"{prefix}  {line}" for line in multiline)]
        return [f"{prefix}{value}"]
    return [f"{prefix}{_render_scalar_value(value)}"]


def _render_human_readable_value(value: Any) -> str:
    return "\n".join(_render_human_readable_lines(value))


def _render_jsonish_text(value: Any) -> str:
    parsed = _json_like_payload(value)
    if parsed is not None:
        return _render_human_readable_value(parsed)
    return _normalize_message_content(value) if not isinstance(value, str) else value.strip()


def _render_tool_parameter_text(value: Any) -> str:
    if isinstance(value, str):
        return value.rstrip("\n")
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, ensure_ascii=True)
    return str(value)


def _normalize_tool_call_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return json.loads(json.dumps(arguments))
    if isinstance(arguments, str):
        stripped = arguments.strip()
        if not stripped:
            return {}
        if stripped.startswith("<tool_call>"):
            recovered_tool_calls = _recover_tool_calls_from_markup(stripped)
            if recovered_tool_calls:
                recovered_arguments = recovered_tool_calls[0].get("function", {}).get("arguments", "{}")
                return json.loads(str(recovered_arguments))
            return {}
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("tool call arguments must decode to a JSON object")
    raise ValueError("tool call arguments must be a JSON object or JSON object string")


def _render_tool_call_markup(tool_name: str, arguments: Any) -> str:
    name = str(tool_name or "").strip()
    if not name:
        return ""
    try:
        parsed_arguments = _normalize_tool_call_arguments(arguments)
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""

    lines = ["<tool_call>", f"<function={name}>"]
    for parameter_name, parameter_value in parsed_arguments.items():
        lines.append(f"<parameter={parameter_name}>")
        rendered_value = _render_tool_parameter_text(parameter_value)
        if rendered_value:
            lines.extend(rendered_value.splitlines())
        lines.append("</parameter>")
    lines.append("</function>")
    lines.append("</tool_call>")
    return "\n".join(lines)


def _render_tool_result_display(content: Any) -> str:
    parsed = _json_like_payload(content) if isinstance(content, str) else (content if isinstance(content, dict) else None)
    if isinstance(parsed, dict):
        stdout = str(parsed.get("stdout", "") or "").rstrip("\n")
        error = str(parsed.get("error", "") or "").rstrip("\n")
        result = parsed.get("result")
        has_result = result not in (None, "", [], {})
        if stdout and not error and not has_result:
            return stdout

        blocks: list[str] = []
        if stdout:
            blocks.append(stdout)
        if has_result:
            rendered_result = _render_human_readable_value(result)
            if stdout:
                blocks.append(f"result:\n{rendered_result}")
            else:
                blocks.append(rendered_result)
        if error:
            if stdout or has_result:
                blocks.append(f"error:\n{error}")
            else:
                blocks.append(error)
        if blocks:
            return "\n\n".join(block for block in blocks if block.strip())

    return _render_jsonish_text(content)


def _resolve_run_artifact_location(state_path: Path) -> tuple[Path, str | None]:
    parent = state_path.parent
    if parent.name == "artifacts" and parent.parent != parent:
        run_root = parent.parent
        runtime_state_files = list(parent.glob(f"*_{RUNTIME_STATE_FILENAME}"))
        if len(runtime_state_files) <= 1:
            return run_root, None
        runtime_state_stem = Path(RUNTIME_STATE_FILENAME).stem
        suffix = f"_{runtime_state_stem}"
        state_stem = state_path.stem
        game_stem = state_stem[:-len(suffix)] if state_stem.endswith(suffix) else state_stem
        return run_root, game_stem
    return parent, None


def _resolve_named_run_artifact(
    state_path: Path,
    *,
    default_name: str,
    per_game_suffix: str,
    directory_name: str | None = None,
) -> Path:
    run_root, game_stem = _resolve_run_artifact_location(state_path)
    output_root = run_root / directory_name if directory_name else run_root
    if game_stem:
        return output_root / f"{game_stem}{per_game_suffix}"
    return output_root / default_name


def _render_prompt_log_message(message: dict[str, Any]) -> str:
    role = str(message.get("role", "")).strip().upper() or "UNKNOWN"
    header = f"[{role}]"
    tool_call_id = str(message.get("tool_call_id", "")).strip()
    if role == "TOOL" and tool_call_id:
        header = f"[TOOL RESULT: {tool_call_id}]"
    blocks = [header]

    content = _normalize_message_content(message.get("content", ""))
    if content:
        blocks.append(_render_tool_result_display(content) if role == "TOOL" else content)

    reasoning = _extract_reasoning_text(message)
    if reasoning:
        blocks.append("[REASONING]")
        blocks.append(reasoning)

    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        for tool_call in tool_calls:
            function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
            name = str(function.get("name", "")).strip() or "unknown"
            blocks.append(f"[ASSISTANT TOOL CALL: {name}]")
            tool_call_id = str(tool_call.get("id", "")).strip()
            if tool_call_id:
                blocks.append(f"id: {tool_call_id}")
            rendered_tool_call = _render_tool_call_markup(name, function.get("arguments", "{}"))
            if rendered_tool_call:
                blocks.append(rendered_tool_call)
            else:
                raw_arguments = function.get("arguments", "{}")
                try:
                    parsed_arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                    rendered_arguments = json.dumps(parsed_arguments, indent=2, ensure_ascii=True)
                except (TypeError, ValueError, json.JSONDecodeError):
                    rendered_arguments = str(raw_arguments)
                blocks.append("arguments:")
                blocks.append(rendered_arguments if rendered_arguments.strip() else "{}")

    return "\n".join(blocks)


def _resolve_prompt_log_path(state_path: Path) -> Path:
    return _resolve_named_run_artifact(
        state_path,
        default_name="prompt.log",
        per_game_suffix=".log",
        directory_name="prompts",
    )


def _resolve_request_log_path(state_path: Path) -> Path:
    return _resolve_named_run_artifact(
        state_path,
        default_name="requests.jsonl",
        per_game_suffix="_requests.jsonl",
    )


def _append_request_snapshot(
    log_path: Path,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    event: str | None = None,
    tool_choice: str | None = None,
    finish_reason: str | None = None,
    served_by: str | None = None,
    analysis_step: int | None = None,
    action: int | None = None,
    request_index_within_turn: int | None = None,
    usage: dict[str, Any] | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> None:
    payload = {
        "messages": messages,
        "tools": tools or [],
    }
    if isinstance(chat_template_kwargs, dict) and chat_template_kwargs:
        payload["chat_template_kwargs"] = dict(chat_template_kwargs)
    if isinstance(usage, dict) and usage:
        payload["usage"] = usage
    if event:
        payload["event"] = event
    if tool_choice:
        payload["tool_choice"] = tool_choice
    if finish_reason is not None:
        payload["finish_reason"] = str(finish_reason)
    if served_by:
        payload["served_by"] = str(served_by)
    if analysis_step is not None:
        payload["analysis_step"] = analysis_step
    if action is not None:
        payload["action"] = action
    if request_index_within_turn is not None:
        payload["request_index_within_turn"] = request_index_within_turn
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                payload,
                ensure_ascii=True,
            )
        )
        f.write("\n")


def _write_prompt_log_snapshot(
    log_path: Path,
    *,
    model_id: str,
    base_url: str,
    display_action_num: int,
    analysis_step: int | None,
    request_index: int,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: str | None,
    transcript: str,
) -> None:
    rendered_messages = "\n\n".join(_render_prompt_log_message(message) for message in messages)
    rendered_tools: list[str] = []
    for tool in tools or []:
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = str(function.get("name", "")).strip() or "unknown"
        description = str(function.get("description", "")).strip()
        if description:
            rendered_tools.append(f"- {name}: {description}")
        else:
            rendered_tools.append(f"- {name}")
    analysis_label = str(analysis_step) if analysis_step is not None else "n/a"
    transcript_text = transcript.strip()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("LATEST MODEL CALL SNAPSHOT\n")
        f.write(f"model: {model_id}\n")
        f.write(f"base_url: {base_url}\n")
        f.write(f"analysis_step: {analysis_label}\n")
        f.write(f"action: {display_action_num}\n")
        f.write(f"request_index_within_turn: {request_index}\n")
        f.write(f"message_count: {len(messages)}\n")
        f.write(f"tool_choice: {tool_choice or '(none)'}\n")
        f.write("\n[AVAILABLE TOOLS]\n")
        f.write("\n".join(rendered_tools) if rendered_tools else "(none)")
        f.write("\n\n[MODEL INPUT]\n")
        f.write(rendered_messages.strip())
        f.write("\n\n[TURN TRANSCRIPT SO FAR]\n")
        f.write(transcript_text)
        f.write("\n")


def _normalize_message_content(content: Any) -> str:
    def _strip_think_tags(text: str) -> str:
        cleaned = _THINK_TAG_RE.sub("", text)
        cleaned = "\n".join(line for line in cleaned.splitlines() if line.strip())
        return cleaned.strip()

    if isinstance(content, str):
        return _strip_think_tags(content)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return _strip_think_tags("\n".join(part for part in parts if part))
    return ""


def _extract_reasoning_text(message: dict[str, Any]) -> str:
    reasoning = message.get("reasoning")
    if reasoning in (None, ""):
        reasoning = message.get("reasoning_content", "")
    return _normalize_message_content(reasoning)


def _is_context_length_error(exc: BaseException) -> bool:
    message = str(exc).lower().replace("\u2019", "'")
    return (
        "maximum context length" in message
        # SGLang reports an oversized prompt with this wording instead of
        # "maximum context length". Route it through the same drain/retry path.
        or "is longer than the model's context length" in message
        or "context_length_exceeded" in message
        or "reduce the length of the input prompt" in message
        or "parameter=input_tokens" in message
        or '"param":"input_tokens"' in message
    )


@dataclass
class _ChatCompletionResult:
    message: dict[str, Any]
    finish_reason: str = ""
    usage: dict[str, Any] | None = None
    served_by: str = ""


class ToolAgent:
    """Direct tool-calling analyzer compatible with OpenAI-style endpoints."""

    def __init__(
        self,
        *,
        model: str = _DEFAULT_ANALYZER_MODEL,
        timeout: Optional[float] = None,
        save_request_logs: bool = False,
        dispatch_index: int = 0,
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str | None = None,
    ) -> None:
        resolved_model = _resolve_analyzer_model(model)
        if base_url is not None or provider is not None:
            resolved_model = AnalyzerModelConfig(
                provider=str(provider or resolved_model.provider).strip() or resolved_model.provider,
                base_url=(
                    _host_accessible_base_url(str(base_url).strip())
                    if base_url is not None and str(base_url).strip()
                    else resolved_model.base_url
                ),
                model_id=resolved_model.model_id,
            )
        self._model = resolved_model
        configured_timeout = _LOCAL_ANALYZER_TIMEOUT if timeout is None else timeout
        self._timeout = None if configured_timeout is None or configured_timeout <= 0 else float(configured_timeout)
        self._api_key = str(api_key or "").strip()
        self._tool_steps = None if _LOCAL_ANALYZER_TOOL_STEPS <= 0 else max(1, _LOCAL_ANALYZER_TOOL_STEPS)
        self._python_timeout = min(30, max(1, _LOCAL_ANALYZER_TOOL_TIMEOUT))
        self._yield_seconds = None if _LOCAL_ANALYZER_YIELD_SECONDS <= 0 else float(_LOCAL_ANALYZER_YIELD_SECONDS)
        self._yield_tokens = (
            None if _LOCAL_ANALYZER_YIELD_TOKENS <= 0 else int(_LOCAL_ANALYZER_YIELD_TOKENS)
        )
        self._turn_generated_tokens = 0
        configured_max_output = _LOCAL_ANALYZER_MAX_OUTPUT
        self._max_output_tokens = None if configured_max_output <= 0 else max(1, configured_max_output)
        self._reply_reserve_tokens = self._max_output_tokens or 512
        self._tool_output_tokens = max(64, _LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS)
        self._tool_output_chars = max(256, self._tool_output_tokens * 4)
        self._save_request_logs = bool(save_request_logs)
        self._system_prompt = _build_system_prompt(
            tool_output_tokens=self._tool_output_tokens,
        )
        self._request_safety_margin_tokens = _REQUEST_SAFETY_MARGIN_TOKENS
        self._context_budget_tokens = max(
            1024,
            _LOCAL_ANALYZER_CONTEXT_WINDOW - self._reply_reserve_tokens - self._request_safety_margin_tokens,
        )
        self._turns_without_wm_update = 0
        self._pending_wm_rebuild_reason = ""
        self._wm_absorbed_in_turn = False
        self._auto_diff_prev_frame: Frame | None = None
        self._pending_diff_image: dict[str, Any] | None = None
        self._context_was_trimmed = False
        self._has_evicted = False
        self._actions_at_level_start = 0
        self._tokens_at_level_start = 0
        self._progress_pace = ProgressPace()
        self._pace_last_completed_level = 0
        # queue position spreads the untrimmed band so games do not tie
        self._diag_name = f"g{int(dispatch_index):02d}"
        self._game_elapsed_seconds: float | None = None
        self._endgame_announced = False
        self._priority_current = _PRIORITY_UNTRIMMED_BASE - max(0, int(dispatch_index))
        self._priority_next = self._priority_current
        self._kept_functions: dict[str, str] = {}
        self._last_animation_chain: list[Any] = []
        self._last_action_gameplay_changed: bool | None = None
        self._pending_animation_image: dict[str, Any] | None = None
        self._pending_animation_image_label: str = ""
        self._pending_diff_image_label: str = ""
        self._turn_diff_image: dict[str, Any] | None = None
        self._turn_diff_image_label: str = ""
        self._pending_gameover_images: list[tuple[str, dict[str, Any]]] = []
        self._noop_guard_pre_hash: str = ""
        # (interior board hash, action signature) already run in THIS snippet.
        # Written in _record_guard_observations, read in _action_guard_hook,
        # cleared before each sandbox call - three sites, one lifetime.
        self._summary_context_tokens_since_attempt = 0
        self._http_initial_grace_used = False
        self._repeat_state_seen: set[tuple[str, str]] = set()
        # (state_hash, action_sig) for each action the guard hook waved through,
        # in execution order - the only place the intermediate board states of a
        # batch are visible to the agent, since the solver reports outcomes per
        # position but not the boards they ran against.
        self._guard_hook_trace: list[tuple[str, str]] = []
        self._rejected_section_updates: list[str] = []
        # characters per token, measured from the last response; None until one
        # arrives, and reported in the transcript whenever it changes
        self._text_chars_per_token: float | None = None
        # index into the reasoning-effort ladder; -1 means "server default"
        self._reasoning_effort_rung: int = -1
        self._guard_death_override_batch: bool = False
        self._noop_guard_deaths_enabled: bool = _get_env_bool(
            "ARC3_DEATH_REPEAT_GUARD", False
        )
        self._noop_repeat_guard: NoopRepeatGuard | None = (
            NoopRepeatGuard()
            if (
                _get_env_bool("ARC3_NOOP_REPEAT_GUARD", False)
                or self._noop_guard_deaths_enabled
            )
            else None
        )
        # only the no-op half is gated separately; the shared store is created
        # when either half is enabled
        self._noop_guard_noops_enabled: bool = _get_env_bool(
            "ARC3_NOOP_REPEAT_GUARD", False
        )
        self._resume_after_yield: bool = False
        self._resume_reason: str = ""
        self._prev_level_start_frame: Frame | None = None
        self._cur_level_start_frame: Frame | None = None
        self._death_ledger_level: Any = None
        self._death_ledger_attempts: list[dict[str, Any]] = []
        self._death_ledger_index: dict[str, list[tuple[int, int]]] = {}
        self._death_ledger_total: int = 0
        self._history_messages: list[dict[str, Any]] = []
        self._session_runtime_dir: Path | None = None
        self._session_total_tokens = 0
        self._session_generated_tokens = 0
        self._step_env_callback: Callable[[dict[str, Any]], dict[str, Any]] | None = None
        self._current_valid_actions: list[str] = []
        self._last_step_summary: dict[str, Any] | None = None
        self._last_action_call_result: dict[str, Any] | None = None
        self._summarized_knowledge = _empty_world_model()

    def _headers(self) -> dict[str, str]:
        api_key = (
            self._api_key
            or os.environ.get("LOCAL_ANALYZER_API_KEY", "").strip()
            or os.environ.get("OPENROUTER_API_KEY", "").strip()
            or os.environ.get("OPENAI_API_KEY", "").strip()
        )
        site_url = os.environ.get("LOCAL_ANALYZER_SITE_URL", "").strip()
        app_name = os.environ.get("LOCAL_ANALYZER_APP_NAME", "ARC3 Agent Harness").strip()
        return build_headers(
            provider=self._model.provider,
            api_key=api_key,
            referer=site_url,
            title=app_name,
        )

    def _ensure_session(self, state_path: Path) -> None:
        runtime_dir = state_path.parent
        if self._session_runtime_dir != runtime_dir:
            self._session_runtime_dir = runtime_dir
            self._priority_level_count_warned = False
            if _persistent_functions():
                self._kept_functions = {}
                self._retained_clear_notice = ""
            self._history_messages = []
            self._session_total_tokens = 0
            self._session_generated_tokens = 0
            self._summary_context_tokens_since_attempt = 0
            self._last_step_summary = None
            self._last_action_call_result = None
            self._summarized_knowledge = _empty_world_model()
            self._turns_without_wm_update = 0
            self._pending_wm_rebuild_reason = ""
            self._pending_wm_revise_reason = ""

    @property
    def total_tokens(self) -> int:
        return max(0, int(self._session_total_tokens))

    @property
    def generated_tokens(self) -> int:
        return max(0, int(self._session_generated_tokens))

    def _accumulate_usage_tokens(
        self, usage: dict[str, Any] | None, *, count_toward_turn: bool = True
    ) -> None:
        """Add a response's usage to the counters.

        count_toward_turn=False for generation that is not part of the turn's
        deliberation - the rolling summary. Its tokens are real GPU time and
        belong in the session and per-game totals, or a summary-heavy run looks
        cheaper than it is and gets more compute than
        max_generated_tokens_per_game intends. But the per-turn yield budget
        measures how long the model has been working on THIS turn, and charging
        it for a summary would shorten whichever turn happened to follow one.
        """
        if not isinstance(usage, dict):
            return
        generated_token_count = 0
        for key in ("completion_tokens", "output_tokens", "generated_tokens"):
            raw_value = usage.get(key)
            try:
                generated_token_count = max(0, int(raw_value))
                break
            except (TypeError, ValueError):
                continue
        self._session_generated_tokens += generated_token_count
        if count_toward_turn:
            self._turn_generated_tokens += generated_token_count

        total_tokens = usage.get("total_tokens")
        try:
            if total_tokens is not None:
                self._session_total_tokens += max(0, int(total_tokens))
                return
        except (TypeError, ValueError):
            pass

        token_count = 0
        for key in ("prompt_tokens", "completion_tokens", "input_tokens", "output_tokens"):
            raw_value = usage.get(key)
            try:
                token_count += max(0, int(raw_value))
            except (TypeError, ValueError):
                continue
        self._session_total_tokens += token_count

    @staticmethod
    def _turn_action_trace(executed_results: list[dict[str, Any]]) -> list[str]:
        """Stitch per-call action traces into one sequentially numbered
        turn-level trace; synthesize entries for calls without a trace."""
        entries: list[str] = []
        for item in executed_results:
            item_trace = item.get("action_trace")
            if isinstance(item_trace, list) and item_trace:
                entries.extend(
                    re.sub(r"^\d+:\s*", "", str(line)) for line in item_trace
                )
                continue
            names = item.get("executed_actions")
            if not isinstance(names, list) or not names:
                display = str(item.get("action_display") or "").strip()
                names = [display] if display else []
            if item.get("game_over"):
                verdict = "GAME OVER (attempt ended immediately after this action)"
            elif item.get("run_complete") or item.get("done"):
                verdict = "RUN COMPLETE"
            elif item.get("level_completed"):
                verdict = "LEVEL COMPLETED"
            elif item.get("gameplay_changed") is False:
                verdict = (
                    "NO-OP (executed, no change inside the board area)"
                    if _new_noop_wording()
                    else "NO-OP (executed, changed nothing)"
                )
            elif item.get("gameplay_changed") is True:
                verdict = "effect"
            else:
                verdict = "executed"
            entries.extend(f"{name} -> {verdict}" for name in names)
        return [f"{position}: {entry}" for position, entry in enumerate(entries, 1)]

    def _summarize_step_sequence(self, action_results: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not action_results:
            return None
        executed_results = [item for item in action_results if item.get("executed")]
        if not executed_results:
            return None

        total_executed = 0
        executed_actions: list[str] = []
        for item in executed_results:
            count = item.get("executed_count")
            try:
                parsed = int(count) if count is not None else 1
            except (TypeError, ValueError):
                parsed = 1
            total_executed += max(1, parsed)
            action_names = item.get("executed_actions")
            if isinstance(action_names, list):
                executed_actions.extend(str(name).strip() for name in action_names if str(name).strip())
            else:
                fallback_action = str(item.get("action_display") or "").strip()
                if fallback_action:
                    executed_actions.append(fallback_action)

        last = executed_results[-1]
        try:
            end_action_num = int(last.get("action_num"))
        except (TypeError, ValueError):
            end_action_num = None
        start_action_num = None
        if end_action_num is not None and total_executed > 0:
            start_action_num = max(1, end_action_num - total_executed + 1)

        return {
            "start_action_num": start_action_num,
            "end_action_num": end_action_num,
            "executed_count": total_executed,
            "executed_actions": executed_actions,
            "level": last.get("level"),
            "level_transition": any(bool(item.get("level_completed")) for item in executed_results),
            "run_complete": any(bool(item.get("run_complete")) for item in executed_results),
            "game_over": any(bool(item.get("game_over")) for item in executed_results),
            "game_over_trace": (
                ToolAgent._turn_action_trace(executed_results)
                if any(item.get("game_over") for item in executed_results)
                else []
            ),
            "fatal_action": next(
                (
                    str(item.get("fatal_action") or item.get("action_display") or "")
                    for item in executed_results
                    if item.get("game_over")
                ),
                "",
            ),
            "board_changed": any(bool(item.get("board_changed")) for item in executed_results),
            # the interior-only flag, so the opener can say WHICH kind of change
            # happened instead of hedging over the broader one
            "gameplay_changed": any(
                bool(item.get("gameplay_changed")) for item in executed_results
            ),
            "stop_reason": last.get("stop_reason"),
            "no_op_stops": [
                {
                    "action": item.get("no_op_action"),
                    "index": item.get("no_op_action_index"),
                    "skipped": item.get("skipped_actions") or [],
                    "trace": item.get("action_trace") or [],
                }
                for item in executed_results
                if item.get("stop_reason") == "no_op_action"
            ],
            # the animation of the last executed action, which is what
            # last_animation_frames / last_animation_timeline will return when asked
            "animation": (executed_results[-1].get("animation") if executed_results else None),
            "animation_action": (
                executed_results[-1].get("action_display") if executed_results else None
            ),
            "guard_stops": [
                {
                    "kind": item.get("stop_reason"),
                    "action": item.get("blocked_action"),
                    "detail": item.get("stop_detail"),
                    "skipped": item.get("skipped_actions") or [],
                    "trace": item.get("action_trace") or [],
                }
                # scans ALL results, not just executed ones: a refusal has
                # executed=False by construction and is filtered out of
                # executed_results before this point
                for item in action_results
                if item.get("stop_reason") in ("known_noop", "known_death", "stale_state")
            ],
        }

    # _describe_last_outcome removed. It built an outcome sentence and was
    # called from nowhere - in this harness AND upstream - so neither its
    # wording nor the fallback it carried ever reached a transcript. Two
    # patches were spent revising text no model ever read. What it uniquely
    # provided is now rendered inline in _build_user_prompt.

    def _update_summarized_knowledge_from_assistant(self, content: str) -> list[str]:
        if _memory_sections_disabled():
            # nothing is carried or shown, so parsing would only build state
            # that never reaches the model
            return []
        note = _extract_scientist_note(content, self._summarized_knowledge)
        if not note:
            return []
        updated: list[str] = []
        rejected: list[str] = []
        for key, value in note.items():
            if not value:
                continue
            if _is_degenerate_text(value):
                # keep the previous version rather than truncating: a truncated
                # degenerate section is still degenerate, and it would be
                # re-injected into every later prompt
                rejected.append(key)
                continue
            self._summarized_knowledge[key] = value
            updated.append(key)
        if rejected:
            self._rejected_section_updates = sorted(set(rejected))
            log.warning(
                "rejected degenerate memory section update(s): %s",
                ", ".join(self._rejected_section_updates),
            )
        if updated:
            self._turns_without_wm_update = 0
            self._wm_absorbed_in_turn = True
            if "world_model" in updated:
                self._pending_wm_rebuild_reason = ""
            if "world_model" in updated or "current_plan" in updated:
                self._pending_wm_revise_reason = ""
        return updated

    def _update_summarized_knowledge_from_step_summary(self) -> None:
        summary = self._last_step_summary
        if not summary:
            return
        wipe_reason = ""
        revise_reason = ""
        if summary.get("level_transition"):
            # the priority estimate is per level, so both counts restart here
            if _get_env_bool("ARC3_PRIORITY_PACE", False):
                completed_level = max(1, int(summary.get("level") or 2) - 1)
                if completed_level > self._pace_last_completed_level:
                    self._progress_pace.record_completion(
                        self._session_generated_tokens - self._tokens_at_level_start,
                        _pace_reference_tokens(completed_level),
                    )
                    self._pace_last_completed_level = completed_level
            self._actions_at_level_start = _priority_action_count(summary)
            self._tokens_at_level_start = self._session_generated_tokens
        if summary.get("level_transition") and _persistent_functions_scope() == "levelup":
            # a helper can encode the old layout in its code even when it holds
            # no stale data, so the level boundary is where it stops being safe
            self._kept_functions = {}
            if _persistent_functions():
                self._retained_clear_notice = (
                    "Your retained functions were cleared at this level transition. "
                    "Redefine any helpers you still need after checking their assumptions "
                    "against the new board."
                )
        elif (
            summary.get("level_transition")
            and _persistent_functions_scope() == "game"
            and _persistent_functions()
            and getattr(self, "_kept_functions", None)
        ):
            self._retained_clear_notice = (
                "Your retained functions remain available from previous levels. "
                "Check level-specific assumptions against the new board before reusing them."
            )
        if summary.get("level_transition"):
            # read at call time (not a module constant) so this patch's hunks
            # stay clear of the env-knob block that other patches extend
            if _get_env_int("ARC3_WM_WIPE_ON_LEVEL", 1):
                wipe_reason = "a level transition"
            else:
                revise_reason = "level"
        elif summary.get("run_complete"):
            wipe_reason = "run completion"
        elif summary.get("game_over"):
            if _WM_WIPE_ON_GAME_OVER:
                wipe_reason = "a game over"
            else:
                revise_reason = "death"
        if revise_reason:
            self._pending_wm_revise_reason = revise_reason
            self._wm_revise_hint_shown = False
            # a newer keep-event supersedes any stale rebuild nag (and vice
            # versa below): at most one pending nudge, always for the latest event
            self._pending_wm_rebuild_reason = ""
        if wipe_reason:
            self._pending_wm_rebuild_reason = wipe_reason
            self._pending_wm_revise_reason = ""
            for key in (
                "world_model",
                "goal_model",
                "action_model",
                "recent_findings",
                "open_questions",
                "current_plan",
            ):
                self._summarized_knowledge[key] = ""

    def _summarized_knowledge_lines(self) -> list[str]:
        entries = [
            ("World model", self._summarized_knowledge.get("world_model", "")),
            ("Goal model", self._summarized_knowledge.get("goal_model", "")),
            ("Action model", self._summarized_knowledge.get("action_model", "")),
            ("Recent findings", self._summarized_knowledge.get("recent_findings", "")),
            ("Open questions", self._summarized_knowledge.get("open_questions", "")),
            ("Plan", self._summarized_knowledge.get("current_plan", "")),
            ("Cross-level notes", self._summarized_knowledge.get("cross_level_notes", "")),
        ]
        if _memory_sections_disabled():
            return []
        rejected = list(getattr(self, "_rejected_section_updates", []))
        self._rejected_section_updates = []
        allowed = _memory_section_labels()
        if allowed is not None:
            entries = [(label, value) for label, value in entries if label in allowed]
        lines = [f"- {label}: {value}" for label, value in entries if value]
        if rejected:
            # without this the model sees its section unchanged and is likely to
            # emit the same degenerate text again
            lines.append(
                "- NOTE: your last update to "
                + ", ".join(f"`{name.replace('_', ' ')}`" for name in rejected)
                + " was REJECTED as degenerate (highly repetitive or corrupted "
                "text) and the previous version above was kept. Rewrite it "
                "concisely in plain prose."
            )
        if not lines:
            return []
        return [
            "Working world model carried from earlier turns:",
            *lines,
            "- Revise any item above immediately if `current_frame` or `history` contradicts it.",
        ]

    def _prepare_auto_diff(
        self,
        current_frame: Frame | None,
        previous_step_summary: dict[str, Any] | None,
    ) -> list[str]:
        """Track the frame shown at the previous analyzer turn and, when
        enabled, produce the auto-injected frame-diff lines and/or the diff
        image for the outgoing user message. Suppressed on the first turn and
        on the turn after a game over (the auto-RESET makes that span
        meaningless)."""
        if not (_AUTO_FRAME_DIFF or _DIFF_IMAGE or _level_inventory_enabled()):
            return []
        # Imported lazily so this patch's hunks stay clear of the module
        # import header, which other optional patches also modify.
        from inference.agent.vision_context import diff_image_part
        from inference.utils.frame_diff import compute_frame_diff
        from inference.utils.grid_utils import ARC_COLOR_CHARS
        from inference.utils.segmentation import segment_layer
        self._pending_diff_image = None
        self._pending_diff_image_label = ""
        # a new opener supersedes the previous turn's diff whether or not it
        # produces one of its own
        self._turn_diff_image = None
        self._turn_diff_image_label = ""
        if current_frame is None:
            return []
        previous_frame = self._auto_diff_prev_frame
        self._auto_diff_prev_frame = current_frame
        if previous_frame is None:
            self._cur_level_start_frame = current_frame
            return []
        prev_level = getattr(previous_frame, "level", None)
        cur_level = getattr(current_frame, "level", None)
        level_changed = (
            prev_level is not None
            and cur_level is not None
            and prev_level != cur_level
        )
        if level_changed:
            # shift the per-level start frames even when a game over follows:
            # the current frame is this level's (near-)initial board
            self._prev_level_start_frame = self._cur_level_start_frame
            self._cur_level_start_frame = current_frame
        if previous_step_summary and previous_step_summary.get("game_over"):
            return []
        if level_changed:
            if not _level_inventory_enabled():
                # still clear the diff image: a cross-level diff has no meaning
                self._pending_diff_image = None
                self._pending_diff_image_label = ""
                return []
            return self._level_transition_diff_lines(
                current_frame, previous_step_summary
            )
        if not (_AUTO_FRAME_DIFF or _DIFF_IMAGE):
            # reached only when the inventory knob alone kept us here
            return []
        prev_step = getattr(previous_frame, "step", None)
        cur_step = getattr(current_frame, "step", None)
        span = f"step {prev_step} -> {cur_step}"
        if _DIFF_IMAGE:
            part = diff_image_part(previous_frame, current_frame)
            if part is not None and previous_frame.grid != current_frame.grid:
                label = (
                    f"Diff image ({span}): cells that changed since your previous "
                    "analyzer turn in their new true color; unchanged cells dimmed "
                    "to dark navy (an off-palette color)."
                )
                self._pending_diff_image = part
                self._pending_diff_image_label = label
                # kept beyond the single use above so a resumption can re-show
                # it; replaced whenever a new opener produces a fresh diff, so
                # a stale one from an earlier turn can never leak in
                self._turn_diff_image = part
                self._turn_diff_image_label = label
        if not _AUTO_FRAME_DIFF:
            return []
        executed = 0
        if previous_step_summary and not previous_step_summary.get("stale"):
            # a stale summary describes an earlier exchange whose changes were
            # already diffed then; nothing has executed since, so don't claim
            # "no cell changed despite N executed actions"
            try:
                executed = int(previous_step_summary.get("executed_count") or 0)
            except (TypeError, ValueError):
                executed = 0
        if previous_frame.grid == current_frame.grid:
            if executed > 0:
                return [
                    f"Auto frame diff ({span}): no cell changed since your previous "
                    f"analyzer turn despite {executed} executed action(s)."
                ]
            return []
        prev_rows = len(previous_frame.grid)
        cur_rows = len(current_frame.grid)
        prev_cols = max((len(r) for r in previous_frame.grid), default=0)
        cur_cols = max((len(r) for r in current_frame.grid), default=0)
        if (prev_rows, prev_cols) != (cur_rows, cur_cols):
            return [
                f"Auto frame diff ({span}): grid shape changed "
                f"({prev_rows}x{prev_cols} -> {cur_rows}x{cur_cols}); diff skipped."
            ]
        diff = compute_frame_diff(
            previous_frame.grid,
            current_frame.grid,
            segment_layer(previous_frame.grid, ARC_COLOR_CHARS).get("nodes", []),
            segment_layer(current_frame.grid, ARC_COLOR_CHARS).get("nodes", []),
            max_group_match=_AUTO_FRAME_DIFF_MAX_GROUP,
        )
        rendered = _render_auto_frame_diff(
            diff, _AUTO_FRAME_DIFF_BUDGET, total_cells=cur_rows * cur_cols
        )
        lines = [
            f"Auto frame diff ({span}; covers everything since your previous "
            "analyzer turn, not just the last action):"
        ]
        lines.extend(rendered)
        return lines

    def _death_ledger_record(self, history_entries: list["HistoryEntry"]) -> None:
        entries = list(history_entries or [])
        if len(entries) < 2 or str(entries[-1].action or "").upper() != "RESET":
            return
        death_idx = len(entries) - 2
        death = entries[death_idx]
        if death.frame is None or str(death.action or "").upper() == "RESET":
            return
        death_level = getattr(death.frame, "level", None)
        if (
            self._death_ledger_level is not None
            and death_level is not None
            and death_level != self._death_ledger_level
        ):
            self._death_ledger_attempts = []
            self._death_ledger_index = {}
        if death_level is not None:
            self._death_ledger_level = death_level
        start_idx = None
        for idx in range(death_idx - 1, -1, -1):
            entry = entries[idx]
            if str(entry.action or "").upper() == "RESET":
                start_idx = idx
                break
            level = (
                getattr(entry.frame, "level", None)
                if entry.frame is not None
                else None
            )
            if (
                death_level is not None
                and level is not None
                and level != death_level
            ):
                start_idx = idx + 1
                break
        partial = start_idx is None
        if partial:
            start_idx = 0
        actions = [
            str(entries[i].action) for i in range(start_idx + 1, death_idx + 1)
        ]
        frames = [entries[i].frame for i in range(start_idx, death_idx)]
        trim = 0
        for i in range(len(frames) - 1, -1, -1):
            if frames[i] is None:
                trim = i + 1
                break
        if trim:
            frames = frames[trim:]
            actions = actions[trim:]
            partial = True
        if not actions or len(frames) != len(actions):
            return
        for item in self._death_ledger_attempts:
            if item["actions"] == actions:
                self._death_ledger_total += 1
                item["ids"].append(self._death_ledger_total)
                return
        self._death_ledger_total += 1
        uidx = len(self._death_ledger_attempts)
        hashes = [_interior_state_hash(f.grid) for f in frames]
        self._death_ledger_attempts.append(
            {
                "actions": actions,
                "hashes": hashes,
                "ids": [self._death_ledger_total],
                "partial": partial,
            }
        )
        for pos, state_hash in enumerate(hashes):
            self._death_ledger_index.setdefault(state_hash, []).append((uidx, pos))

    def _death_ledger_payload(self) -> dict[str, Any]:
        return {
            "level": self._death_ledger_level,
            "attempts": [
                {
                    "actions": list(item["actions"]),
                    "state_hashes": list(item["hashes"]),
                    "attempt_ids": list(item["ids"]),
                    "partial": bool(item.get("partial")),
                }
                for item in self._death_ledger_attempts
            ],
        }

    def _death_ledger_observe(
        self,
        history_entries: list["HistoryEntry"],
        previous_step_summary: dict[str, Any] | None,
    ) -> None:
        """Record a death, if the previous sequence ended in one.

        Called unconditionally rather than from the rendering path. The ledger
        has TWO consumers - the advisory lines under ARC3_DEATH_LEDGER, and the
        death guard's fatal lookup under ARC3_DEATH_REPEAT_GUARD - and the
        recorder used to live inside the first. So enabling only the guard left
        it consulting an index nothing ever wrote to: armed, checked, always
        empty, every repeat of a fatal route allowed through. Observed as three
        identical deaths in a row with the guard on.
        """
        if not (
            previous_step_summary
            and previous_step_summary.get("game_over")
            and not previous_step_summary.get("stale")
        ):
            return
        if not (
            _get_env_bool("ARC3_DEATH_LEDGER", False)
            or _get_env_bool("ARC3_DEATH_REPEAT_GUARD", False)
        ):
            # nothing reads it; skip the work rather than accumulate state
            return
        self._death_ledger_record(history_entries)

    def _death_ledger_lines(
        self,
        current_frame: Frame | None,
        history_entries: list["HistoryEntry"],
        previous_step_summary: dict[str, Any] | None,
    ) -> list[str]:
        """Advisory display of recorded fatal continuations matching the
        CURRENT interior state. Broad by design (bar/items may not be visible
        in the interior), hence advisory phrasing; exact-sequence enforcement
        is a separate concern."""
        if current_frame is None:
            return []
        level = getattr(current_frame, "level", None)
        if (
            self._death_ledger_level is not None
            and level is not None
            and level != self._death_ledger_level
        ):
            self._death_ledger_attempts = []
            self._death_ledger_index = {}
            self._death_ledger_level = level
        if not self._death_ledger_attempts:
            return []
        state_hash = _interior_state_hash(current_frame.grid)
        matches: dict[int, int] = {}
        for uidx, pos in self._death_ledger_index.get(state_hash, []):
            matches[uidx] = max(matches.get(uidx, -1), pos)
        if not matches:
            return []
        by_suffix: dict[tuple, list[int]] = {}
        for uidx, pos in matches.items():
            item = self._death_ledger_attempts[uidx]
            suffix = tuple(item["actions"][pos:])
            by_suffix.setdefault(suffix, []).extend(item["ids"])
        ordered = sorted(by_suffix.items(), key=lambda kv: len(kv[0]))
        total_deaths = sum(
            len(item["ids"]) for item in self._death_ledger_attempts
        )
        lines = [
            "Known fatal continuations FROM THE CURRENT BOARD STATE (this "
            f"level has {total_deaths} failed attempt(s) recorded; the game "
            "is deterministic):"
        ]
        for suffix, ids in ordered[:3]:
            shown = ", ".join(suffix[:12]) + (", ..." if len(suffix) > 12 else "")
            id_text = ", ".join(str(i) for i in sorted(set(ids)))
            lines.append(
                f"- {len(suffix)} action(s): {shown} -> GAME OVER "
                f"(attempt(s) {id_text})"
            )
        if len(ordered) > 3:
            lines.append(
                f"- +{len(ordered) - 3} more matching fatal continuation(s)"
            )
        lines.append(
            "Do NOT replay a listed continuation exactly - vary the route or "
            "action at or before its first step. (Advisory: fatality can "
            "depend on state not visible in the board interior, e.g. bar "
            "level or held items; if you are certain the situation differs, "
            "state why before proceeding.) The full ledger is available in "
            "python as `death_ledger`."
        )
        return lines

    def _level_transition_diff_lines(
        self,
        current_frame: Frame | None,
        previous_step_summary: dict[str, Any] | None,
    ) -> list[str]:
        """First prompt of a new level: acknowledge the completing action and
        render a START-vs-START inventory against the previous level instead
        of the meaningless end-of-old-board vs new-board diff. The diff image
        is suppressed for this prompt (no cross-level semantics)."""
        from inference.utils.frame_diff import compute_frame_diff
        from inference.utils.grid_utils import ARC_COLOR_CHARS
        from inference.utils.segmentation import segment_layer

        self._pending_diff_image = None
        self._pending_diff_image_label = ""
        lines: list[str] = []
        final_action = ""
        if previous_step_summary and not previous_step_summary.get("stale"):
            actions = previous_step_summary.get("executed_actions")
            if isinstance(actions, list) and actions:
                final_action = str(actions[-1]).strip()
        lines.append(
            "Level completed - the board was replaced, so no object diff is "
            "shown across the boundary"
            + (f" (the final action {final_action!r} finished the level)" if final_action else "")
            + "."
        )
        prev_start = self._prev_level_start_frame
        cur_start = self._cur_level_start_frame
        if prev_start is None or cur_start is None:
            return lines
        grid_a, grid_b = prev_start.grid, cur_start.grid
        rows = len(grid_b)
        cols = max((len(r) for r in grid_b), default=0)
        if len(grid_a) != rows or max(
            (len(r) for r in grid_a), default=0
        ) != cols:
            lines.append(
                "New level inventory skipped (boards have different shapes)."
            )
            return lines
        diff = compute_frame_diff(
            grid_a,
            grid_b,
            segment_layer(grid_a, ARC_COLOR_CHARS).get("nodes", []),
            segment_layer(grid_b, ARC_COLOR_CHARS).get("nodes", []),
            max_group_match=_AUTO_FRAME_DIFF_MAX_GROUP,
        )
        lines.append(
            "New level inventory (previous level START vs this level START; "
            "read it as a TYPE comparison, not motion: 'moved' = object types "
            "recurring from the previous level at a new spot, 'appeared' = "
            "types NOT present on the previous level - likely new mechanics, "
            "investigate these first, 'disappeared' = previous types absent "
            "here; identical objects at identical positions are not listed):"
        )
        lines.extend(
            _render_auto_frame_diff(
                diff, _AUTO_FRAME_DIFF_BUDGET, total_cells=rows * cols
            )
        )
        return lines

    def _prepare_gameover_images(
        self,
        history_entries: list["HistoryEntry"],
        previous_step_summary: dict[str, Any] | None = None,
    ) -> None:
        """Queue the death-turn visuals (ARC3_GAMEOVER_DIFF_IMAGE).

        The auto-diff - and with it the diff image - is suppressed after a game
        over because the auto-RESET makes that span meaningless, so the visual
        slot is idle on exactly the turn where the death is invisible: the
        board the model sees has already been reset. Two parts fill it:

          1. the death frame itself (the budget bar in it is what the BAR RULE
             asks about, and it exists nowhere else in the message), and
          2. a change-highlight diff of the killing action (last alive frame ->
             death frame), the spatial form of the textual 'Fatal step diff'.

        Anchoring matches _game_over_diff_lines: history ends with the RESET
        marker, entries[-2] is the death frame, entries[-3] the last alive one.
        """
        self._pending_gameover_images = []
        if not _get_env_bool("ARC3_GAMEOVER_DIFF_IMAGE", False):
            return
        from inference.agent.vision_context import (
            current_grid_image_part,
            diff_image_part,
        )

        entries = list(history_entries or [])
        if len(entries) < 3 or str(entries[-1].action or "").upper() != "RESET":
            return
        death = entries[-2]
        last_alive = entries[-3]
        if death.frame is None or str(death.action or "").upper() == "RESET":
            return
        death_part = current_grid_image_part(death.frame)
        if death_part is not None:
            self._pending_gameover_images.append(
                (
                    "Death frame image (the board at the moment the attempt "
                    "ended, before the automatic reset): inspect the budget bar "
                    "here for the BAR RULE.",
                    death_part,
                )
            )
        self._append_fatal_animation_image(previous_step_summary)
        if last_alive.frame is None:
            return
        fatal_part = diff_image_part(last_alive.frame, death.frame)
        if fatal_part is not None and last_alive.frame.grid != death.frame.grid:
            self._pending_gameover_images.append(
                (
                    "Fatal step diff image (last alive frame -> death frame): "
                    "cells changed by the killing action in their new true "
                    "color; unchanged cells dimmed to dark navy (an off-palette "
                    "color).",
                    fatal_part,
                )
            )

    def _game_over_diff_lines(
        self,
        history_entries: list["HistoryEntry"],
        previous_step_summary: dict[str, Any] | None = None,
    ) -> list[str]:
        """After a game over (fresh rendering only), push two diffs over the
        fatal sequence itself: sequence progress (frame before the sequence's
        first action -> last alive frame, omitted when fewer than two actions
        executed) and the fatal step (last alive -> death frame). Anchored by
        the summary's executed_count - the turns before the fatal sequence
        already delivered their own diffs while the model was alive."""
        from inference.utils.frame_diff import compute_frame_diff
        from inference.utils.grid_utils import ARC_COLOR_CHARS
        from inference.utils.segmentation import segment_layer

        entries = list(history_entries or [])
        if len(entries) < 2 or str(entries[-1].action or "").upper() != "RESET":
            return []
        death_idx = len(entries) - 2
        death = entries[death_idx]
        if death.frame is None or str(death.action or "").upper() == "RESET":
            return []
        executed = 0
        if previous_step_summary:
            try:
                executed = int(previous_step_summary.get("executed_count") or 0)
            except (TypeError, ValueError):
                executed = 0
        if executed < 1:
            executed = 1
        start_idx = death_idx - executed
        partial = start_idx < 0
        if partial:
            start_idx = 0
        origin = "earliest retained frame" if partial else "sequence start"
        death_level = getattr(death.frame, "level", None)
        while start_idx < death_idx - 1:
            frame = entries[start_idx].frame
            level = getattr(frame, "level", None) if frame is not None else None
            if (
                frame is None
                or (
                    death_level is not None
                    and level is not None
                    and level != death_level
                )
            ):
                # the sequence itself crossed a level-up (or a frame is
                # missing); anchor at the first same-level frame within it
                start_idx += 1
                origin = "level start within the sequence"
                continue
            break
        start = entries[start_idx]
        last_alive = entries[death_idx - 1]
        if start.frame is None or last_alive.frame is None:
            return []

        def _diff_lines(frame_a, frame_b, label: str) -> list[str]:
            grid_a, grid_b = frame_a.grid, frame_b.grid
            rows = len(grid_b)
            cols = max((len(r) for r in grid_b), default=0)
            if len(grid_a) != rows or max(
                (len(r) for r in grid_a), default=0
            ) != cols:
                return [f"{label} skipped (grid shape changed)."]
            diff = compute_frame_diff(
                grid_a,
                grid_b,
                segment_layer(grid_a, ARC_COLOR_CHARS).get("nodes", []),
                segment_layer(grid_b, ARC_COLOR_CHARS).get("nodes", []),
                max_group_match=_AUTO_FRAME_DIFF_MAX_GROUP,
            )
            rendered = _render_auto_frame_diff(
                diff, _AUTO_FRAME_DIFF_BUDGET, total_cells=rows * cols
            )
            return [label, *rendered]

        out: list[str] = []
        pre_death_actions = (death_idx - 1) - start_idx
        if pre_death_actions >= 1:
            out.extend(
                _diff_lines(
                    start.frame,
                    last_alive.frame,
                    f"Sequence progress diff ({origin} step "
                    f"{getattr(start.frame, 'step', '?')} -> step "
                    f"{getattr(last_alive.frame, 'step', '?')}, what the first "
                    f"{pre_death_actions} action(s) of the fatal sequence "
                    "changed - the reset reverted all of it):",
                )
            )
        out.extend(
            _diff_lines(
                last_alive.frame,
                death.frame,
                f"Fatal step diff (step {getattr(last_alive.frame, 'step', '?')} "
                f"-> step {getattr(death.frame, 'step', '?')}, what the killing "
                f"action {str(death.action)!r} changed):",
            )
        )
        return out

    def _append_fatal_animation_image(self, summary: dict[str, Any] | None) -> None:
        """An image of what the fatal action's animation hid, on a game over.

        Gated on transient pixels alone, NOT on gameplay_changed: after a death
        the board has already reset, so gameplay_changed compares the
        pre-action board against the restarted level and is essentially always
        true. The justification differs from the ordinary case too - there the
        board shows nothing, here it shows a different board entirely, and
        either way the fatal moment is unreachable.
        """
        if not _animation_enabled():
            return
        mode = animation_death_image_mode()
        if mode == "off" or not summary:
            return
        animation = summary.get("animation") or {}
        if not int(animation.get("transient_pixels") or 0):
            return
        chain = self._last_animation_chain or []
        cells = _animation_transient_cells(chain)
        if not cells:
            return
        if mode == "composite":
            part = animation_composite_part(
                chain, cells, border=_get_env_int("ARC3_NOOP_GUARD_BORDER", 4)
            )
            label = (
                "Fatal action animation image: the cells that changed during the fatal "
                "action's animation, each in the last colour it held before reverting; "
                "everything else, including the border, is dimmed to dark navy (an "
                "off-palette colour). This is a composite of all cells that changed "
                "during the animation, not a single board state that the game "
                "displayed. A cell is drawn nearer its true colour the longer it held "
                "it, so a bright trail marks where something stayed and a uniformly "
                "dim one marks something that passed through. Neither the death frame "
                "nor the last alive frame contains these cells - they existed only "
                "between the two."
            )
        else:
            part, index = animation_peak_part(chain, cells)
            label = (
                f"Fatal action animation image (frame {index} of {len(chain) - 1}): the "
                "board partway through the fatal action's animation, at the point where "
                "the most cells were showing something they later reverted. This is a "
                "real state the game displayed, between the last alive frame and the "
                "death frame."
            )
        if part is not None:
            self._pending_gameover_images.append((label, part))

    def _prepare_animation_image(self, summary: dict[str, Any] | None) -> None:
        """Attach an image of what the animation hid, when the board did not move.

        Gated on gameplay_changed being false: that is exactly when the board
        image the model receives is uninformative, because the action did
        something and no frame it can reach shows what. On any other turn the
        current board already carries the outcome.
        """
        if not _animation_enabled():
            # ARC3_ANIMATION_IMAGE needs ARC3_ANIMATION: without it no chain is
            # captured, so this would silently do nothing. Stated rather than
            # left implicit - a knob that quietly has no effect is the failure
            # mode that costs the most time to diagnose.
            return
        mode = animation_image_mode()
        if mode == "off" or not summary:
            return
        if self._last_action_gameplay_changed is not False:
            return
        animation = summary.get("animation") or {}
        if not int(animation.get("transient_pixels") or 0):
            return
        chain = self._last_animation_chain or []
        cells = _animation_transient_cells(chain)
        if not cells:
            return
        if mode == "composite":
            part = animation_composite_part(
                chain, cells, border=_get_env_int("ARC3_NOOP_GUARD_BORDER", 4)
            )
            label = (
                "Animation image: the cells that changed during the animation, each in "
                "the last colour it held before reverting; everything else, including "
                "the border, is dimmed to dark navy (an off-palette colour). This is a "
                "composite of all cells that changed during the animation, not a single "
                "board state that the game displayed. A cell is drawn nearer its true "
                "colour the longer it held it, so a bright trail marks where something "
                "stayed and a uniformly dim one marks something that passed through. "
                "The action left the board area unchanged, so none of these cells "
                "appear in `current_frame`."
            )
        else:
            part, index = animation_peak_part(chain, cells)
            label = (
                f"Animation image (frame {index} of {len(chain) - 1}): the board partway "
                "through the animation, at the point where the most cells were showing "
                "something they later reverted. This is a real state the game displayed, "
                "but not one that survives into `current_frame`."
            )
        if part is not None:
            self._pending_animation_image = part
            self._pending_animation_image_label = label

    def _retained_function_context(self) -> str:
        """Optional prompt metadata must never interrupt game execution."""
        if not _persistent_functions():
            return ""
        try:
            notice = getattr(self, "_retained_clear_notice", "")
            signatures = [function_signature(src) for src in self._kept_functions.values()]
            parts = [notice] if notice else []
            if signatures:
                scope = {"levelup": "on this level", "turn": "in this turn", "game": "in this game"}[_persistent_functions_scope()]
                parts.append(
                    "Your retained functions from earlier Python calls " + scope + ": "
                    + ", ".join(signatures) + ". These are your previous definitions, "
                    "available to call directly. Their presence does not mean they are "
                    "correct; revise them if new evidence contradicts their assumptions."
                )
            self._retained_clear_notice = ""
            return "\n\n" + "\n".join(parts) if parts else ""
        except Exception:
            self._kept_functions = {}
            self._retained_clear_notice = ""
            return "\n\nYour retained functions were cleared after a retention error; redefine any you need."

    def _retained_sources(self):
        if not _persistent_functions():
            return None
        try:
            return list(self._kept_functions.values())
        except Exception:
            self._kept_functions = {}
            return []

    def _record_retained_functions(self, sandbox_result, payload) -> None:
        if not _persistent_functions():
            return
        try:
            previous = self._kept_functions
            kept = {}
            signatures = {}
            for item in sandbox_result.get("keepable_functions") or []:
                name, source = item["name"], item["source"]
                signature = function_signature(source)
                if signature.split("(", 1)[0] != name:
                    raise ValueError("invalid retained function metadata")
                kept[name] = source
                signatures[name] = signature
            notes = []
            scope = {"levelup": "on this level", "turn": "in this turn", "game": "in this game"}[_persistent_functions_scope()]
            for name, source in kept.items():
                if previous.get(name) != source:
                    notes.append("Retained your function " + signatures[name] + " for later Python calls " + scope + ".")
            rejected = sandbox_result.get("retention_rejected") or []
            repair_hints = _get_env_bool("ARC3_PERSISTENT_FUNCTIONS_REPAIR_HINTS", False)
            rejected_names = set()
            seen_rejections = set()
            for item in rejected:
                name, reason = str(item["name"]), str(item["reason"])
                hint = (str(item.get("hint") or "").strip() if repair_hints else
                        "Pass snippet-specific data as arguments or redefine it if needed.")
                rejection_key = (name, reason, hint)
                if rejection_key in seen_rejections:
                    continue
                seen_rejections.add(rejection_key)
                rejected_names.add(name)
                if name == "all":
                    notes.append("Your retained functions were cleared: " + reason + ".")
                    continue
                notes.append("Your function " + name + " was not retained: " + reason + "."
                             + (" " + hint if hint else ""))
            for name in previous.keys() - kept.keys() - rejected_names:
                notes.append("Your previous function " + name + " is no longer retained; redefine it if needed.")
            self._kept_functions = kept
            if notes:
                payload["function_retention"] = "\n".join(notes)
        except Exception:
            self._kept_functions = {}
            payload["function_retention"] = "Your retained functions were cleared after a retention error; redefine any you need."

    def _build_user_message(self, user_prompt: str, current_frame: Frame | None) -> dict[str, Any]:
        user_prompt += self._retained_function_context()
        image_part = current_grid_image_part(current_frame)
        diff_part = self._pending_diff_image
        diff_label = self._pending_diff_image_label
        animation_part = self._pending_animation_image
        animation_label = self._pending_animation_image_label
        gameover_images = list(self._pending_gameover_images)
        self._pending_diff_image = None
        self._pending_diff_image_label = ""
        self._pending_animation_image = None
        self._pending_animation_image_label = ""
        self._pending_gameover_images = []
        if animation_part is not None:
            # the animation image only ever fires when nothing inside the board
            # moved, which is precisely when the diff image shows the ticking
            # budget bar and nothing else - two images, one of them noise
            diff_part = None
        if image_part is None:
            return {"role": "user", "content": user_prompt}

        content: list[dict[str, Any]] = [
            {"type": "text", "text": f"{user_prompt}\n\nCurrent grid image (the board AFTER the automatic reset - this is what you act on now):" if gameover_images else f"{user_prompt}\n\nCurrent grid image:"},
            image_part,
        ]
        if diff_part is not None:
            content.append({"type": "text", "text": diff_label})
            content.append(diff_part)
        if animation_part is not None:
            content.append({"type": "text", "text": animation_label})
            content.append(animation_part)
        for label, part in gameover_images:
            content.append({"type": "text", "text": label})
            content.append(part)
        return {"role": "user", "content": content}


    def _memory_section_guidance(self, block_shown: bool) -> list[str]:
        """The section explanation, in upstream's slot just before the tool-call
        format guidance.

        Returns nothing when the mechanic is off. In 'full' mode the wording
        depends on whether a stored block was shown: the REPLACES warning is
        meaningless when there is nothing stored to replace, so a first turn or
        a post-wipe turn gets the format glossary without it."""
        if _memory_sections_disabled():
            return []
        if _wm_section_explanation_mode() == "upstream":
            return [_memory_section_guidance_upstream()]
        if not block_shown:
            return [_memory_section_guidance_empty()]
        if _memory_section_labels() is None:
            return [_MEMORY_SECTION_GUIDANCE_FULL]
        # a mode carrying a subset needs the subset described, not the fixed
        # two-section text: `medium` carries three
        return [
            _memory_section_guidance_empty()
            + " IMPORTANT: each labeled section you write REPLACES the stored version shown "
            "under 'Working world model carried from earlier turns' - write the COMPLETE "
            "updated section; any fact you omit is permanently lost."
        ]

    @staticmethod
    def _manual_reset_context_line(summary: dict[str, Any]) -> str | None:
        if not reset_exposed() or any(summary.get(flag) for flag in (
            "stale", "game_over", "run_complete", "done", "level_transition", "level_completed",
        )):
            return None
        actions = [str(name).strip().upper() for name in summary.get("executed_actions") or []]
        if "RESET" not in actions:
            return None
        if actions == ["RESET"]:
            return (
                "You deliberately reset the current level. The current frame shows "
                "the state returned by RESET."
            )
        return (
            "You deliberately reset the current level, then executed the subsequent "
            "actions listed above. The current frame shows their resulting state."
        )

    def _summary_turn_context_lines(self) -> list[str]:
        """What just happened, for the summary request.

        Deliberately not _build_user_prompt: that consumes _pending_diff_image,
        _pending_animation_image and the wipe-nudge state, so calling it twice
        would swallow images the real opener should carry. This reads
        _last_step_summary and mutates nothing - the stale flag is set by tool
        dispatch, never by building a prompt, so an extra read is safe.
        """
        summary = self._last_step_summary
        if not summary or summary.get("stale"):
            # nothing executed since the last opener; the block would say
            # "Executed actions: none" and repeat an unchanged state line
            return []
        lines: list[str] = []
        count = summary.get("executed_count")
        try:
            normalized = int(count) if count is not None else None
        except (TypeError, ValueError):
            normalized = None
        if not normalized:
            return []
        label = "action" if normalized == 1 else "actions"
        lines.append(f"The code executed {normalized} {label} in the previous sequence.")
        rendered = summary.get("executed_actions") or []
        if rendered:
            prefix = (
                "Executed actions:" if len(rendered) <= 10
                else "Executed actions (first 10):"
            )
            lines.append(f"{prefix} {', '.join(str(a) for a in rendered[:10])}.")
        changed = summary.get("gameplay_changed")
        if _report_gameplay_changed():
            if changed is True:
                lines.append("That sequence changed cells inside the board area.")
            elif changed is False and summary.get("board_changed"):
                lines.append(
                    "That sequence changed only cells near the edge - nothing inside "
                    "the board area moved."
                )
            elif changed is False:
                lines.append("That sequence produced no change anywhere on the board.")
        for stop in (summary.get("no_op_stops") or [])[:3]:
            lines.append(
                f"A batched sequence was stopped by the no-op guard: "
                f"{stop.get('action')!r} executed but changed nothing, and "
                f"{stop.get('skipped', 0)} later action(s) were skipped."
            )
        for stop in (summary.get("guard_stops") or [])[:3]:
            lines.append(
                f"The sequence was stopped before {stop.get('action')!r}: "
                f"{stop.get('reason', 'a guard refused it')}."
            )
        if summary.get("level_completed"):
            lines.append("You have progressed to a new level!")
        elif summary.get("game_over"):
            lines.append("The attempt ended and the level was reset.")
        else:
            lines.append(self._manual_reset_context_line(summary) or "You are still on the same level.")
        step = summary.get("action_num")
        level = summary.get("level")
        if step is not None and level is not None:
            lines.append(f"Current state: step {step}, level {level}.")
        return lines

    def _auto_probe_report_lines(self, current_level: int) -> list[str]:
        """The solver's level-1 probe table (see auto_probe), while that level lasts.

        Shown only when it is not already in the retained history: once in the
        first opener, and again if eviction or a rolled-back turn removed it.
        A shown-once flag would lose it on either.
        """
        report = getattr(self, "auto_probe_lines", None)
        if not report or current_level > getattr(self, "auto_probe_level", 1):
            return []
        marker = report[0][:60]
        for message in self._history_messages:
            content = message.get("content")
            parts = content if isinstance(content, list) else [content]
            for part in parts:
                text = part.get("text") if isinstance(part, dict) else part
                if isinstance(text, str) and marker in text:
                    return []
        return list(report)

    def _build_user_prompt(
        self,
        action_num: int,
        *,
        valid_actions: list[str] | None,
        current_frame: Frame | None = None,
        history_entries: list[HistoryEntry] | None = None,
        previous_step_summary: dict[str, Any] | None = None,
    ) -> str:
        history_entries = history_entries or []
        current_step = max(current_frame.step if current_frame is not None else 0, max(0, action_num)) + 1
        current_level = current_frame.level if current_frame is not None else 1
        summary_level = None
        if previous_step_summary is not None:
            try:
                summary_level = int(previous_step_summary.get("level"))
            except (TypeError, ValueError):
                summary_level = None
        if summary_level is not None:
            current_level = max(current_level, summary_level)
        observed_max_level = max(
            [current_level, *[entry.frame.level for entry in history_entries if entry.frame is not None]],
            default=current_level,
        )
        lines: list[str] = []
        if previous_step_summary and previous_step_summary.get("stale"):
            lines.extend(_stale_summary_lines(previous_step_summary))
        elif previous_step_summary:
            count = previous_step_summary.get("executed_count")
            try:
                normalized_count = int(count) if count is not None else None
            except (TypeError, ValueError):
                normalized_count = None
            action_label = "action" if normalized_count == 1 else "actions"
            lines.append(f"The code executed {normalized_count or 0} {action_label} in the previous sequence.")
            executed_actions = previous_step_summary.get("executed_actions")
            rendered_actions: list[str] = []
            if isinstance(executed_actions, list):
                rendered_actions = [str(name).strip() for name in executed_actions if str(name).strip()]
            if rendered_actions:
                action_prefix = "Executed actions (first 10):" if len(rendered_actions) > 10 else "Executed actions:"
                lines.append(f"{action_prefix} {', '.join(rendered_actions[:10])}.")
            else:
                lines.append("Executed actions: none.")
            if _report_gameplay_changed() and normalized_count:
                # Whether the sequence MOVED anything. The harness computes this
                # for every action - it drives the batch stop - and until now
                # said nothing about it: `_describe_last_outcome` produces the
                # sentence but is called nowhere, so neither its wording nor the
                # older one it replaced has ever reached a transcript. On a
                # single-action turn no per-action trace is built either, so the
                # model had no account of the outcome from any source.
                border = _get_env_int("ARC3_NOOP_GUARD_BORDER", 4)
                if previous_step_summary.get("gameplay_changed"):
                    lines.append("That sequence changed cells inside the board area.")
                elif previous_step_summary.get("board_changed"):
                    lines.append(
                        f"That sequence changed only cells within {border} of the edge "
                        "(usually a timer or remaining-steps bar) - nothing inside the "
                        "board area moved."
                    )
                else:
                    lines.append("That sequence produced no change anywhere on the board.")
            if _animation_enabled() and not previous_step_summary.get("game_over"):
                animation_line = describe_animation(
                    previous_step_summary.get("animation"),
                    action_display=previous_step_summary.get("animation_action"),
                    # the animated action's own flag, not the sequence's: an
                    # inert SPACE at the end of a batch that moved other things
                    # still hid everything it did
                    gameplay_changed=self._last_action_gameplay_changed,
                    timeline=_animation_timeline_enabled(),
                )
                if animation_line:
                    lines.append(animation_line)
                self._prepare_animation_image(previous_step_summary)
            no_op_stops = previous_step_summary.get("no_op_stops") or []
            for stop in no_op_stops[:3]:
                skipped = stop.get("skipped") or []
                skipped_text = (
                    f" The remaining {len(skipped)} action(s) {skipped!r} were skipped."
                    if skipped
                    else ""
                )
                lines.append(
                    f"Note: a batched sequence was stopped by the no-op guard: action "
                    f"{stop.get('action')!r} (position {stop.get('index')} in that batch) "
                    + (f"executed but produced no change inside the board area.{skipped_text} "
                       if _new_noop_wording() else
                       f"executed but changed nothing outside the border.{skipped_text} ")
                    +
                    "That action IS included in the executed list above, so it is the "
                    f"LAST of them - only the first {max(0, int(stop.get('index') or 1) - 1)} "
                    "executed action(s) moved anything. Do not count submitted or executed "
                    "actions to update your position; re-locate from current_frame."
                )
                trace = stop.get("trace") or []
                if trace:
                    lines.append("Per-action trace: " + "; ".join(str(t) for t in trace) + ".")
            if len(no_op_stops) > 3:
                lines.append(
                    f"({len(no_op_stops) - 3} further no-op guard stop(s) occurred this sequence.)"
                )
            guard_stops = previous_step_summary.get("guard_stops") or []
            for stop in guard_stops[:3]:
                skipped = stop.get("skipped") or []
                skipped_text = (
                    f" The remaining {len(skipped)} action(s) ({skipped!r}) were never executed."
                    if skipped
                    else ""
                )
                if stop.get("kind") == "stale_state":
                    # the exception said this in the turn where it happened;
                    # restated here because the plan being void is worth
                    # carrying into the turn that has to replace it
                    lines.append(
                        f"Note: the sequence was stopped before {stop.get('action')!r} - an "
                        + ("earlier action in that snippet produced no change inside the "
                           "board area, so the positions and "
                           if _new_noop_wording() else
                           "earlier action in that snippet changed nothing, so the positions and ")
                        +
                        f"paths it had computed may have been wrong.{skipped_text} Re-ground on "
                        "the current board before planning again."
                    )
                elif stop.get("kind") == "known_death":
                    lines.append(
                        f"Note: the sequence was stopped before {stop.get('action')!r} - that "
                        "action ended a previous attempt from that exact board state, so it was "
                        f"NOT executed and the attempt is intact.{skipped_text} Plan a different "
                        "move, or repeat it deliberately if you now believe it is safe."
                    )
                else:
                    lines.append(
                        f"Note: the sequence was stopped before {stop.get('action')!r} - that "
                        + ("action already produced no change inside the board area in that "
                           "exact board state, so it was "
                           if _new_noop_wording() else
                           "action already changed nothing in that exact board state, so it was ")
                        +
                        f"NOT executed and no action was spent.{skipped_text}"
                    )
                trace = stop.get("trace") or []
                if trace:
                    lines.append("Per-action trace: " + "; ".join(str(t) for t in trace) + ".")
            if len(guard_stops) > 3:
                lines.append(
                    f"({len(guard_stops) - 3} further guard stop(s) occurred this sequence.)"
                )
            if previous_step_summary.get("run_complete"):
                lines.append("You have completed the run!")
            elif previous_step_summary.get("level_transition"):
                lines.append(
                    LEVEL_START_USER_PROMPT
                    if _level_transfer_guidance_enabled()
                    else "You have progressed to a new level!"
                )
            else:
                lines.append(
                    self._manual_reset_context_line(previous_step_summary)
                    or "You are still on the same level."
                )
            if previous_step_summary.get("game_over"):
                lines.append(
                    "GAME OVER occurred during the previous sequence. This means the "
                    "attempt FAILED (it does NOT mean the run is finished). The game has "
                    "already been automatically RESET: the current level restarted from "
                    "its initial state and previously completed levels are kept. A game "
                    "over is caused either by the specific action taken (e.g., moving "
                    "onto a hazard cell) or by a step/time budget running out, which is "
                    "usually indicated by a bar at the grid border that shrinks with each "
                    "action. Before acting again, diagnose the cause by inspecting the "
                    "board right before the death. `history[-1].action` is 'RESET', "
                    "`history[-2].action` is the fatal action, `history[-2].frame` is the "
                    "death frame (the board after the fatal action), and "
                    "`history[-3].frame` is the board immediately before "
                    "the fatal action — check what cell the fatal move targeted there, and "
                    "inspect the budget bar in the death frame. BAR RULE: the bar "
                    "shrinking with each action is NORMAL — a slightly shorter bar is "
                    "NOT evidence of the cause of death. The budget caused the death "
                    "ONLY if the bar is fully (or almost fully) depleted in the death "
                    "frame (`history[-2].frame`). If it IS depleted: you must reach the "
                    "goal in fewer actions, or look for a way to restore the bar (e.g., "
                    "collecting an item that refills it). If the bar clearly had budget "
                    "remaining: the fatal action itself caused the death (hazard or "
                    "forbidden move) — do NOT retry the same path; a DIFFERENT approach "
                    "must be tried. Record the diagnosed cause in your world model."
                )
                if _animation_enabled():
                    # after the BAR RULE, which tells the model how to read
                    # history[-2] and history[-3]; this says what those two
                    # frames cannot contain
                    death_animation_line = describe_animation(
                        previous_step_summary.get("animation"),
                        action_display=previous_step_summary.get("animation_action"),
                        # the animated action's own flag, not the sequence's: an
                    # inert SPACE at the end of a batch that moved other things
                    # still hid everything it did
                    gameplay_changed=self._last_action_gameplay_changed,
                        game_over=True,
                        timeline=_animation_timeline_enabled(),
                    )
                    if death_animation_line:
                        lines.append(death_animation_line)
                fatal_action = str(previous_step_summary.get("fatal_action") or "")
                if fatal_action:
                    lines.append(
                        f"The game over occurred immediately after action {fatal_action!r}. "
                        "Apply the BAR RULE above to decide: bar fully depleted in the "
                        "death frame means budget death (shorter route or bar restore "
                        "needed); bar with budget remaining means this action itself was "
                        "fatal and a DIFFERENT approach is required."
                    )
                game_over_trace = previous_step_summary.get("game_over_trace") or []
                if game_over_trace:
                    lines.append(
                        "Per-action trace of the previous sequence: "
                        + "; ".join(str(t) for t in game_over_trace)
                        + "."
                    )
                self._prepare_gameover_images(
                    history_entries, previous_step_summary
                )
                if _get_env_bool("ARC3_GAMEOVER_DIFF", False):
                    # read at call time so this patch's hunks stay clear of the
                    # env-knob block that other patches extend
                    lines.extend(
                        self._game_over_diff_lines(
                            history_entries, previous_step_summary
                        )
                    )
                lines.append(
                    "Do NOT re-submit this same series of moves: the game is "
                    "deterministic, so replaying the sequence that just killed you "
                    "will kill you again. Change the plan BEFORE the fatal point - a "
                    "different route, action, or timing. If you believe the same "
                    "moves 'should' work, that belief is exactly what this death "
                    "falsified; record the correction in your world model instead "
                    "of retesting it."
                )
        elif (current_frame is not None and current_frame.step > 0) or action_num > 0:
            lines.append("No previous action sequence was captured.")
        else:
            lines.append("No previous sequence has been executed yet.")
        if self._pending_wm_rebuild_reason and _wm_rebuild_nudge_enabled():
            lines.append(
                f"IMPORTANT: your memory sections were reset by {self._pending_wm_rebuild_reason} "
                "(cross-level notes were kept). Rebuild them NOW: write fresh "
                # naming a section a restricted mode drops would ask for a
                # rebuild that is parsed and then thrown away
                f"{_section_list_phrase(_advertised_section_labels())} sections for this "
                "level in your next message, before or alongside your tool call."
            )
        elif (
            _wm_revise_nudges_enabled()
            and getattr(self, "_pending_wm_revise_reason", "") in ("level", "death")
        ):
            _revise_reason = getattr(self, "_pending_wm_revise_reason", "")
            lines.append(
                _wm_revise_hint(
                    _revise_reason,
                    getattr(self, "_wm_revise_hint_shown", False),
                )
            )
            self._wm_revise_hint_shown = True
        elif _WM_NUDGE_TURNS > 0 and self._turns_without_wm_update >= _WM_NUDGE_TURNS:
            lines.append(
                f"You have not updated any memory section for "
                f"{self._turns_without_wm_update} analysis steps. Review what recent "
                "action traces, stops, and observations taught you, and write updated "
                "sections now - knowledge that is not written down is eventually lost "
                "to context eviction."
            )
        lines.extend(self._prepare_auto_diff(current_frame, previous_step_summary))
        state_line = f"Current state: step {current_step}, level {current_level}"
        if observed_max_level > current_level:
            state_line += f" out of observed max level {observed_max_level} so far"
        state_line += "."
        lines.extend(
            [
                state_line,
                f"Valid actions right now: {_format_valid_action_line(valid_actions)}.",
            ]
        )
        lines.extend(self._auto_probe_report_lines(current_level))
        # recording is independent of display: the guard needs the ledger even
        # when the advisory lines are switched off
        self._death_ledger_observe(history_entries, previous_step_summary)
        if _get_env_bool("ARC3_DEATH_LEDGER", False):
            # read at call time so this patch's hunks stay clear of the
            # env-knob block that other patches extend; placed directly after
            # the valid-actions line so the fatal continuations sit adjacent
            # to the action choice they constrain
            lines.extend(
                self._death_ledger_lines(
                    current_frame, history_entries, previous_step_summary
                )
            )
        lines.extend(
            [
                "Only tool: `python`. It receives `current_frame`, `previous_frame`, `history`, `transitions`, `last_transition`, `valid_actions`, `last_action_call_result`, `frame_diff(before, after)`, and `action(actions)`.",
                "Only letter-coded board views and lightweight metadata are exposed; raw numeric color IDs are not available.",
                "Keep tool output compact: use `current_frame.segmentation` as the primary view, and `current_frame.ascii` only for a small specific region; never print full boards.",
                "For the most recent change, compare `previous_frame` to `current_frame`, or `last_transition.before_frame` to `last_transition.after_frame`; `history[-1].frame` is the current frame, not the previous one.",
            ]
        )
        # Upstream states four things about the world model here, and with
        # ARC3_MEMORY_SECTIONS=off three of them are false or unanswerable: the
        # block they refer to is suppressed, so "Below you are provided with the
        # current world model" is followed by nothing, and the model is told in
        # capitals to always give a revised version with nowhere to put it.
        # Observed as a drop in replies carrying any assistant text at all,
        # 76% to 43% - a broken instruction, not an absent one, which is why
        # `off` was never a clean control.
        #
        # The first sentence loses only its middle clause: "from the newest
        # history" attaches to inspecting the evidence just as well, so the
        # remainder still says use Python, inspect, and score against the goal.
        if _memory_sections_disabled():
            lines.append(
                "Use Python to inspect the evidence from the newest history, and search or score candidate actions or short sequences against the current goal as you currently understand it."
            )
        else:
            lines.extend(
                [
                    "Use Python to inspect the evidence, refine that world model from the newest history, and search or score candidate actions or short sequences against the current goal as you currently understand it.",
                    "Maintain a compact working world model of what the current level seems to contain, what actions appear to do, what the goal seems to be, what is still uncertain, and what plan currently looks best.",
                    "Below you are provided with the current world model from the previous turn. The default behavior is to copy it and add or remove things based on the evidence that you gathered. BEFORE EXECUTING NEW ACTIONS YOU MUST ALWAYS GIVE THE REVISED VERSION OF THE WORLD MODEL.",
                ]
            )
        if not _dedupe_multicall_line():
            # kept mid-opener only for the upstream wording; the surviving copy
            # in the closing block is where a note about calling action() twice
            # belongs, next to the instruction to call it at all
            lines.append(
                "You may call `action(actions)` more than once in one Python snippet if your search or control loop needs it, "
                "but stop immediately if a result reports `game_over`, `run_complete`, `level_completed`, or `done`."
            )
        knowledge_lines = self._summarized_knowledge_lines()
        if knowledge_lines:
            lines.extend(knowledge_lines)
            lines.append("end of world model. ")
        if action_num == 0:
            lines.append(
                "Ground yourself in `current_frame` before acting, but start with a compact structural summary rather than restating the full frame."
            )
        else:
            lines.append(
                "Focus on what changed most recently in `history`, update the target environment change if needed, and separate gameplay-object changes from HUD-only changes."
            )
        lines.extend(
            [
                "When ready, call `action(actions)` from inside the `python` tool with the best valid action or ordered batch selected by your code. If your code has found a reliable short sequence, prefer batching it in one call.",
                "You may call `action(actions)` more than once in one Python snippet if your search or control loop needs it.",
                *self._memory_section_guidance(bool(knowledge_lines)),
                TOOL_CALL_FORMAT_GUIDANCE,
            ]
        )
        if "MOUSE" in _normalize_valid_actions(valid_actions):
            lines.append("If you use MOUSE, include integer row and col arguments.")
        if _opener_commit_hint():
            # last line of the opener, where the resumption prompt sits relative
            # to the reply it influences
            lines.append(
                "When your evidence supports a choice, call `action(actions)` with the "
                "sequence it supports. Probe with a single action when you genuinely "
                "need to distinguish two possibilities; otherwise batch the route you "
                "have."
            )
        return "\n".join(lines)

    def _tools(self, state_path: Path) -> list[dict[str, Any]]:
        self._ensure_session(state_path)
        return [
            {
                "type": "function",
                "function": {
                    "name": "python",
                    "description": _PYTHON_TOOL_DESCRIPTION,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": (
                                    "Python code to run. The snippet is ephemeral and is not saved across tool calls."
                                ),
                            },
                        },
                        "required": ["code"],
                    },
                },
            }
        ]

    def _harness_template_kwargs(self) -> dict[str, Any]:
        """Chat-template kwargs the HARNESS sets per request.

        Shared by the request builder and the request log so the log records
        what was actually sent rather than a reconstruction - which matters
        for the effort ladder, where the value changes mid-run, and for
        preserve_thinking, where the question is whether it reaches the server
        at all."""
        kwargs: dict[str, Any] = {}
        preserve_thinking = _preserve_thinking_kwarg()
        if preserve_thinking is not None:
            kwargs["preserve_thinking"] = preserve_thinking
        ladder = _reasoning_effort_ladder()
        rung = getattr(self, "_reasoning_effort_rung", -1)
        if ladder and 0 <= rung < len(ladder):
            kwargs["reasoning_effort"] = ladder[rung]
        return kwargs

    def _chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        request_timeout_seconds: float | None = None,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
    ) -> _ChatCompletionResult:
        payload = build_chat_payload(
            provider=self._model.provider,
            model=self._model.model_id,
            messages=messages,
            max_tokens=self._max_output_tokens if max_tokens is None else max_tokens,
            temperature=_LOCAL_ANALYZER_TEMPERATURE,
            top_p=_LOCAL_ANALYZER_TOP_P,
            top_k=_LOCAL_ANALYZER_TOP_K,
            thinking=(
                bool(_LOCAL_ANALYZER_ENABLE_THINKING)
                if enable_thinking is None
                else enable_thinking
            ),
            tools=tools,
            tool_choice=_request_tool_choice(tools),
            seed=_LOCAL_ANALYZER_SEED,
        )
        provider_prefs = _openrouter_provider_prefs(self._model.provider)
        if provider_prefs:
            payload["provider"] = provider_prefs
        # merge rather than assign: the payload builder may already have set
        # chat_template_kwargs (enable_thinking), and replacing the dict would
        # drop it - as a request-level dict can also drop a server-side default
        template_kwargs = dict(payload.get("chat_template_kwargs") or {})
        template_kwargs.update(self._harness_template_kwargs())
        if enable_thinking is not None:
            # after the merge: _harness_template_kwargs does not set
            # enable_thinking today, but an override that a later edit could
            # silently undo is the bug that keeps recurring here
            template_kwargs["enable_thinking"] = bool(enable_thinking)
        if template_kwargs:
            payload["chat_template_kwargs"] = template_kwargs
        payload["messages"] = _strip_control_keys(
            _apply_summary_visibility(
                payload.get("messages") or [],
                evicted=getattr(self, "_has_evicted", False),
            )
        )
        def post_chat(request_payload: dict[str, Any]) -> requests.Response:
            return requests.post(
                f"{self._model.base_url.rstrip('/')}/chat/completions",
                headers=self._headers(),
                json=request_payload,
                # (connect, read): one scalar gives the connect phase the whole
                # read budget, so an unreachable host stalls a single attempt
                # for the rest of the game
                timeout=(
                    _CONNECT_TIMEOUT_SECONDS,
                    request_timeout_seconds if request_timeout_seconds is not None else self._timeout,
                ),
                stream=bool(request_payload.get("stream")),
            )

        # Per agent, not per process: every game meets the cold server on its
        # own first request, so one shared flag would grant the grace to
        # whichever game happened to go first and let the rest fail fast.
        initial_grace = 0.0
        if not self._http_initial_grace_used:
            initial_grace = _env_float("ARC3_HTTP_RETRY_INITIAL_SECONDS", 0.0)
        _diag(self, "request")
        try:
            response = _post_with_retries(
                lambda: post_chat(payload),
                retries=_env_int("ARC3_HTTP_RETRIES", 3),
                base_seconds=_env_float("ARC3_HTTP_RETRY_BASE_SECONDS", 5.0),
                max_seconds=_env_float("ARC3_HTTP_RETRY_MAX_SECONDS", 5.0),
                initial_seconds=initial_grace,
            )
        finally:
            # consumed whether the call succeeded or failed: the grace covers
            # the server coming up, and by now it either did or will not
            self._http_initial_grace_used = True
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text.strip()
            message = f"{exc}"
            if detail:
                message += f" | response: {detail}"
            raise requests.RequestException(message) from exc
        if getattr(response, "status_code", 200) >= 400:
            detail = response.text.strip()
            message = f"{response.status_code} Error"
            if detail:
                message += f" | response: {detail}"
            raise requests.RequestException(message)
        if payload.get("stream"):
            # decode_unicode=True decodes each transport chunk INDEPENDENTLY, so a
            # multi-byte UTF-8 sequence straddling a chunk boundary is split into
            # two broken halves - an em-dash (E2 80 94) becomes "\u00e2" plus
            # fragments. Most characters survive, which is why short probes look
            # clean and only long streamed replies are corrupted. The assembler
            # already decodes bytes itself, and a complete SSE line is always a
            # whole UTF-8 sequence, so hand it raw bytes instead.
            payload = assemble_streamed_chat_response(
                response.iter_lines(decode_unicode=False)
            )
        else:
            payload = response.json()
        choices = payload.get("choices", [])
        if not choices:
            detail = payload.get("error") or {
                k: v for k, v in payload.items() if k not in ("choices",)
            }
            raise requests.RequestException(
                f"server returned no choices | body: {str(detail)[:600]}"
            )
        choice = choices[0]
        return _ChatCompletionResult(
            message=_extract_leaked_tool_calls(choice.get("message", {})),
            finish_reason=str(choice.get("finish_reason", "") or ""),
            usage=payload.get("usage"),
            served_by=str(payload.get("provider", "") or ""),
        )

    def _trim_tool_text(self, text: str) -> tuple[str, bool]:
        """Keep both ends and drop the middle.

        Truncating the tail loses the half that matters. An `error` field ends
        with `SomeError: <the explanation>` after the traceback, so a deep
        traceback pushed the explanation out entirely. And a snippet's `stdout`
        typically prints the state, acts, then prints the state again - so the
        tail holds the result and the head holds the setup.

        Weighted toward the HEAD: a snippet's output is mostly sequential
        reasoning that reads from the top, so two thirds there. The remaining
        third is ample for what has to survive at the end - the longest refusal
        message measures ~550 characters and a user-frame traceback ~250, well
        inside a 1365-character tail at the default budget.
        """
        limit = self._tool_output_chars
        if len(text) <= limit:
            return text, False
        omitted = len(text) - limit
        if not _get_env_bool("ARC3_MIDDLE_TRUNCATION", False):
            return f"{text[:limit]}\n... [truncated {omitted} chars]", True
        tail = limit // 3
        head = limit - tail
        return (
            f"{text[:head]}\n... [truncated {omitted} chars from the middle] ...\n"
            f"{text[-tail:]}",
            True,
        )

    def _summarize_planned_actions(self, value: Any) -> Any:
        if isinstance(value, dict):
            compacted = {
                key: self._summarize_planned_actions(item)
                for key, item in value.items()
            }
            planned_actions = compacted.pop("planned_actions", None)
            if isinstance(planned_actions, list):
                compacted["planned_action_count"] = len(planned_actions)
                action_result = compacted.get("action_result")
                if isinstance(action_result, dict):
                    executed_count = action_result.get("executed_count")
                    try:
                        compacted["executed_action_count"] = int(executed_count)
                    except (TypeError, ValueError):
                        compacted["executed_action_count"] = 1 if action_result.get("executed") else 0
            return compacted
        if isinstance(value, list):
            return [self._summarize_planned_actions(item) for item in value]
        return value

    def _render_tool_payload(self, payload: dict[str, Any], *, truncate_fields: tuple[str, ...] = ()) -> str:
        result = self._summarize_planned_actions(dict(payload))
        truncated = False
        for field in truncate_fields:
            value = result.get(field)
            if isinstance(value, str):
                value, suppressed_rows = _suppress_board_dumps(value)
                if suppressed_rows:
                    result["board_print_suppressed_rows"] = suppressed_rows
                result[field], field_truncated = self._trim_tool_text(value)
                truncated = truncated or field_truncated
        if truncated:
            result["truncated"] = True
            result["truncation_note"] = (
                f"Tool output was cut off to stay within the ~{self._tool_output_tokens}-token response budget."
            )
        return json.dumps(result, indent=2)

    def _normalize_python_actions(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, dict):
            items = [value]
        elif isinstance(value, (list, tuple)):
            items = list(value)
        else:
            raise TypeError(
                "action(actions) expects a string, an action object, or a list of action strings/objects."
            )
        if not items:
            raise ValueError("action(actions) requires at least one action.")

        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(items, start=1):
            if isinstance(item, str):
                action_name = item.strip()
                if not action_name:
                    raise ValueError(f"Action {index} is empty.")
                normalized.append({"action": action_name})
                continue
            if isinstance(item, dict):
                action_name = str(item.get("action", "")).strip()
                if not action_name:
                    raise ValueError(f"Action {index} is missing an `action` field.")
                entry = {"action": action_name}
                if action_name.upper() == "MOUSE" and ("x" in item or "y" in item):
                    raise ValueError(f"Action {index} uses legacy MOUSE x/y fields; use row and col.")
                if "row" in item:
                    entry["row"] = item.get("row")
                if "col" in item:
                    entry["col"] = item.get("col")
                normalized.append(entry)
                continue
            raise TypeError(f"Action {index} must be a string or a dict.")
        return normalized

    def _fatal_prefix_match(
        self, state_hash: str, remaining_sigs: list[str]
    ) -> tuple[int, str] | None:
        """Does a recorded fatal sequence continue from here?

        Returns ``(offset, action)`` when executing ``remaining_sigs`` from
        this board would replay a recorded attempt into its death - offset 0
        meaning the very next action is the fatal one. Returns None otherwise.

        Generalizes the single-action check: an attempt recorded as actions
        ``A[0..n-1]`` with pre-action hashes ``H[0..n-1]`` is fatal from here
        if some ``H[k]`` equals this board AND ``A[k:]`` is a prefix of what
        the model is about to run. The environment is deterministic and a
        death resets to level start, so replaying ``A[k:]`` from ``H[k]``
        reproduces ``H[k+1] ...`` and ends the attempt.

        Catching it at the FIRST position of such a batch is what makes the
        block worth having: the per-position check alone would let the harmless
        prefix run and refuse only the final move, spending those actions to
        arrive somewhere the model has already been. A model that genuinely
        wants the prefix can submit it without the fatal tail, which no longer
        matches and is not blocked.
        """
        index = getattr(self, "_death_ledger_index", None)
        attempts = getattr(self, "_death_ledger_attempts", None)
        if not index or not attempts or not remaining_sigs:
            return None
        for attempt_idx, position in index.get(state_hash, ()):  # type: ignore[union-attr]
            try:
                actions = attempts[attempt_idx]["actions"]
            except (IndexError, KeyError, TypeError):
                continue
            tail = [action_signature(a) for a in actions[position:]]
            if not tail or len(tail) > len(remaining_sigs):
                continue
            if remaining_sigs[: len(tail)] == tail:
                return len(tail) - 1, tail[-1]
        return None

    def _action_guard_hook(
        self,
        grid: Any,
        action_name: str,
        action_data: dict[str, Any],
        level: Any = None,
        remaining: list[tuple[str, dict[str, Any]]] | None = None,
    ) -> tuple[str, str] | None:
        """Called by the solver before each action of a batch. Returns
        ``(stop_reason, detail)`` to refuse, or None to proceed."""
        guard = self._noop_repeat_guard
        if guard is None or not grid:
            return None
        if not _guards_active(level):
            # level 1 is where the model probes hardest and where an action is
            # worth least, so the repeat guards can be told to start later. The
            # guard still WATCHES - note_level and the state hash below run
            # either way - so that by the time it starts refusing it already
            # knows what has been tried.
            guard.note_level(level)
            self._noop_guard_pre_hash = _interior_state_hash(
                [list(row) for row in grid]
            )
            return None
        rows = [list(row) for row in grid]
        state_hash = _interior_state_hash(rows)
        guard.note_level(level)
        # the solver passes ENGINE names (ACTION2); every other consumer - the
        # death ledger, the transcript, the model itself - speaks model names
        # (DOWN), so translate here or the guard both mismatches the ledger and
        # reports "ACTION2" to a model that has never seen that label
        action_sig = action_signature(
            {"action": to_model_action(action_name), **(action_data or {})}
        )
        self._noop_guard_pre_hash = state_hash
        if self._noop_guard_deaths_enabled and not self._guard_death_override_batch:
            # with lookahead off, only THIS action is considered, so a match
            # can be found solely at offset 0 - the per-position behaviour
            window = (
                (remaining or [(action_name, action_data)])
                if _fatal_lookahead_enabled()
                else [(action_name, action_data)]
            )
            remaining_sigs = [
                action_signature({"action": to_model_action(name), **(data or {})})
                for name, data in window
            ]
            match = self._fatal_prefix_match(state_hash, remaining_sigs)
            if guard.should_block_death(
                state_hash, action_sig, lambda _h, _a: match is not None
            ):
                offset, fatal_action = match  # type: ignore[misc]
                if offset == 0:
                    detail = (
                        f"{fatal_action} ended a previous attempt on this level from this "
                        "exact board state, so it was NOT executed."
                    )
                else:
                    detail = (
                        "this sequence replays a path that ended a previous attempt: "
                        f"{fatal_action} is fatal {offset} action(s) from here, so NOTHING "
                        "was executed and no actions were spent. Submit a different "
                        "continuation, or the same prefix without the fatal tail."
                    )
                return (
                    "known_death",
                    detail
                    + " This refusal is not catchable and ended the snippet, so anything "
                    "after the call did not run. Issuing the same action again as the "
                    "very next action will override the block and execute the action. "
                    "That is only worth doing if you have a specific reason to believe "
                    "it will not end the attempt this time. Otherwise, plan a different "
                    "continuation from this state.",
                )
            # the override fired: let the rest of THIS call run without further
            # death checks, or the model would have to insist again at every
            # position of a sequence it has already deliberately confirmed
            if match is not None:
                self._guard_death_override_batch = True
        if (
            self._noop_guard_noops_enabled
            and not (reset_exposed() and action_sig == "RESET")
            and guard.should_block(state_hash, action_sig)
        ):
            return (
                "known_noop",
                (f"{action_sig} already produced no change inside the board area in this "
                   "exact board state "
                   if _new_noop_wording() else
                   f"{action_sig} already changed nothing in this exact board state ")
                +
                "earlier on this level, so it was NOT executed and no action was "
                "spent. This refusal is not catchable and ended the snippet, so "
                "anything after the call did not run - a retry loop cannot probe "
                "past it. It applies to that one action from this one board state; "
                "it is not a constraint on sequences or batch size. Issuing the same "
                "action again as the very next action will override the block and "
                "execute the action. That is only worth doing if you have a specific "
                "reason to believe it will behave differently now. Otherwise, plan a "
                "different continuation from this state.",
            )
        if (
            _repeat_state_guard_enabled()
            and (state_hash, action_sig) in self._repeat_state_seen
        ):
            # after known_noop deliberately: a repeated no-op is better reported
            # as a no-op, which also keeps its override intact. Reaching here
            # means the action DID change the board and still returned to a
            # state this snippet has already acted from.
            return (
                "repeated_action_in_state",
                f"{action_sig} was already run from this exact board state earlier "
                "in this snippet, so what it does here is known and it was NOT "
                # not "no action budget was spent": by construction this fires
                # after the snippet has already spent budget on the actions that
                # got it here, and the model reads that phrase as covering the
                # whole snippet
                "executed. A sequence that returns to "
                # examples, not a diagnosis: the harness knows the board
                # repeated, not why, and stating the cause would have the model
                # looking for a wrapping cursor it may not have
                "a board it has already acted from is not advancing (e.g. a cursor "
                "that wraps around, or a loop stepping the wrong way past its "
                "target). "
                "Nothing after the call ran. Re-read `current_frame`, work out why "
                "the sequence is not converging, and write a different snippet - "
                "this block is per snippet, so the next one starts with no history.",
            )
        guard.note_executed(state_hash, action_sig)
        self._guard_hook_trace.append((state_hash, action_sig))
        return None

    def _record_guard_observations(
        self,
        guard: NoopRepeatGuard,
        raw_payload: dict[str, Any],
        compact_payload: dict[str, Any],
    ) -> None:
        """Record every executed position of this call, not just the first.

        The guard hook fires once per position immediately before that action
        runs, so `_guard_hook_trace` holds the board hash each action actually
        executed against - including the intermediate states of a batch, which
        appear nowhere in the returned payload. The solver reports outcomes as
        `gameplay_changed_per_action`, index-aligned with `executed_actions`
        and therefore with the trace; a single-action call has no such list, so
        the aggregate verdict is used instead.

        Zipping to the shorter of the two is deliberate: a batch stopped early
        (blocked, fatal, invalid) leaves one more trace entry than there are
        outcomes, and an unmatched entry must not be recorded against a verdict
        that belongs to a different action.
        """
        trace = list(self._guard_hook_trace)
        self._guard_hook_trace = []
        if not trace:
            return
        verdicts = raw_payload.get("gameplay_changed_per_action")
        if not isinstance(verdicts, list) or not verdicts:
            if len(trace) != 1 or not compact_payload.get("executed"):
                return
            verdicts = [compact_payload.get("gameplay_changed")]
        for (state_hash, action_sig), changed in zip(trace, verdicts):
            if reset_exposed() and action_sig == "RESET":
                # Retain its position in the trace so following verdicts stay
                # aligned, but do not learn that a bar-only reset is ineffective.
                continue
            # unconditional: known_noop reaches a repeated no-op first, so
            # filtering on `changed` here would buy nothing
            self._repeat_state_seen.add((state_hash, action_sig))
            guard.observe(state_hash, action_sig, gameplay_changed=changed)

    def _compact_action_result(self, payload: dict[str, Any]) -> dict[str, Any]:
        compact = {
            "executed": bool(payload.get("executed")),
            "action_num": payload.get("action_num"),
            "level": payload.get("level"),
            "score": payload.get("score"),
            # `reward` is not carried: it is (levels_completed - previous) /
            # number_of_levels, so it reads 0.0 on every action that does not
            # finish a level. A model seeing 0.0 on almost every action can
            # reasonably take it for a verdict on the action, which it is not,
            # and `level_completed` in this same dict already says the thing it
            # was trying to say. The solver still computes it for the viewer
            # and the raw payload.
            "state": payload.get("state"),
            "valid_actions": payload.get("valid_actions", []),
            "board_changed": bool(payload.get("board_changed")),
            "done": bool(payload.get("done")),
            "level_completed": bool(payload.get("level_completed")),
            "game_over": bool(payload.get("game_over")),
            "run_complete": bool(payload.get("run_complete")),
            "action_display": payload.get("action_display") or payload.get("action_name"),
        }
        if "gameplay_changed" in payload:
            compact["gameplay_changed"] = bool(payload.get("gameplay_changed"))
        executed_actions = payload.get("executed_actions")
        if isinstance(executed_actions, list) and executed_actions:
            compact["executed_actions"] = [str(action).strip() for action in executed_actions if str(action).strip()]
        elif compact.get("action_display"):
            compact["executed_actions"] = [str(compact["action_display"]).strip()]
        batch_size = int(payload.get("requested_count") or payload.get("executed_count") or 1)
        if batch_size > 1 or bool(payload.get("stopped_early")):
            compact["requested_count"] = payload.get("requested_count", batch_size)
            compact["executed_count"] = payload.get("executed_count", batch_size)
            compact["stopped_early"] = bool(payload.get("stopped_early"))
        # blocked_action names the refused action for the next turn's opener;
        # without it the model was told "the sequence was stopped before None"
        for passthrough_key in ("no_op_action_index", "no_op_action", "skipped_actions", "gameplay_changed_per_action", "action_trace", "no_op", "fatal_action", "blocked_action", "action_echo", "animation"):
            if passthrough_key in payload:
                compact[passthrough_key] = payload.get(passthrough_key)
        if payload.get("stop_reason"):
            compact["stop_reason"] = payload.get("stop_reason")
        if payload.get("stop_detail"):
            compact["stop_detail"] = payload.get("stop_detail")
        for timing_key in ("run_elapsed_seconds", "time_remaining_seconds"):
            if timing_key in payload:
                compact[timing_key] = payload.get(timing_key)
        # the session's own clock, which is also what decides when the game
        # stops - so the endgame phase cannot drift away from the deadline it
        # is meant to anticipate. It starts at session construction, so a game
        # waiting on the gate is already ageing, which is the same convention
        # max_runtime_s_per_game uses.
        elapsed = payload.get("run_elapsed_seconds")
        if isinstance(elapsed, (int, float)):
            self._game_elapsed_seconds = float(elapsed)
        if payload.get("error"):
            compact["error"] = payload.get("error")
        return compact

    def _run_python_tool(self, state_path: Path, arguments: dict[str, Any]) -> _ToolDispatchResult:
        self._ensure_session(state_path)
        code = str(arguments.get("code", "")).rstrip()
        if not code:
            return _ToolDispatchResult(json.dumps({"error": "python requires a non-empty `code` string."}, indent=2))
        try:
            compile(code, "<python_tool>", "exec")
        except SyntaxError as exc:
            return _ToolDispatchResult(json.dumps({"error": f"Python syntax error: {exc}"}, indent=2))

        current_frame, history_entries = load_runtime_state(state_path)
        valid_actions = list(_normalize_valid_actions(self._current_valid_actions))

        def _serialized_runtime_state(
            *,
            next_valid_actions: list[str] | None = None,
            last_action_call_result: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            refreshed_frame, refreshed_history = load_runtime_state(state_path)
            current_frame_payload = _ascii_frame_view_payload(refreshed_frame)
            if isinstance(next_valid_actions, list):
                sanitized_actions = [str(item).strip() for item in next_valid_actions if str(item).strip()]
            else:
                sanitized_actions = list(valid_actions)
            persisted_action_result = (
                last_action_call_result
                if isinstance(last_action_call_result, dict)
                else self._last_action_call_result
            )
            return {
                "current_frame": current_frame_payload,
                "history": _ascii_history_view_payload(refreshed_history),
                "valid_actions": sanitized_actions,
                "death_ledger": (
                    self._death_ledger_payload()
                    if _get_env_bool("ARC3_DEATH_LEDGER", False)
                    else None
                ),
                "last_action_call_result": (
                    dict(persisted_action_result)
                    if isinstance(persisted_action_result, dict)
                    else {}
                ),
            }

        terminal_action_result: dict[str, Any] | None = None
        snippet_executed_actions = 0

        def _handle_animation(request: dict[str, Any]) -> dict[str, Any]:
            if self._step_env_callback is None:
                return {"error": "Animation frames are not available in this session."}
            raw = self._step_env_callback({"query": "animation"})
            record = raw.get("record") if isinstance(raw, dict) else None
            record = record if isinstance(record, dict) else None
            if str(request.get("kind") or "") == "frames":
                return build_animation_frames(record)
            if not _animation_timeline_enabled():
                return {}
            return build_animation_view(record)

        def _handle_action(
            actions: list[dict[str, Any]], *, stale_after: str | None = None
        ) -> dict[str, Any]:
            nonlocal terminal_action_result, snippet_executed_actions
            if self._step_env_callback is None:
                raise RuntimeError("action(actions) is not available in this session.")
            normalized_actions = self._normalize_python_actions(actions)
            if terminal_action_result is not None:
                reason = _terminal_action_reason(terminal_action_result) or "terminal_state"
                compact_payload = {
                    "executed": False,
                    "action_num": terminal_action_result.get("action_num"),
                    "level": terminal_action_result.get("level"),
                    "score": terminal_action_result.get("score"),
                    "state": terminal_action_result.get("state"),
                    "valid_actions": [],
                    "board_changed": False,
                    "done": bool(terminal_action_result.get("done")),
                    "level_completed": bool(terminal_action_result.get("level_completed")),
                    "game_over": bool(terminal_action_result.get("game_over")),
                    "run_complete": bool(terminal_action_result.get("run_complete")),
                    "requested_count": len(normalized_actions),
                    "executed_count": 0,
                    "stopped_early": True,
                    "stop_reason": f"previous_{reason}",
                    "stop_detail": _terminal_action_stop_detail(reason),
                }
                self._last_action_call_result = dict(compact_payload)
                return {
                    "action_result": compact_payload,
                    "state": _serialized_runtime_state(
                        next_valid_actions=[],
                        last_action_call_result=compact_payload,
                    ),
                }
            if reset_exposed():
                reset_positions = [
                    index for index, item in enumerate(normalized_actions)
                    if to_engine_action(item.get("action")) == "RESET"
                ]
                if reset_positions and (snippet_executed_actions or reset_positions != [0]):
                    detail = (
                        "RESET must be the first executed game action in a Python snippet. "
                        "This call executed no actions. Start a new snippet with RESET "
                        "as its first action; inspection or computation may come before it."
                    )
                    refusal = {
                        "executed": False,
                        "stop_reason": "reset_not_first",
                        "stop_detail": detail,
                        "error": detail,
                        "valid_actions": list(self._current_valid_actions),
                    }
                    self._last_action_call_result = dict(refusal)
                    return {
                        "action_result": refusal,
                        "state": _serialized_runtime_state(last_action_call_result=refusal),
                    }
            guard = self._noop_repeat_guard
            self._guard_hook_trace = []
            self._guard_death_override_batch = False
            # No pre-check here: the solver's per-position hook covers every
            # action of every call, including the first. Checking in both
            # places consumed the override slot twice - the pre-check armed it,
            # the retry cleared it, and the hook then blocked the very action
            # the model had been invited to repeat.
            step_arguments: dict[str, Any] = {"actions": normalized_actions}
            if stale_after:
                # forwarded as an argument so the solver can weigh it AFTER its
                # own per-position guards, which name a specific action and
                # carry an override
                step_arguments["stale_after"] = stale_after
            raw_payload = self._step_env_callback(step_arguments)
            if not isinstance(raw_payload, dict):
                raise RuntimeError("action(actions) did not return a JSON-like payload.")
            if reset_exposed() and raw_payload.get("executed"):
                # Use actual execution, not submitted batch size: a refused
                # call does not consume the snippet's first-action allowance.
                snippet_executed_actions += max(1, int(raw_payload.get("executed_count") or 1))
            if _animation_enabled():
                # Held on the agent, not routed through the sandbox: the chain
                # is dropped by compaction so the raw grids never become text,
                # which also meant it never reached the summary. It is only
                # ever used to render an image here, so it has no business
                # crossing the pipe twice.
                self._last_animation_chain = raw_payload.get("animation_chain") or []
                # Whether the LAST executed action moved anything, which is not
                # the same as whether the sequence did: a batch of UP, DOWN,
                # LEFT, SPACE reports gameplay_changed for the batch, so an
                # inert SPACE at the end was hidden behind three moves that
                # were not. The image exists for the action that animated, so
                # it is that action's flag that decides.
                per_action = raw_payload.get("gameplay_changed_per_action")
                if isinstance(per_action, list) and per_action:
                    self._last_action_gameplay_changed = bool(per_action[-1])
                else:
                    self._last_action_gameplay_changed = raw_payload.get(
                        "gameplay_changed"
                    )
            compact_payload = self._compact_action_result(raw_payload)
            if guard is not None and normalized_actions:
                self._record_guard_observations(guard, raw_payload, compact_payload)
            next_valid_actions = raw_payload.get("valid_actions")
            if isinstance(next_valid_actions, list):
                self._current_valid_actions = _normalize_valid_actions(next_valid_actions)
            if compact_payload.get("executed") and _terminal_action_reason(compact_payload):
                terminal_action_result = compact_payload
            self._last_action_call_result = dict(compact_payload)
            return {
                "action_result": compact_payload,
                "state": _serialized_runtime_state(
                    next_valid_actions=next_valid_actions if isinstance(next_valid_actions, list) else None,
                    last_action_call_result=compact_payload,
                ),
            }

        # per snippet: a repeat across snippets is the model choosing, not a
        # loop failing to converge
        self._repeat_state_seen.clear()
        sandbox_result = run_sandboxed_python(
            code=code,
            timeout_seconds=self._python_timeout,
            initial_state=_serialized_runtime_state(),
            action_handler=_handle_action,
            animation_handler=_handle_animation if _animation_enabled() else None,
            kept_functions=self._retained_sources(),
            retain_imports=_get_env_bool("ARC3_PERSISTENT_FUNCTIONS_IMPORTS", False),
            repair_hints=_get_env_bool("ARC3_PERSISTENT_FUNCTIONS_REPAIR_HINTS", False),
        )

        action_results = [
            item
            for item in sandbox_result.get("action_results") or []
            if isinstance(item, dict)
        ]
        payload: dict[str, Any] = {"tool": "python"}
        self._record_retained_functions(sandbox_result, payload)
        rendered_stdout = str(sandbox_result.get("stdout", "") or "")
        rendered_error = str(sandbox_result.get("error", "") or "")
        if rendered_error:
            payload["error"] = rendered_error
            if rendered_stdout:
                payload["stdout"] = rendered_stdout
        else:
            payload["returncode"] = 0
            if rendered_stdout:
                payload["stdout"] = rendered_stdout
            elif sandbox_result.get("result") is not None:
                payload["result"] = sandbox_result.get("result")
            elif action_results:
                if len(action_results) == 1:
                    payload["result"] = action_results[-1]
                else:
                    payload["result"] = {
                        "action_calls": len(action_results),
                        "last_action_call_result": action_results[-1],
                    }

        step_executed = any(bool(item.get("executed")) for item in action_results)
        made_progress = any(
            bool(item.get("gameplay_changed"))
            or bool(item.get("level_completed"))
            or bool(item.get("game_over"))
            or bool(item.get("run_complete"))
            for item in action_results
        )
        if step_executed:
            self._last_step_summary = self._summarize_step_sequence(action_results)
            self._update_summarized_knowledge_from_step_summary()
        elif self._last_step_summary and not self._last_step_summary.get("stale"):
            # carried over across an exchange that executed nothing: mark it so
            # the next prompt reminds instead of re-narrating it as fresh
            self._last_step_summary = {**self._last_step_summary, "stale": True}
        return _ToolDispatchResult(
            self._render_tool_payload(payload, truncate_fields=("stdout", "error", "result")),
            step_executed=step_executed,
            made_progress=made_progress,
        )

    def _dispatch_tool(self, state_path: Path, name: str, arguments: dict[str, Any]) -> _ToolDispatchResult:
        self._ensure_session(state_path)
        if name == "python":
            return self._run_python_tool(state_path, arguments)
        return _ToolDispatchResult(json.dumps({"error": f"Unknown tool: {name}"}, indent=2))

    def _calibrate_from_usage(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        usage: dict[str, Any] | None,
    ) -> str:
        """Update the characters-per-token divisor from a served request.

        The images are subtracted using the SAME formula that will re-add them,
        so the two cancel and the divisor measures text alone. Clamped because
        an under-count risks context overflow, which is the dangerous
        direction; a bad measurement should cost accuracy, not a rejected
        request."""
        if not _calibrate_text_tokens() or not isinstance(usage, dict):
            return ""
        try:
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
        except (TypeError, ValueError):
            return ""
        if prompt_tokens <= 0:
            return ""
        scrubbed, image_tokens = _split_messages_for_estimate(list(messages))
        payload: dict[str, Any] = {"messages": scrubbed}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = _request_tool_choice(tools)
        try:
            rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        except TypeError:
            return ""
        text_tokens = prompt_tokens - image_tokens
        if text_tokens <= 0 or not rendered:
            return ""
        measured = len(rendered) / text_tokens
        clamped = min(_text_token_chars_max(), max(_text_token_chars_min(), measured))
        previous = self._text_chars_per_token
        self._text_chars_per_token = clamped
        if previous is None or abs(clamped - previous) >= 0.05:
            note = "" if clamped == measured else f" (clamped from {measured:.2f})"
            # returned rather than written here: append_transcript is a closure
            # inside analyze(), not reachable from a method
            return (
                f"text_token_calibration: {clamped:.2f} chars/token{note} "
                f"[prompt_tokens {prompt_tokens}, images {image_tokens}, "
                f"text chars {len(rendered)}]"
            )
        return ""

    def _append_context_message(
        self, messages: list[dict[str, Any]], message: dict[str, Any]
    ) -> None:
        """Append and count new context once, before any history can be evicted.

        Use the context estimator for all roles, including reasoning and image
        tokens. Subtract the empty request envelope; static system/tool schema
        tokens and re-sent history are not new context. The summary exchange
        itself bypasses this helper and starts a fresh interval.
        """
        messages.append(message)
        if _summary_interval_tokens() <= 0:
            return
        added = max(
            0,
            self._estimate_request_input_tokens([message])
            - self._estimate_request_input_tokens([]),
        )
        self._summary_context_tokens_since_attempt = (
            getattr(self, "_summary_context_tokens_since_attempt", 0) + added
        )

    def _maybe_append_rolling_summary(
        self, append_transcript, tools: list[dict[str, Any]] | None,
        *, request_timeout_seconds: float | None = None,
    ) -> None:
        """Attempt a summary after another interval of newly added context.

        Called at turn start, which is the only clean boundary: nothing is in
        flight, and a yield resumption arrives here too so no separate handling
        is needed. Crossing the threshold is not urgent - the summary is not
        needed until eviction reaches it - so overshooting by a turn costs
        nothing and interrupting an investigation would.

        The request carries the current history unchanged, so it is a
        continuation rather than a prefix: both the KV prefix and the recurrent
        state stay warm. Both messages are tagged, and the trimmer's
        summary-aware drain navigates by that tag.
        """
        interval = _summary_interval_tokens()
        if interval <= 0:
            return
        history = self._history_messages
        if not history:
            return

        since = getattr(self, "_summary_context_tokens_since_attempt", 0)
        if since < interval:
            return
        # before the instruction: context first, then the ask, so the ask is
        # still the last thing read. Stored exactly as sent - a request message
        # that differs from the one in history would diverge from the cached
        # prefix at that point and cost a prefill on every later turn.
        context_lines = self._summary_turn_context_lines() if _summary_turn_context() else []
        request_text = (
            "\n".join([*context_lines, "", SUMMARY_REQUEST_PROMPT])
            if context_lines
            else SUMMARY_REQUEST_PROMPT
        )
        request = _mark_control_message(
            {"role": "user", "content": request_text},
            _SUMMARY_CONTROL_KIND,
        )
        messages = [
            {"role": "system", "content": self._system_prompt},
            *history,
            request,
        ]
        # Every attempt starts a new interval, including HTTP errors, empty
        # replies and rejected tool calls. Reset before dispatch so all exit
        # paths share that behavior; the summary exchange is excluded from this clock.
        self._summary_context_tokens_since_attempt = 0
        try:
            result = self._chat_completion(
                messages,
                tools=tools,
                max_tokens=_summary_max_gen_tokens(self._max_output_tokens),
                enable_thinking=_summary_enable_thinking(),
                request_timeout_seconds=request_timeout_seconds,
            )
        except BaseException as exc:  # a failed summary must not end the turn
            log.warning("rolling summary request failed: %s", exc)
            return
        # before any rejection path: the tokens were generated whatever the
        # reply turned out to be, and a summary that is thrown away still cost
        # the time
        self._accumulate_usage_tokens(
            getattr(result, "usage", None), count_toward_turn=False
        )
        dropped_call = ""
        message = result.message or {}
        text = _normalize_message_content(message.get("content", "")).strip()
        if message.get("tool_calls"):
            # A reply that calls a tool may still have written a real summary
            # first - the instruction not to act is advisory, and the model
            # sometimes summarises and then reaches for the tool anyway. Keeping
            # a usable one is better than asking again for something it has
            # already produced, so the text is judged on its length: real
            # summaries here run 4,000 to 6,000 characters, while a preamble
            # introducing a tool call runs to tens.
            #
            # The tool call itself is dropped either way. It was never going to
            # execute: this request is outside the turn loop, with no sandbox
            # and no action budget attached.
            names = ", ".join(
                str((c.get("function") or {}).get("name", "?"))
                for c in message.get("tool_calls") or []
                if isinstance(c, dict)
            )
            if len(text) < _SUMMARY_MIN_USABLE_CHARS:
                log.warning(
                    "rolling summary tried to call %s with only %d chars of text; "
                    "skipping",
                    names or "a tool", len(text),
                )
                if text:
                    append_transcript(
                        "ROLLING SUMMARY REJECTED",
                        f"[tried to call {names}, text too short to use]\n{text}",
                    )
                return
            log.warning(
                "rolling summary tried to call %s but wrote %d chars first; "
                "keeping the text, dropping the call",
                names or "a tool", len(text),
            )
            dropped_call = names or "a tool"
        if not text:
            # A reply that deliberated to the output limit without reaching the
            # content wants a different fix from one that simply said nothing -
            # a larger max_output or a shorter interval - so the two are logged
            # apart rather than both as "empty".
            reasoning_chars = len(_extract_reasoning_text(message))
            if reasoning_chars:
                log.warning(
                    "rolling summary produced %d chars of reasoning and no summary "
                    "(finish_reason=%s); skipping",
                    reasoning_chars,
                    str(getattr(result, "finish_reason", "") or "?"),
                )
            else:
                log.warning("rolling summary came back empty; skipping")
            return
        truncated = str(getattr(result, "finish_reason", "")).strip() == "length"
        if truncated:
            # Kept rather than rejected: a summary cut at the output limit still
            # covers the early material, and the alternative is no summary at
            # all, which means the same span is evicted with nothing standing in
            # for it. But the tail is where "what I was in the middle of" lives,
            # so the model is told the text is incomplete rather than being left
            # to infer it from a sentence that stops.
            log.warning(
                "rolling summary hit the output limit at %d chars of summary, after "
                "%d chars of reasoning; keeping it, marked as incomplete",
                len(text),
                len(_extract_reasoning_text(message)),
            )
            text = f"{text}\n[This summary was cut off at the output limit and is incomplete.]"
        reply = _mark_control_message(
            {"role": "assistant", "content": text}, _SUMMARY_CONTROL_KIND
        )
        if _summary_replaces_history():
            # only now, with a usable summary in hand - a failed or rejected one
            # returned above and left history untouched
            replaced_count = len(self._history_messages)
            self._history_messages = [request, reply]
            # this discards history without going through the trimmer, so the
            # prefix is invalid and the flag has to be set by hand - otherwise
            # the next request keeps a priority that assumes a warm cache
            self._note_history_evicted()
        else:
            self._history_messages.append(request)
            self._history_messages.append(reply)
            replaced_count = 0
        reasoning_chars = len(_extract_reasoning_text(message))
        append_transcript(
            "ANALYZER STATUS",
            f"rolling_summary: {len(text)} chars of summary after "
            f"{reasoning_chars} chars of reasoning, {since} new context tokens since the previous attempt"
            + (" [TRUNCATED]" if truncated else "")
            + (f" [tool call dropped: {dropped_call}]" if dropped_call else "")
            + (f" [replaced {replaced_count} messages of history]"
               if replaced_count else ""),
        )
        # the text itself, so a run can be read back and judged rather than only
        # counted - this is the artefact the whole mechanism produces. Salvaged
        # ones get their own section name so they can be grepped and read apart
        # from the clean ones, which is the comparison that says whether a reply
        # that also reached for a tool wrote a worse summary.
        append_transcript(
            "ROLLING SUMMARY SALVAGED" if dropped_call else "ROLLING SUMMARY",
            text,
        )

    def _estimate_request_input_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> int:
        messages = _apply_summary_visibility(
            messages, evicted=getattr(self, "_has_evicted", False)
        )
        image_tokens = 0
        if _image_token_estimate_enabled():
            messages, image_tokens = _split_messages_for_estimate(messages)
        payload: dict[str, Any] = {"messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = _request_tool_choice(tools)
        return _estimate_tokens(payload, self._text_chars_per_token) + image_tokens

    def _drop_oldest_history_block(self, history: list[dict[str, Any]], *, preserve_recent: int) -> bool:
        removable = len(history) - preserve_recent
        if removable <= 0:
            return False
        # Never drop the last user message. Dropping one and draining to the
        # next is fine while another exists; when none does, the drain runs to
        # preserve_recent and _drop_until_first_user_message then empties the
        # history outright - the request becomes the system prompt alone and the
        # server answers "No user query found in messages".
        #
        # Reachable mid-turn, where the list ends with a tool result rather than
        # the opener, so preserve_recent protects the tool message and the
        # opener is the first thing popped. Seen in a live request log as a
        # 1-message request at request_index_within_turn=2. Pre-existing and
        # independent of the summary drain, which only makes it easier to reach
        # by evicting further in one step.
        if not any(str(m.get("role", "")).strip() == "user" for m in history[1:]):
            return False
        first = history.pop(0)
        first_role = str(first.get("role", "")).strip()
        if first_role in {"assistant", "tool"}:
            while history and history[0].get("role") == "tool" and len(history) > preserve_recent:
                history.pop(0)
            return True
        while history and history[0].get("role") == "tool" and len(history) > preserve_recent:
            history.pop(0)
        # Only navigate by summaries when there is one left to land on, and only
        # one that is a REQUEST (a user message) and sits within the removable
        # range. Checking merely that a summary exists somewhere let the loop
        # skip every remaining user message when the last one was already at the
        # head, stripping history down to preserve_recent with no user message
        # in it at all - which the chat template rejects outright:
        # "No user query found in messages", HTTP 400, turn lost.
        while history and history[0].get("role") != "user" and len(history) > preserve_recent:
            history.pop(0)
        return True

    def _keep_recent_history_turns(
        self,
        messages: list[dict[str, Any]],
        *,
        max_turns: int,
        drain_turns: int = 0,
        force_drain: bool = False,
    ) -> list[dict[str, Any]]:
        if not messages:
            return []
        if max_turns <= 0:
            # explicit opt-out: leave the token budget as the only limit
            return list(messages)

        drain_turns = min(max(0, drain_turns), max(0, max_turns - 1))
        if drain_turns:
            present = sum(
                1
                for message in messages
                if str(message.get("role", "")).strip() == "assistant"
            )
            # trigger at the cap unless this commit has already invalidated the
            # prefix for token reasons, in which case take the deep cut now
            if present > max_turns or force_drain:
                max_turns = max_turns - drain_turns
            elif not force_drain:
                return list(messages)

        kept_reversed: list[dict[str, Any]] = []
        assistant_turns = 0
        for message in reversed(messages):
            kept_reversed.append(message)
            if str(message.get("role", "")).strip() == "assistant":
                assistant_turns += 1
                if assistant_turns >= max_turns:
                    break

        kept = list(reversed(kept_reversed))
        while kept and str(kept[0].get("role", "")).strip() == "tool":
            kept.pop(0)
        return kept

    def _drop_until_first_user_message(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        trimmed = list(history)
        while trimmed and str(trimmed[0].get("role", "")).strip() != "user":
            trimmed.pop(0)
        return trimmed

    def _note_history_evicted(self) -> None:
        # The scheduler consumes the first flag at the next handover. The
        # second stays set so summaries remain visible after history is lost.
        self._context_was_trimmed = True
        self._has_evicted = True

    def _persistent_history_messages(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        original_history = messages[1:]
        if _any_control_pruning_enabled():
            messages = _prune_control_messages(messages)
        keep_images = _history_image_keep()
        if keep_images is not None:
            # after pruning, so a dropped resumption opener cannot consume a
            # retention slot
            messages = _apply_history_image_window(messages, keep_images)
        before_trim = len(messages)
        trimmed = self._trim_messages_for_context(messages, tools=tools)
        if not trimmed:
            if original_history:
                self._note_history_evicted()
            return []
        token_trim_dropped = len(trimmed) < before_trim
        trimmed_history = trimmed[1:]
        history = self._keep_recent_history_turns(
            trimmed_history,
            max_turns=_persistent_history_assistant_turns(),
            drain_turns=_history_turn_drain(),
            force_drain=token_trim_dropped and _history_drain_coalesce(),
        )
        if (
            history
            and str(history[0].get("role", "")).strip() != "user"
            and len(trimmed_history) > len(history)
        ):
            previous_message = trimmed_history[len(trimmed_history) - len(history) - 1]
            if str(previous_message.get("role", "")).strip() == "user":
                history = [previous_message, *history]
        history = self._drop_until_first_user_message(history)
        # Include turn-cap eviction, control pruning and image removal, which
        # can change the cached prefix without the token trimmer dropping any
        # messages. Image removal can leave the message count unchanged.
        if history != original_history:
            self._note_history_evicted()
        return history

    def _priority_total_levels(self) -> int:
        """Read public game metadata through the existing session callback.

        No new solver state or prompt fields. A custom callback, old object,
        or unavailable metadata can fall back without interrupting a game.
        """
        try:
            callback = getattr(self, "_step_env_callback", None)
            session = getattr(callback, "__self__", None)
            game = getattr(session, "game", None)
            count = parse_total_levels(getattr(game, "number_of_levels", None))
        except Exception:
            count = None
        if count is not None:
            return count
        if not getattr(self, "_priority_level_count_warned", False):
            self._priority_level_count_warned = True
            log.warning("priority: total level count unavailable or invalid; using 10")
        return 10

    def _maybe_handover(self) -> None:
        """Re-price this game, and yield the slot if a gate is running.

        Called right after the trimmer and before the request goes out, which is
        where the prefix has just been invalidated. Consumes the trim flag: it
        describes the trim that produced THIS request, and leaving it set would
        keep the game demoted long after the prefix had been rebuilt.
        """
        if not self._context_was_trimmed:
            return
        self._context_was_trimmed = False
        summary = self._last_step_summary or {}
        level = int(summary.get("level") or 1)
        actions = max(
            0, _priority_action_count(summary) - self._actions_at_level_start
        )
        # getattr: an agent unpickled from before this field existed would
        # otherwise raise here on every turn
        tokens = max(
            0,
            self._session_generated_tokens
            - getattr(self, "_tokens_at_level_start", 0),
        )
        gate = _priority_gate()
        snapshot = None
        options = _priority_level_options()
        total_levels = (
            self._priority_total_levels()
            if options["normalize_score"] or options["tail_lookup"] else None
        )
        if gate is not None and _priority_refresh_enabled():
            pace = self._progress_pace.cost_multiplier() if _get_env_bool("ARC3_PRIORITY_PACE", False) else 1.0
            snapshot = PrioritySnapshot(level, actions, tokens, pace, total_levels)
            estimate = gate._snapshot_priority(snapshot, time.monotonic())
        else:
            estimate = _game_priority(
                level, actions, tokens, endgame=self._in_endgame(),
                total_levels=total_levels,
            )
        # this request pays the prefill, so it drops to the bottom band; the one
        # after it, with the prefix rebuilt, returns to the warm band
        self._priority_current = estimate
        self._priority_next = estimate + _PRIORITY_BAND
        if gate is not None:
            _diag(self, "gate")
            gate.handover(estimate, snapshot)

    def _in_endgame(self) -> bool:
        """Whether this game has been running long enough to switch phase.

        Read from the session's run_elapsed_seconds rather than a clock of its
        own: that is the timer max_runtime_s_per_game uses to end the game, so
        the two cannot disagree about how old a game is.
        """
        minutes = _endgame_start_minutes()
        if minutes <= 0:
            return False
        elapsed = getattr(self, "_game_elapsed_seconds", None)
        if elapsed is None:
            return False          # no step has reported the clock yet
        if elapsed < minutes * 60.0:
            return False
        if not getattr(self, "_endgame_announced", False):
            self._endgame_announced = True
            log.warning(
                "endgame phase after %.0f minutes: ranking on what completing "
                "the current level would bank, with no tail and no hazard",
                elapsed / 60.0,
            )
        return True

    def _note_request_succeeded(self) -> None:
        """Promote to the warm band once a request has actually landed.

        Called on a completed response rather than on dispatch: a retry after a
        connection failure has rebuilt nothing, so the game stays in the bottom
        band until one lands.
        """
        nxt = getattr(self, "_priority_next", None)
        if nxt is not None:
            self._priority_current = nxt

    def _trim_messages_for_context(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        preserve_recent: int = 1,
        extra_safety_tokens: int = 0,
        force_drain: bool = False,
    ) -> list[dict[str, Any]]:
        if not messages:
            return []
        system_message = messages[0]
        history = list(messages[1:])
        preserve_recent = max(0, preserve_recent)
        budget_tokens = max(1, self._context_budget_tokens - max(0, extra_safety_tokens))
        # capped at half the budget: a mis-set knob should cost history depth,
        # not erase history
        drain_tokens = min(_context_drain_tokens(), budget_tokens // 2)
        target_tokens = max(1, budget_tokens - drain_tokens) if drain_tokens else budget_tokens
        floor_mode = bool(drain_tokens) and _context_drain_floor_mode()
        # Normally eviction starts only once the estimate exceeds the budget.
        # After the server has rejected a request the estimate is known to be
        # wrong, so waiting for it to admit the request is too big would drop
        # nothing at all; force_drain enters the loop already draining and lets
        # the existing target - budget minus one drain, computed against the
        # real budget - decide where to stop.
        draining = bool(force_drain and drain_tokens)
        def _head_is_summary() -> bool:
            """A summary REQUEST at the head, not its reply.

            Landing on the assistant reply leaves history starting with an
            assistant message, which _drop_until_first_user_message then strips -
            so the exit would drop the very thing it stopped for.
            """
            return bool(
                history
                and _message_is_summary(history[0])
                and str(history[0].get("role", "")).strip() == "user"
            )

        # `draining` means two different things. On the normal path it is the
        # hysteresis phase - we crossed the budget and are heading for the
        # target - and stopping early there is exactly the point. Under
        # force_drain it means the server rejected the request and the estimate
        # cannot be trusted, so a summary-shaped head says nothing about whether
        # the request now fits; stopping early there risks another rejection and
        # another round trip.
        def _stop_early(used_tokens: int) -> bool:
            return (
                _drain_stop_at_summaries()
                and draining
                and not force_drain
                and used_tokens <= budget_tokens
                and _head_is_summary()
            )

        while history:
            used = self._estimate_request_input_tokens([system_message, *history], tools=tools)
            over_budget = used > budget_tokens
            if not over_budget and not draining:
                break
            if floor_mode and not over_budget:
                # the mandatory phase is done; keep draining only while the NEXT
                # drop would still leave us at or above the target. Looking
                # ahead rather than dropping and checking is what puts a real
                # floor under the retained history.
                if _stop_early(used):
                    break
                probe = list(history)
                if not self._drop_oldest_history_block(probe, preserve_recent=preserve_recent):
                    break
                after = self._estimate_request_input_tokens(
                    [system_message, *probe], tools=tools
                )
                if after < target_tokens:
                    break
                history[:] = probe
                continue
            if not floor_mode and used <= (target_tokens if draining else budget_tokens):
                break
            if _stop_early(used):
                break
            draining = True
            if not self._drop_oldest_history_block(history, preserve_recent=preserve_recent):
                break
        history = self._drop_until_first_user_message(history)
        # A dropped message invalidates the server's prefix for this
        # conversation, so the next request has to be prefilled from the
        # divergence point regardless. That makes it the one moment when
        # yielding the slot to another game costs nothing.
        if len(history) != len(messages) - 1:
            self._note_history_evicted()
        return [system_message, *history]

    def _force_reduce_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        preserve_recent: int = 1,
    ) -> list[dict[str, Any]]:
        if not messages:
            return []
        system_message = messages[0]
        history = list(messages[1:])
        if not self._drop_oldest_history_block(history, preserve_recent=max(0, preserve_recent)):
            return list(messages)
        self._note_history_evicted()
        return [system_message, *history]

    def analyze(
        self,
        state_path: Path,
        action_num: int,
        valid_actions: list[str] | None = None,
        step_env: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        transcript_path: Path | None = None,
        analysis_step: int | None = None,
        transcript_updated: Callable[[str], None] | None = None,
        request_timeout_seconds: float | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> AnalyzerTurnResult | None:
        if not state_path.exists():
            return None
        self._ensure_session(state_path)
        self._step_env_callback = step_env
        if self._noop_repeat_guard is not None:
            solver_obj = getattr(step_env, "__self__", None)
            if solver_obj is not None:
                solver_obj.action_guard_hook = self._action_guard_hook
        self._current_valid_actions = _normalize_valid_actions(valid_actions)

        analyzer_log = transcript_path or (state_path.parent / f"{state_path.stem}_analyzer.txt")
        prompt_log = _resolve_prompt_log_path(state_path)
        current_frame, history_entries = load_runtime_state(state_path)
        user_prompt = self._build_user_prompt(
            action_num,
            valid_actions=valid_actions,
            current_frame=current_frame,
            history_entries=history_entries,
            previous_step_summary=self._last_step_summary,
        )
        display_action_num = _display_action_number(action_num)
        self._wm_absorbed_in_turn = False

        with open(analyzer_log, "a", encoding="utf-8") as f:
            step_label = f"analysis_step={analysis_step} | " if analysis_step is not None else ""
            transcript_header = (
                f"\n--- {step_label}action={display_action_num} | "
                f"{time.strftime('%H:%M:%S')} | tool-agent ---\n"
            )
            f.write(transcript_header)
        transcript_parts = [transcript_header]

        def append_transcript(label: str, content: str) -> None:
            _append_transcript_section(analyzer_log, label, content)
            transcript_parts.append(_render_transcript_section(label, content))
            if transcript_updated is not None:
                transcript_updated("".join(transcript_parts))

        append_transcript("SYSTEM PROMPT", self._system_prompt)

        previous_history_messages = list(self._history_messages)
        # Mirrors the history snapshot: a turn abandoned on a request failure is
        # rolled back, and the per-game token allowance has to roll back with it
        # or the limit stops measuring the game and starts measuring how flaky
        # the endpoint was. The tokens were really generated - the server spent
        # them - but the point of the limit is that the same game gets the same
        # budget whatever the infrastructure did.
        tokens_at_turn_start = self._session_generated_tokens
        # The tools go with it, even though the summary must not call one. The
        # template renders their definitions into a system block near the start
        # of the prompt, so a request without them diverges from the first
        # tokens and prefills the whole context: measured at 15% shared prefix
        # against 99% with them, and as prefix reuse falling from 75% to 45%
        # across a run. The instruction not to act, and the harness rejecting a
        # reply that calls a tool, are what keep it from acting - not withholding
        # the definitions.
        self._maybe_append_rolling_summary(
            append_transcript, self._tools(state_path),
            request_timeout_seconds=request_timeout_seconds,
        )
        preserve_history = True
        resuming_after_yield = bool(getattr(self, "_resume_after_yield", False))
        resume_reason = str(getattr(self, "_resume_reason", "") or "yield_bare")
        self._resume_after_yield = False
        self._resume_reason = ""
        if not resuming_after_yield and _persistent_functions_scope() == "turn":
            # A resumption is the same turn, so helpers survive it either way.
            if _persistent_functions() and getattr(self, "_kept_functions", None):
                self._retained_clear_notice = (
                    "Your retained functions were cleared at the start of this new analyzer turn; "
                    "redefine any you still need."
                )
            self._kept_functions = {}
        resume_mode = _yield_resume_prompt_mode() if resuming_after_yield else "full"
        # state_only always carries the board - that IS the mode - so the
        # image knob does not apply to it
        resume_frame = (
            current_frame
            if not resuming_after_yield
            or resume_mode == "state_only"
            or _resume_prompt_images_enabled()
            else None
        )
        if (
            resuming_after_yield
            and resume_mode in ("state_only", "short")
            and _resume_diff_image_enabled()
            and self._turn_diff_image is not None
        ):
            # _build_user_message consumes the pending slot, so re-arm it from
            # the turn cache for this one message
            self._pending_diff_image = self._turn_diff_image
            self._pending_diff_image_label = self._turn_diff_image_label
        if resume_mode == "state_only":
            opening_message = self._build_user_message(
                _yield_resume_prompt(resume_reason) + "\n\n" + _STATE_ONLY_CAPTION,
                current_frame,
            )
        elif resume_mode == "short":
            opening_message = self._build_user_message(
                _yield_resume_prompt(resume_reason), resume_frame
            )
        else:
            opening_message = self._build_user_message(
                user_prompt, resume_frame if resuming_after_yield else current_frame
            )
        # log what was actually SENT, not the argument that was passed in: on a
        # resumed invocation the short/state_only substitution happens here, so
        # transcribing `user_prompt` would report a full opener the model never
        # saw - and at a 60s yield budget that is most turns
        append_transcript("USER PROMPT", _message_display_text(opening_message))
        if resuming_after_yield:
            # Tag EVERY resumption message, not just the short form. A yield
            # leaves this turn's original opener in history, so the message
            # appended here is a duplicate restatement of unchanged state
            # (same step, level, board, diffs, ledger - a no-execution yield
            # changes nothing). It is needed for the request that resumes the
            # turn and is pure duplication afterwards, which is precisely what
            # ARC3_PRUNE_CONTROL_CONTEXT exists to drop.
            _mark_control_message(opening_message, "resume")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt}, *self._history_messages,
        ]
        self._append_context_message(messages, opening_message)
        messages = self._trim_messages_for_context(
            messages,
            tools=self._tools(state_path),
            preserve_recent=1,
        )
        step_executed = False
        turn_tool_call_count = 0
        captured_reasoning = ""
        latest_request_messages: list[dict[str, Any]] | None = None
        latest_request_tools: list[dict[str, Any]] | None = None
        latest_request_tool_choice: str | None = None
        latest_request_index = 0
        turn_started_at = time.monotonic()
        # per TURN, not per session: the budget is how much the model may
        # generate before the turn ends, and a resumption continues the same
        # turn, so this resets where the clock does
        self._turn_generated_tokens = 0
        yielded_control_reason: str | None = None

        def control_yield_reason() -> str | None:
            if should_stop is not None:
                try:
                    if should_stop():
                        return "stop_requested"
                except Exception as exc:
                    log.warning("analyzer stop check failed at action %d: %s", display_action_num, exc)
            if self._yield_seconds is not None and (time.monotonic() - turn_started_at) >= self._yield_seconds:
                return "turn_time_budget"
            if (
                self._yield_tokens is not None
                and self._turn_generated_tokens >= self._yield_tokens
            ):
                return "turn_token_budget"
            return None

        try:
            turn_count = 0
            while self._tool_steps is None or turn_count < self._tool_steps:
                yielded_control_reason = control_yield_reason()
                if yielded_control_reason is not None:
                    break
                turn_count += 1
                tools = self._tools(state_path)
                tool_choice = _request_tool_choice(tools)
                messages = self._trim_messages_for_context(messages, tools=tools)
                self._maybe_handover()
                latest_request_messages = json.loads(json.dumps(messages))
                latest_request_tools = json.loads(json.dumps(tools))
                latest_request_tool_choice = tool_choice
                latest_request_index = turn_count
                _write_prompt_log_snapshot(
                    prompt_log,
                    model_id=self._model.model_id,
                    base_url=self._model.base_url,
                    display_action_num=display_action_num,
                    analysis_step=analysis_step,
                    request_index=turn_count,
                    messages=latest_request_messages,
                    tools=latest_request_tools,
                    tool_choice=tool_choice,
                    transcript="".join(transcript_parts),
                )
                try:
                    request_kwargs: dict[str, Any] = {"tools": tools}
                    if request_timeout_seconds is not None:
                        request_kwargs["request_timeout_seconds"] = request_timeout_seconds
                    if self._save_request_logs:
                        _append_request_snapshot(
                            _resolve_request_log_path(state_path),
                            messages=latest_request_messages,
                            tools=latest_request_tools,
                            event="request",
                            tool_choice=latest_request_tool_choice,
                            analysis_step=analysis_step,
                            action=display_action_num,
                            request_index_within_turn=latest_request_index,
                            chat_template_kwargs=self._harness_template_kwargs(),
                        )
                    result = self._chat_completion(messages, **request_kwargs)
                    _calibration_note = self._calibrate_from_usage(
                        messages, request_kwargs.get("tools"), result.usage
                    )
                    if _calibration_note:
                        append_transcript("ANALYZER STATUS", _calibration_note)
                    self._accumulate_usage_tokens(result.usage)
                    self._note_request_succeeded()
                    _diag(self, "local")
                    if self._save_request_logs:
                        _append_request_snapshot(
                            _resolve_request_log_path(state_path),
                            messages=latest_request_messages,
                            tools=latest_request_tools,
                            event="response",
                            tool_choice=latest_request_tool_choice,
                            analysis_step=analysis_step,
                            action=display_action_num,
                            request_index_within_turn=latest_request_index,
                            finish_reason=result.finish_reason,
                            served_by=result.served_by,
                            usage=result.usage,
                            chat_template_kwargs=self._harness_template_kwargs(),
                        )
                except requests.RequestException as exc:
                    if not _is_context_length_error(exc):
                        raise
                    # A rejected request means the estimate was wrong, and 512
                    # tokens only helps when it was wrong by less than that -
                    # otherwise nothing is dropped and the fallback removes one
                    # block per rejected round trip. force_drain evicts down to
                    # the drain target instead, in one step, which is what the
                    # drain is for: land clear of the ceiling rather than just
                    # under it.
                    trimmed_messages = self._trim_messages_for_context(
                        messages,
                        tools=tools,
                        extra_safety_tokens=_CONTEXT_OVERFLOW_RETRY_TRIM_TOKENS,
                        force_drain=True,
                    )
                    if trimmed_messages == messages:
                        trimmed_messages = self._force_reduce_messages(messages)
                    if trimmed_messages == messages:
                        raise
                    append_transcript(
                        "ANALYZER STATUS",
                        "context_overflow_recovered: dropped older history after server rejected the request as too long.",
                    )
                    messages = trimmed_messages
                    continue
                truncated_tool_call = False
                if str(result.finish_reason or "").strip() == "length":
                    ladder = _reasoning_effort_ladder()
                    if ladder and self._reasoning_effort_rung < len(ladder) - 1:
                        self._reasoning_effort_rung += 1
                        log.warning(
                            "generation hit the output ceiling; reducing reasoning_effort to %r",
                            ladder[self._reasoning_effort_rung],
                        )
                        append_transcript(
                            "ANALYZER STATUS",
                            "reasoning_effort_reduced: generation was truncated at the output "
                            f"limit; subsequent requests use reasoning_effort="
                            f"{ladder[self._reasoning_effort_rung]!r} until an action executes.",
                        )
                raw_reasoning = _extract_reasoning_text(result.message)
                raw_content = _normalize_message_content(result.message.get("content", ""))
                tool_calls = json.loads(json.dumps(result.message.get("tool_calls") or []))
                tool_call_markup_in_text = _contains_tool_call_markup(raw_reasoning, raw_content)
                recovered_tool_calls_from_markup = False
                if not tool_calls and tool_call_markup_in_text:
                    tool_calls = _recover_tool_calls_from_markup(raw_reasoning, raw_content)
                    recovered_tool_calls_from_markup = bool(tool_calls)
                reasoning = _strip_tool_call_markup(raw_reasoning) if tool_call_markup_in_text else raw_reasoning
                content = _strip_tool_call_markup(raw_content) if tool_call_markup_in_text else raw_content
                malformed_argument_errors: list[str] = []
                for tool_call in tool_calls:
                    function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
                    tool_name = str(function.get("name", "")).strip() or "unknown"
                    raw_arguments = function.get("arguments", "{}")
                    if isinstance(raw_arguments, str):
                        try:
                            json.loads(raw_arguments)
                        except json.JSONDecodeError as exc:
                            malformed_argument_errors.append(f"{tool_name}: invalid JSON arguments ({exc})")
                if malformed_argument_errors:
                    # Drop ALL calls from this reply rather than the bad one.
                    #
                    # An unparseable arguments string means the reply was cut
                    # off mid-call (finish_reason=length is the usual cause),
                    # so anything after the truncation point is missing and an
                    # earlier "valid" call in the same message expresses only
                    # part of an intent - dispatching half a plan is worse than
                    # dispatching none.
                    #
                    # Critically, the message must also never reach history:
                    # servers re-parse tool calls in the incoming message list
                    # when rendering the chat template, so one malformed call
                    # committed to history makes EVERY later request fail at
                    # render time, before inference - an unbreakable loop that
                    # the HTTP retry cannot help with. Content and reasoning
                    # are kept; only the calls are dropped, which routes this
                    # into the existing no-tool-call path.
                    log.warning(
                        "dropping %d malformed tool call(s) (finish_reason=%s): %s",
                        len(tool_calls),
                        result.finish_reason or "?",
                        "; ".join(malformed_argument_errors),
                    )
                    tool_calls = []
                    truncated_tool_call = str(result.finish_reason or "").strip() == "length"
                response_meta = _format_model_response_meta(
                    finish_reason=result.finish_reason,
                    served_by=result.served_by,
                    reasoning=reasoning,
                    content=content,
                    tool_calls=tool_calls,
                    tool_call_markup_in_text=tool_call_markup_in_text,
                    recovered_tool_calls_from_markup=recovered_tool_calls_from_markup,
                    malformed_argument_errors=malformed_argument_errors,
                )
                append_transcript(
                    "MODEL RESPONSE META",
                    response_meta,
                )
                assistant_message: dict[str, Any] = {"role": "assistant"}

                if reasoning:
                    captured_reasoning = reasoning
                    append_transcript("THINKING", reasoning)
                    for _reasoning_key in _reasoning_history_keys():
                        assistant_message[_reasoning_key] = reasoning

                if not tool_calls:
                    if content:
                        self._update_summarized_knowledge_from_assistant(content)
                        append_transcript("ASSISTANT", content)
                        assistant_message["content"] = content
                    elif reasoning:
                        assistant_message["content"] = None

                    if content or reasoning:
                        if (
                            not content
                            and reasoning
                            and _is_degenerate_text(reasoning)
                        ):
                            # A reasoning-only reply that is one character
                            # repeated to the output cap is a serving fault, not
                            # thinking: observed as exactly 6144 "!" from the
                            # first token, with content_chars 0 and no tool
                            # call, under speculative decoding. Keeping it cost
                            # far more than the lost turn - 12k characters of
                            # noise pushed the trimmer from 31 history messages
                            # to 12, evicting real history, and the model then
                            # concluded "my replies keep failing to emit the
                            # tool call" and started writing shorter answers to
                            # fix a formatting error it had not made.
                            # transcript AND log: the transcript records it
                            # where it happened, but a serving fault should be
                            # visible while the run is going rather than only
                            # on a later grep
                            log.warning(
                                "degenerate reasoning dropped: %d chars, no content "
                                "and no tool call - the reply was a repetition loop. "
                                "This is a serving fault, not the model reasoning.",
                                len(reasoning),
                            )
                            append_transcript(
                                "ANALYZER STATUS",
                                "degenerate_reasoning_dropped: the reply was a "
                                f"repetition loop ({len(reasoning)} chars) with no "
                                "content or tool call, and was replaced in context.",
                            )
                            assistant_message = _mark_control_message(
                                {"role": "assistant", "content": _DEGENERATE_CONTENT_STUB},
                                "stub",
                            )
                        elif content and _is_degenerate_text(content):
                            # a repetition loop feeds on its own output, so the
                            # garbage must not reach the next request's context
                            append_transcript(
                                "ANALYZER STATUS",
                                "degenerate_content_stubbed: assistant text was highly "
                                "repetitive and was replaced in context.",
                            )
                            assistant_message = {
                                "role": "assistant",
                                "content": _DEGENERATE_CONTENT_STUB,
                            }
                        elif (
                            not content
                            and reasoning
                            and _stub_dead_reasoning_enabled()
                        ):
                            # reasoning-only reply: nothing to continue from,
                            # and re-sending it biases the retry toward the
                            # same dead end. Transcript keeps the full text.
                            assistant_message = _mark_control_message(
                                {"role": "assistant", "content": _DEAD_REASONING_STUB},
                                "stub",
                            )
                        self._append_context_message(messages, assistant_message)
                    yielded_control_reason = control_yield_reason()
                    if yielded_control_reason is not None:
                        break
                    followup_prefix = "You have not acted yet. Investigate first. "
                    if truncated_tool_call:
                        # the only signal the model gets that LENGTH, not
                        # formatting, is why its call vanished
                        followup_prefix = (
                            "Your previous reply was CUT OFF before the tool call finished, so the "
                            "call was incomplete and could not be executed. Keep your reasoning "
                            "much shorter this time and emit the `python` tool call early enough "
                            "that it completes within the output limit. Prefer one small snippet "
                            "over a long analysis. "
                        )
                    elif malformed_argument_errors:
                        followup_prefix = (
                            "Your previous tool call had arguments that were not valid JSON, so it "
                            "was not executed. Emit exactly one `python` tool call whose arguments "
                            "are a well-formed JSON object. "
                        )
                    if tool_call_markup_in_text:
                        followup_prefix = (
                            "You did not call a tool. We detected `<tool_call>` markup inside your reasoning or assistant text, "
                            "so no parsed tool call was executed. On this retry, do not add a note or explanation first. "
                            "Emit exactly one `python` tool call directly as your next response. "
                            "Do not place `<tool_call>` markup inside reasoning, explanation, or notes. "
                        )
                    # the clause names where transient facts SHOULD go, so it
                    # holds only while that section is carried; a mode that
                    # drops it would be pointing the model nowhere
                    _transient_clause = (
                        " \u2014 current positions and this turn's events go in "
                        "`Recent findings:`. "
                        if "Recent findings" in _advertised_section_labels()
                        # "only" wants something to contrast against, and the
                        # clause that supplied it names a section this mode has
                        # dropped; keep the contrast, drop the destination
                        else ", not current positions or what just happened. "
                    )
                    followup_prompt = (
                        f"{followup_prefix}"
                        "Then investigate and revise your working world model of what the level contains, what actions appear to do, what the current goal seems to be, and what plan looks best. "
                        # the same narrowing the opener applies: advertising a section a
                        # restricted mode drops invites the model to write
                        # where it will never read it back
                        f"If helpful, include memory update lines such as "
                        f"{_section_list_phrase(_advertised_section_labels(), 'or')}. "
                        "`World model:` holds durable environment rules"
                        + (" only" if "Recent findings" in _advertised_section_labels() else "")
                        + f"{_transient_clause}"
                        "Each section you write replaces the stored one entirely, "
                        "so restate what is still true, not only what is new. "
                        "Call the `python` tool with code that inspects `current_frame`, `previous_frame`, `last_transition`, `history`, or `valid_actions` -- use `current_frame.segmentation` as the primary view, and `.ascii` only for a small specific region -- "
                        "compare `previous_frame` to `current_frame` for the most recent change, "
                        "derives a compact board summary, programs a small search or scorer over candidate actions or short sequences, "
                        "then call `action(actions)` inside Python with the best valid action or ordered batch that your code selected. "
                        f"{TOOL_CALL_FORMAT_GUIDANCE}"
                    )
                    append_transcript("USER PROMPT", followup_prompt)
                    self._append_context_message(
                        messages,
                        _mark_control_message(
                            {"role": "user", "content": followup_prompt}, "nudge"
                        )
                    )
                    continue

                if content:
                    self._update_summarized_knowledge_from_assistant(content)
                    append_transcript("ASSISTANT", content)
                    assistant_message["content"] = content
                assistant_message["tool_calls"] = tool_calls
                turn_tool_call_count += len(tool_calls)
                self._append_context_message(messages, assistant_message)

                for tool_index, tool_call in enumerate(tool_calls):
                    function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
                    tool_name = str(function.get("name", "")).strip()
                    raw_args = function.get("arguments", "{}")
                    try:
                        if isinstance(raw_args, str):
                            arguments = json.loads(raw_args)
                        elif isinstance(raw_args, dict):
                            arguments = json.loads(json.dumps(raw_args))
                        else:
                            arguments = {}
                    except json.JSONDecodeError:
                        arguments = {}
                    rendered_tool_call = _render_tool_call_markup(tool_name, raw_args)
                    append_transcript(
                        f"TOOL CALL: {tool_name}",
                        rendered_tool_call or (json.dumps(arguments, indent=2) if arguments else "{}"),
                    )
                    dispatch = self._dispatch_tool(state_path, tool_name, arguments)
                    if dispatch.step_executed:
                        step_executed = True
                        if self._reasoning_effort_rung >= 0 and dispatch.made_progress:
                            # the turn achieved something, so restore the server
                            # default; an inert action leaves the reduction in
                            # place rather than inviting another truncation
                            log.info(
                                "action changed the board; restoring default reasoning_effort"
                            )
                            self._reasoning_effort_rung = -1
                    append_transcript(f"TOOL RESULT: {tool_name}", _render_tool_result_display(dispatch.content))
                    self._append_context_message(
                        messages,
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.get("id", ""),
                            "content": dispatch.content,
                        }
                    )
                    if dispatch.step_executed:
                        if tool_index < len(tool_calls) - 1:
                            preserve_history = False
                        break
                    yielded_control_reason = control_yield_reason()
                    if yielded_control_reason is not None:
                        if tool_index < len(tool_calls) - 1:
                            preserve_history = False
                        break
                if yielded_control_reason is not None:
                    break
                if step_executed:
                    break

        except requests.RequestException as exc:
            if _is_read_timeout(exc) and _yield_on_timeout_enabled():
                # Keep this turn's completed exchanges and hand control back to
                # the solver, which re-invokes on the same game state.
                append_transcript(
                    "ANALYZER STATUS",
                    f"request_timeout (yielding, history preserved): {exc}",
                )
                log.warning(
                    "analyzer request timed out at action %d; yielding with %d tool call(s) preserved: %s",
                    display_action_num,
                    turn_tool_call_count,
                    exc,
                )
                self._resume_after_yield = True
                self._resume_reason = "timeout"
                if latest_request_messages is not None:
                    _write_prompt_log_snapshot(
                        prompt_log,
                        model_id=self._model.model_id,
                        base_url=self._model.base_url,
                        display_action_num=display_action_num,
                        analysis_step=analysis_step,
                        request_index=latest_request_index,
                        messages=latest_request_messages,
                        tools=latest_request_tools,
                        tool_choice=latest_request_tool_choice,
                        transcript="".join(transcript_parts),
                    )
                # preserve_history stays True, so the finally block commits
                # this turn's exchanges exactly as a normal yield would.
                return AnalyzerTurnResult(
                    step_executed=False,
                    reasoning=captured_reasoning,
                    yielded_control=True,
                )
            append_transcript("ANALYZER STATUS", f"request_error: {exc}")
            preserve_history = False
            self._session_generated_tokens = tokens_at_turn_start
            if latest_request_messages is not None:
                _write_prompt_log_snapshot(
                    prompt_log,
                    model_id=self._model.model_id,
                    base_url=self._model.base_url,
                    display_action_num=display_action_num,
                    analysis_step=analysis_step,
                    request_index=latest_request_index,
                    messages=latest_request_messages,
                    tools=latest_request_tools,
                    tool_choice=latest_request_tool_choice,
                    transcript="".join(transcript_parts),
                )
            log.warning("analyzer request failed at action %d: %s", display_action_num, exc)
            return AnalyzerTurnResult(step_executed=False, retryable_failure=True, reasoning=captured_reasoning)
        except Exception as exc:
            append_transcript("ANALYZER STATUS", f"error: {exc}")
            preserve_history = False
            self._session_generated_tokens = tokens_at_turn_start
            if latest_request_messages is not None:
                _write_prompt_log_snapshot(
                    prompt_log,
                    model_id=self._model.model_id,
                    base_url=self._model.base_url,
                    display_action_num=display_action_num,
                    analysis_step=analysis_step,
                    request_index=latest_request_index,
                    messages=latest_request_messages,
                    tools=latest_request_tools,
                    tool_choice=latest_request_tool_choice,
                    transcript="".join(transcript_parts),
                )
            log.warning("analyzer failed at action %d: %s", display_action_num, exc)
            return None
        finally:
            if step_executed and not self._wm_absorbed_in_turn:
                self._turns_without_wm_update += 1
            if preserve_history:
                self._history_messages = self._persistent_history_messages(messages, tools=self._tools(state_path))
            else:
                self._history_messages = previous_history_messages
            solver_obj = getattr(self._step_env_callback, "__self__", None)
            if solver_obj is not None and hasattr(solver_obj, "action_guard_hook"):
                solver_obj.action_guard_hook = None
            self._step_env_callback = None
            self._current_valid_actions = []

        if step_executed:
            status_message = "Step executed."
        elif yielded_control_reason is not None:
            status_message = f"Yielded control to solver: {yielded_control_reason}."
        else:
            status_message = "No action(...) call was captured."

        status = (
            f"model: {self._model.model_id}\n"
            f"base_url: {self._model.base_url}\n"
            f"max_output_tokens: {self._max_output_tokens if self._max_output_tokens is not None else 'server default'}\n"
            f"reply_reserve_tokens: {self._reply_reserve_tokens}\n"
            f"context_budget_tokens: {self._context_budget_tokens}\n"
            f"request_safety_margin_tokens: {self._request_safety_margin_tokens}\n"
            f"tool_output_tokens: {self._tool_output_tokens}\n"
            f"yield_seconds: {self._yield_seconds if self._yield_seconds is not None else 'disabled'}\n"
            f"yield_tokens: {self._yield_tokens if self._yield_tokens is not None else 'disabled'}\n"
            f"available_tools: python\n"
            f"python_timeout_seconds: {self._python_timeout}\n"
            f"history_messages: {len(self._history_messages)}\n"
            f"step_executed: {step_executed}\n"
            f"message: {status_message}"
        )
        append_transcript("ANALYZER STATUS", status)
        if latest_request_messages is not None:
            _write_prompt_log_snapshot(
                prompt_log,
                model_id=self._model.model_id,
                base_url=self._model.base_url,
                display_action_num=display_action_num,
                analysis_step=analysis_step,
                request_index=latest_request_index,
                messages=latest_request_messages,
                tools=latest_request_tools,
                tool_choice=latest_request_tool_choice,
                transcript="".join(transcript_parts),
            )
        # An invocation that executed nothing will be re-invoked by the solver
        # on the same game state, so whatever it appends next is a duplicate
        # restatement. This covers BOTH non-execution exits: the time budget
        # (yielded_control_reason set) and the tool-step budget (loop condition
        # went false, no yield reason). Execution and yielding are mutually
        # exclusive, so `not step_executed` is the whole condition.
        self._resume_after_yield = not step_executed
        if yielded_control_reason is not None:
            self._resume_reason = (
                "yield_tools" if turn_tool_call_count > 0 else "yield_bare"
            )
        else:
            self._resume_reason = (
                "tool_steps" if turn_tool_call_count > 0 else "yield_bare"
            )
        if not step_executed and turn_tool_call_count == 0 and self._last_step_summary:
            # no dispatch ran, so _dispatch_tool never marked the summary
            # stale; without this the next opener re-narrates an older
            # execution as if it had just happened (or claims nothing has run
            # yet) directly beneath this turn's own reasoning.
            self._last_step_summary = {
                **self._last_step_summary,
                "stale": True,
                "stale_reason": "no_tool_call",
            }
        return AnalyzerTurnResult(
            step_executed=step_executed,
            reasoning=captured_reasoning,
            yielded_control=yielded_control_reason is not None,
        )
