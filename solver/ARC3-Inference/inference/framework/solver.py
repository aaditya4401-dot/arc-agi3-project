"""TAAF solver adapter for the existing tool-using harness."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import functools
import html
import logging
import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import arcengine
import taaf.game
from taaf.solver import Solver

from inference.utils.animation import (
    collapse_frames,
    normalize_frames,
    summarize_animation,
)
from inference.agent import auto_probe
from inference.agent.action_names import (
    reset_exposed,
    to_engine_action,
    to_model_action,
    to_model_actions,
)
from inference.agent.runtime_state import (
    Frame,
    HistoryEntry,
    RUNTIME_STATE_FILENAME,
    write_runtime_state,
)
from inference.agent.tool_agent import (
    _PRIORITY_UNTRIMMED_BASE,
    ToolAgent,
    _priority_gate,
)
from inference.framework.kaggle import (
    DEFAULT_QWEN_MODEL_DATASET_SOURCE,
    DEFAULT_SERVED_MODEL_NAME,
    DEFAULT_VLLM_MAX_MODEL_LEN,
    DEFAULT_VLLM_PORT,
    DEFAULT_VLLM_TENSOR_PARALLEL_SIZE,
    DEFAULT_VLLM_WHEELHOUSE_DATASET_SOURCE,
    DEFAULT_WHEELHOUSE_STAMP_TEXT,
    DuckKaggleVllmConfig,
    duck_kaggle_dataset_sources,
    duck_kaggle_setup_command,
    duck_kaggle_teardown_command,
)
from inference.utils.viewer_artifacts import (
    append_raw_events_sidecar,
    reset_raw_events_sidecar,
)

log = logging.getLogger(__name__)

AnalyzerFactory = Callable[[taaf.game.Game, int], Any]

def _env_number(name: str, default: float, cast=float) -> float:
    """A numeric env override that cannot take the run down at import time."""
    try:
        return cast(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return cast(default)


ANALYZER_RETRY_BACKOFF_SECONDS = max(
    0.0, _env_number("ARC3_ANALYZER_RETRY_BACKOFF_SECONDS", 10.0)
)
# A request that fails outright - a dead endpoint, a refused connection - is
# reported as retryable and the solver goes back to the top of the loop. With
# no limit that is an infinite spin: nothing monitors the local server, so if
# it exits mid-run every game retries once a second until its runtime budget is
# gone, and the run finishes reporting whatever each game had reached with no
# crash marker anywhere. Giving up after a run of consecutive failures makes
# that visible - the run ends in seconds rather than at the wall clock, and the
# games carry a note saying why. 0 restores the unlimited behaviour.
# Negative means keep retrying; 0 means give up on the first failure. No
# max(0, ...) clamp here - it would flatten a negative to zero and turn
# "never give up" into "give up at once".
# Kaggle requires an action within 15 minutes of the run starting. SGLang
# needs about 10.5 of those to load weights - 6 on the main model, 3 more on the
# MTP draft head, which is a second full pass over the same 206 shards - and the
# notebook blocks on /health before playing anything, so a slow load spends the
# whole allowance before a single action is issued and the submission scores
# zero. With this set, the first N games to start execute one RESET straight
# against the engine before touching the analyzer, which satisfies the clock
# while the server is still loading; the HTTP retry loop then covers the rest of
# the wait. 0 disables.
WARMUP_ACTION_GAMES = _env_number("ARC3_WARMUP_ACTION_GAMES", 0, int)


MAX_CONSECUTIVE_ANALYZER_FAILURES = _env_number("ARC3_MAX_ANALYZER_FAILURES", 10, int)
DEFAULT_CANCEL_DRAIN_TIMEOUT_SECONDS = 120.0

# No-op guard: width of the border frame treated as HUD (progress bars,
# timers). 0 disables the guard. Simple v1 heuristic; see step_env.
# arcengine renders a frame after every internal step(), so one action can come
# back as a short animation whose intermediate frames the harness has always
# discarded. Off by default: new information for the model, not a fix.
ANIMATION_ENABLED = os.environ.get("ARC3_ANIMATION", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
NOOP_GUARD_BORDER_WIDTH = max(0, int(os.environ.get("ARC3_NOOP_GUARD_BORDER", "4")))
# Print one line per action(...) call into the snippet's stdout. Costs a line of
# tool output per call, which is charged against tool_output_tokens.
# The patch-110 rewordings, behind one switch. They replaced "changed nothing"
# with "no change inside the board area" across every guard message, because the
# old phrasing was false in the case that matters - the HUD usually DID change.
# The reasoning still looks right, but they shipped alongside the outcome line
# in a run that lost 43% of the previous competition score, so the two need to
# be separable before either can be judged. Off restores the wording of the run
# that scored 4.05.
NEW_NOOP_WORDING = os.environ.get(
    "ARC3_NEW_CHANGED_PROMPTS", "0"
).strip().lower() in ("1", "true", "yes", "on")
ACTION_ECHO = os.environ.get("ARC3_ACTION_ECHO", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# When an action changes nothing, the sandbox arms a per-snippet latch and the
# NEXT action(...) call is refused. Off restores the older behaviour of letting
# the snippet keep acting on a state its own beliefs no longer describe.
STALE_STATE_BLOCK = os.environ.get(
    "ARC3_STALE_STATE_BLOCK", "0"
).strip().lower() in ("1", "true", "yes", "on")


# Stops a batch at the first action that changes nothing inside the board, so a
# wrong movement model costs one action rather than the whole sequence. It used
# to be implied by ARC3_NOOP_GUARD_BORDER being positive, which conflated the
# crop width with whether the guard runs at all - the border is also what
# gameplay_changed means, and that is wanted independently. Off by default:
# the guard converts actions into turns, which is a good trade only when tokens
# are cheap relative to the action budget.
# Level 1 pays the smallest share of the score - one part in N - and it is
# where the model knows least, so it probes hardest: repeating an action to see
# whether the repeat differs, retrying something that killed it to learn what
# killed it. The guards read that as waste. Set to 2 to let level 1 run
# unguarded and start guarding at level 2, where an action is worth more and
# the model has a working model of the game.
#
# The batch no-op stop is deliberately NOT covered: it costs one action of a
# planned sequence rather than refusing a deliberate probe, and a wrong
# movement model is exactly as expensive on level 1 as anywhere. Nor is the
# terminal latch, which stops a snippet acting after a level-up or a death -
# that is a board that no longer exists, not a judgement about the action.
GUARDS_FROM_LEVEL = max(1, int(os.environ.get("ARC3_GUARDS_FROM_LEVEL", "1") or 1))


def _guards_active(level: Any) -> bool:
    """Whether the per-action guards apply at this level."""
    try:
        return int(level) >= GUARDS_FROM_LEVEL
    except (TypeError, ValueError):
        return True  # unknown level: guard rather than not


BATCH_NOOP_BLOCK = os.environ.get(
    "ARC3_BATCH_NOOP_BLOCK", "0"
).strip().lower() in ("1", "true", "yes", "on")


_BLOCKED_VERDICTS = {
    "known_noop": "BLOCKED (known no-op in this exact board state; never executed)",
    "known_death": "BLOCKED (ended a previous attempt from this exact board state; never executed)",
}


def _action_verdict(item: dict) -> str:
    """The one-word account of what an executed action did.

    Shared with the per-action trace so the echo introduces no second
    vocabulary for the model to learn."""
    if item.get("game_over"):
        return "GAME OVER (attempt ended immediately after this action)"
    if item.get("run_complete") or item.get("done"):
        return "RUN COMPLETE"
    if item.get("level_completed"):
        return "LEVEL COMPLETED"
    changed = item.get("gameplay_changed")
    if changed is False:
        return (
            "NO-OP (executed, no change inside the board area)"
            if NEW_NOOP_WORDING
            else "NO-OP (executed, changed nothing)"
        )
    if changed is True:
        return "effect"
    return "executed"


def _build_action_echo(
    requested_displays: list[str],
    executed_payloads: list[dict],
    stop_reason: str | None,
) -> str:
    """One line per action(...) call, for the snippet's own stdout.

    A single action produces no per-action trace at all - the trace is built
    only for batches, game overs and refusals - so the snippet's only account
    of it arrives a turn later in the opener. Inside a loop that is too late to
    branch on. This fills the gap at the moment of the call, in the trace's own
    words, without repeating the trace for batches that already get one.
    """
    if not requested_displays:
        return ""
    if len(requested_displays) == 1:
        if executed_payloads:
            return f"[action] {requested_displays[0]} -> {_action_verdict(executed_payloads[0])}"
        reason = f" | stop_reason: {stop_reason}" if stop_reason else ""
        return f"[action] {requested_displays[0]} -> not executed{reason}"
    ran = ", ".join(
        str(item.get("action_display") or item.get("action_name") or "?")
        for item in executed_payloads
    )
    line = (
        f"[action] batch of {len(requested_displays)} actions, "
        f"{len(executed_payloads)} executed"
    )
    if ran:
        line += f": {ran}"
    if stop_reason:
        line += f" | stop_reason: {stop_reason}"
    return line


def _build_action_trace(
    requested_displays: list[str],
    executed_payloads: list[dict],
    blocked_kind: str | None = None,
) -> list[str]:
    """One self-describing line per submitted action: index, name, verdict.

    A guard refusal is called out at its own position rather than lumped in
    with the generic SKIPPED verdict: the prompt tells the model to treat this
    trace as the authoritative account of what happened, so a refusal that
    reads as "skipped" would send it looking for an upstream cause that does
    not exist. Positions AFTER the refusal are genuinely skipped and keep the
    plain verdict.
    """
    trace: list[str] = []
    blocked_position = len(executed_payloads) + 1 if blocked_kind else None
    for position, name in enumerate(requested_displays, 1):
        if blocked_position is not None and position == blocked_position:
            trace.append(
                f"{position}: {name} -> "
                + _BLOCKED_VERDICTS.get(blocked_kind or "", "BLOCKED (never executed)")
            )
            continue
        if position <= len(executed_payloads):
            verdict = _action_verdict(executed_payloads[position - 1])
        else:
            verdict = "SKIPPED (never executed)"
        trace.append(f"{position}: {name} -> {verdict}")
    return trace


def _interior_gameplay_changed(
    previous_grid: list[list[int]] | None,
    new_grid: list[list[int]] | None,
    *,
    border: int = NOOP_GUARD_BORDER_WIDTH,
) -> bool:
    """True when any cell outside the border frame differs between grids.

    Falls back to a plain inequality check (treat as changed) when the guard
    is disabled, grids are missing, shapes differ, or the border swallows the
    whole grid — the guard must fail open.
    """
    if border <= 0 or previous_grid is None or new_grid is None:
        return previous_grid != new_grid
    rows = len(previous_grid)
    cols = len(previous_grid[0]) if rows else 0
    if rows != len(new_grid) or any(len(r) != cols for r in (previous_grid[0:1] + new_grid[0:1])):
        return True
    if rows <= 2 * border or cols <= 2 * border:
        return previous_grid != new_grid
    for r in range(border, rows - border):
        prev_row = previous_grid[r]
        new_row = new_grid[r]
        if prev_row[border:cols - border] != new_row[border:cols - border]:
            return True
    return False

_LOCAL_SERVER_PROCESS_ENV_KEYS = (
    "LOCAL_ANALYZER_API_KEY",
    "OPENAI_API_KEY",
    "LOCAL_ANALYZER_BASE_URL",
    "OPENAI_BASE_URL",
    "LOCAL_ANALYZER_PROVIDER",
    "OPENAI_PROVIDER",
)


@dataclass
class _LocalServerRuntime:
    index: int
    repo_dir: Path
    api_key_file: Path
    env_overrides: dict[str, str]
    base_url: str
    api_key: str = ""


def _analyzer_reported_tokens(analyzer: Any) -> int:
    value = (
        getattr(analyzer, "generated_tokens", None)
        if hasattr(analyzer, "generated_tokens")
        else getattr(analyzer, "total_tokens", 0)
    )
    return max(0, int(value or 0))


def artifact_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _animation_chain(
    previous_grid: tuple[tuple[int, ...], ...],
    state: taaf.game.GameState | None,
) -> list[tuple[tuple[int, ...], ...]]:
    """The pre-action board, then each distinct state the action passed through.

    `state.frame` is only `raw.frame[-1]`; the rest of that list is the
    animation. Runs of identical grids collapse here rather than at render
    time, because a single action can return dozens of frames and they stay in
    memory until the next action replaces them.
    """
    if state is None:
        return []
    return collapse_frames(
        previous_grid, normalize_frames(getattr(state.raw, "frame", None))
    )


def _grid_from_state(state: taaf.game.GameState | None) -> tuple[tuple[int, ...], ...]:
    if state is None:
        return ()
    data = state.frame.data
    rows = data.tolist() if hasattr(data, "tolist") else data
    return tuple(tuple(int(cell) for cell in row) for row in rows)


def _level_number(game: taaf.game.Game) -> int:
    state = game.current_state
    completed = int(state.levels_completed)
    if state.won:
        return max(1, int(game.number_of_levels))
    return max(1, min(int(game.number_of_levels), completed + 1))


def _engine_action_names(game: taaf.game.Game) -> list[str]:
    names: list[str] = []
    for action_id in game.current_state.available_actions:
        try:
            name = arcengine.GameAction.from_id(int(action_id)).name
        except Exception:
            continue
        if name == "RESET" and not reset_exposed():
            continue
        if name not in names:
            names.append(name)
    return names


def _model_mouse_action_data(
    action_data: dict[str, Any] | None = None,
) -> dict[str, int]:
    data = action_data or {}
    return {"row": int(data.get("y", 0)), "col": int(data.get("x", 0))}


def _guard_action_data(action: Any) -> dict[str, Any]:
    """Action data in the row/col form the guard signature expects.

    ACTION6 carries x/y from the engine; `_format_action_display` already
    translates it the same way, so the ledger and the guard agree on what
    identifies a click."""
    data = dict(getattr(action, "data", None) or {})
    if getattr(getattr(action, "id", None), "name", "") == "ACTION6":
        return _model_mouse_action_data(data)
    return data


def _stale_repeat(stale_after: str, action: Any) -> bool:
    """Is this the same action that just changed nothing?

    Compared on the DISPLAY form, which carries mouse coordinates - a click
    somewhere else after a no-op click is a different action, and the repeat
    guard's override is meaningful for it.
    """
    return str(stale_after).strip() == _format_action_display(
        action.id.name, dict(action.data)
    ).strip()


def _stale_detail(action: Any, stale_after: str) -> str:
    return (
        f"{_format_action_display(action.id.name, dict(action.data))} was not "
        f"executed. The action immediately before it, {stale_after}, was a "
        + ("no-op - it produced no change inside the board area - so positions "
           if NEW_NOOP_WORDING else
           "no-op - it changed nothing outside the board border - so positions ")
        +
        "and paths computed since then may be wrong, which is why execution "
        "has been halted here. Re-read current_frame and last_action_call_result, "
        "work out why nothing moved, then act on what you find. "
        "(StaleStateActionError can be caught if a snippet has good reason to "
        "keep acting through no-ops.)"
    )


def _format_action_display(
    action_name: str, action_data: dict[str, Any] | None = None
) -> str:
    if action_name == "ACTION6":
        data = _model_mouse_action_data(action_data)
        return f"MOUSE(row={data['row']}, col={data['col']})"
    return to_model_action(action_name)


def _is_engine_game_over(game: taaf.game.Game) -> bool:
    return game.current_state.raw.state == arcengine.GameState.GAME_OVER


def _is_run_complete(game: taaf.game.Game) -> bool:
    return game.current_state.raw.state == arcengine.GameState.WIN


def _write_transcript_html(transcript_path: Path, html_path: Path, title: str) -> None:
    if not transcript_path.exists():
        return
    html_path.parent.mkdir(parents=True, exist_ok=True)
    text = transcript_path.read_text(encoding="utf-8")
    body = (
        '<!doctype html>\n<html><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title>"
        "<style>"
        "body{background:#1e1e1e;color:#e0e0e0;font-family:-apple-system,system-ui,sans-serif;"
        "padding:20px;max-width:1100px;margin:0 auto;line-height:1.4;}"
        "h1{color:#fff;}pre{white-space:pre-wrap;background:#111;padding:16px;border-radius:6px;"
        "border:1px solid #333;overflow:auto;}"
        "</style></head><body>"
        f"<h1>{html.escape(title)}</h1><pre>{html.escape(text)}</pre>"
        "</body></html>\n"
    )
    html_path.write_text(body, encoding="utf-8")


@dataclass
class _HarnessGameSession:
    solver: "HarnessSolver"
    game: taaf.game.Game
    analyzer: Any
    game_index: int
    pass_index: int
    state_path: Path
    transcript_path: Path
    analysis_html_relpath: str
    stop_event: threading.Event
    viewer_data_path: Path
    started_at: float = field(default_factory=time.monotonic)
    history_entries: list[HistoryEntry] = field(default_factory=list)
    viewer_events: list[dict[str, Any]] = field(default_factory=list)
    analysis_step: int = 0
    last_engine_action: str | None = None
    token_baseline: int = 0
    # token_baseline is a cursor: it is dragged forward after every step so the
    # engine can be told what that step cost, and so cannot answer "how much
    # has this game spent". This one is captured once and never moved.
    game_token_baseline: int = 0
    # The last executed action's animation, or None. One record, not a
    # history: the model reads it via animation() in the turn that follows,
    # and older animations are almost never what it is asking about.
    animation_record: dict[str, Any] | None = None
    _viewer_events_flushed: int = field(default=0, init=False, repr=False)

    def current_frame(self) -> Frame:
        return Frame(
            grid=_grid_from_state(self.game.current_state),
            step=self.action_count,
            level=_level_number(self.game),
        )

    def write_runtime_state(self) -> None:
        write_runtime_state(
            self.state_path,
            current_frame=self.current_frame(),
            history=self.history_entries,
        )

    def seed_initial_history(self) -> None:
        if not self.history_entries:
            self.history_entries.append(
                HistoryEntry(action="", frame=self.current_frame())
            )

    @property
    def action_count(self) -> int:
        run = self.game.game_run
        return len(run.history) if run is not None else 0

    def runtime_limit_reached(self) -> bool:
        if self.solver.max_runtime_s_per_game is None:
            return False
        return (
            time.monotonic() - self.started_at
        ) >= self.solver.max_runtime_s_per_game

    def timing_payload(self) -> dict[str, float | None]:
        elapsed = max(0.0, time.monotonic() - self.started_at)
        if self.solver.max_runtime_s_per_game is None:
            remaining = None
        else:
            remaining = max(0.0, self.solver.max_runtime_s_per_game - elapsed)
        return {"run_elapsed_seconds": elapsed, "time_remaining_seconds": remaining}

    def request_timeout_seconds(self) -> float | None:
        candidates: list[float] = []
        configured = getattr(self.analyzer, "_timeout", None)
        try:
            if configured is not None:
                candidates.append(float(configured))
        except (TypeError, ValueError):
            pass
        if self.solver.max_runtime_s_per_game is not None:
            remaining = self.timing_payload()["time_remaining_seconds"]
            if remaining is not None:
                candidates.append(float(remaining))
        soft_remaining = self.solver.soft_time_remaining_seconds()
        if soft_remaining is not None:
            candidates.append(soft_remaining)
        if not candidates:
            return None
        return max(0.1, min(candidates))

    def token_limit_reached(self) -> bool:
        """Whether this game has generated its allowance.

        Read from the analyzer rather than summed per step: should_stop is
        passed into the analyzer and runs at every loop boundary, so reading a
        counter that only advances when an action executes would leave a game
        that deliberates without acting running past its allowance. The
        analyzer's accumulator advances on every response.

        Measured against a baseline captured at game start, which is 0 for the
        per-game analyzer the solver builds but need not be if an
        analyzer_factory hands back a shared one.
        """
        # getattr, not attribute access: the notebook unpickles a solver whose
        # __dict__ was captured before this field existed, and __setstate__ does
        # a plain dict update - it applies no dataclass defaults. A bare access
        # would raise AttributeError here on the first turn of every game, since
        # should_stop() runs this every time.
        limit = getattr(self.solver, "max_generated_tokens_per_game", None)
        if limit is None or limit <= 0:
            return False
        spent = _analyzer_reported_tokens(self.analyzer) - self.game_token_baseline
        return spent >= limit

    def should_stop(self) -> bool:
        run = self.game.game_run
        if run is None or run.state != "playing":
            return True
        if self.stop_event.is_set():
            return True
        if _is_run_complete(self.game):
            return True
        if self.runtime_limit_reached():
            return True
        if (
            self.solver.max_actions_per_game is not None
            and self.action_count >= self.solver.max_actions_per_game
        ):
            return True
        if self.token_limit_reached():
            return True
        return False

    def _queue_position(self) -> int:
        """Position in the dispatch order, used only to break ties in the
        untrimmed priority band so games do not all queue at one value."""
        return max(0, int(self.game_index)) * 100 + max(0, int(self.pass_index))

    def play(self) -> None:
        run = self.game.game_run
        assert run is not None, "TAAF starts games before invoking the solver."
        run.solver_analysis_html = self.analysis_html_relpath
        self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
        self.transcript_path.touch(exist_ok=True)
        self.token_baseline = _analyzer_reported_tokens(self.analyzer)
        self.game_token_baseline = self.token_baseline
        # A game that has not started yet sits in the top band, so every game
        # gets going in queue order before any that has already trimmed comes
        # back for a second turn.
        gate = _priority_gate()
        if gate is not None:
            deadline = (
                self.started_at + self.solver.max_runtime_s_per_game
                if self.solver.max_runtime_s_per_game is not None else None
            )
            soft_remaining = self.solver.soft_time_remaining_seconds()
            if soft_remaining is not None:
                soft_deadline = time.monotonic() + soft_remaining
                deadline = min(deadline, soft_deadline) if deadline is not None else soft_deadline
            gate.configure_clock(self.started_at, deadline)
            gate.acquire(_PRIORITY_UNTRIMMED_BASE - self._queue_position())
        try:
            self.seed_initial_history()
            self.write_runtime_state()
            self._append_initial_viewer_event()
            self.write_viewer_payload()
            self._play_inner()
        finally:
            if gate is not None:
                gate.release()

    def _play_inner(self) -> None:
        run = self.game.game_run
        if self.solver._claim_warmup_slot():
            # Straight to the engine - _execute_auto_reset takes the same path
            # and needs no analyzer, so this works with the endpoint still
            # loading. One action off this game's budget buys the whole
            # submission a deadline it would otherwise miss.
            # warning, not info: _configure_logging only runs under run.py, so
            # the notebook has no handler and falls back to logging's lastResort,
            # which emits WARNING and above. An info line would vanish exactly
            # where this message matters most.
            log.warning(
                "%s: warmup RESET issued before the analyzer is reachable",
                run.game_id,
            )
            self._execute_auto_reset()
        if auto_probe.ENABLED:
            self._run_auto_probe()
        try:
            retry_analysis_step: int | None = None
            consecutive_failures = 0
            while not self.should_stop():
                if (
                    _is_engine_game_over(self.game)
                    and self.last_engine_action != "RESET"
                ):
                    self._execute_auto_reset()
                    continue

                if retry_analysis_step is None:
                    self.analysis_step += 1
                    analysis_step = self.analysis_step
                else:
                    analysis_step = retry_analysis_step

                self.write_runtime_state()
                transcript_before = self._read_transcript_bytes()
                try:
                    result = self.analyzer.analyze(
                        self.state_path,
                        self.action_count,
                        valid_actions=_engine_action_names(self.game),
                        step_env=self.step_env,
                        transcript_path=self.transcript_path,
                        analysis_step=analysis_step,
                        request_timeout_seconds=self.request_timeout_seconds(),
                        should_stop=self.should_stop,
                    )
                finally:
                    transcript_delta = self._transcript_delta_since(transcript_before)
                    if transcript_delta.strip():
                        self._append_analysis_viewer_event(
                            analysis_step, transcript_delta
                        )
                        self.write_viewer_payload()
                if result is None:
                    raise RuntimeError("Analyzer did not return a result.")
                if result.retryable_failure:
                    consecutive_failures += 1
                    if (
                        MAX_CONSECUTIVE_ANALYZER_FAILURES >= 0
                        and consecutive_failures > MAX_CONSECUTIVE_ANALYZER_FAILURES
                    ):
                        log.error(
                            "%s: analyzer failed %d times in a row; giving up on this "
                            "game. The endpoint is probably down - nothing here can "
                            "restart it.",
                            run.game_id,
                            consecutive_failures,
                        )
                        run.solver_note = (
                            f"analyzer failed {consecutive_failures} times consecutively"
                        )
                        break
                    retry_analysis_step = analysis_step
                    if self.should_stop():
                        break
                    time.sleep(ANALYZER_RETRY_BACKOFF_SECONDS)
                    continue

                # a turn that came back at all clears the run: the limit is for
                # an endpoint that has stopped answering, not for a game that
                # hits an occasional blip
                consecutive_failures = 0
                retry_analysis_step = None
                if getattr(result, "yielded_control", False):
                    # Reuse the step id only for a pure thinking continuation.
                    # If this invocation already executed actions, the resumed
                    # invocation opens with a fresh batch-summary prompt - a new
                    # logical turn - and must get a fresh analysis_step, or the
                    # viewer merges two turns into one card (and sums their
                    # token usage).
                    if not result.step_executed:
                        retry_analysis_step = analysis_step
                    continue
                if not result.step_executed:
                    continue
        except Exception as exc:
            if run.final_score is None:
                run.solver_note = f"error: {type(exc).__name__}: {exc}"
                if run.state == "playing":
                    run.state = "crashed"
                self._finish_if_needed()
        finally:
            total_tokens = _analyzer_reported_tokens(self.analyzer)
            if run.solver_note is None:
                run.solver_note = f"tokens={total_tokens}"
            self._finish_if_needed()
            self.state_path.unlink(missing_ok=True)
            self._write_analysis_html()
            self.write_viewer_payload()

    def _finish_if_needed(self) -> None:
        run = self.game.game_run
        if run is not None and run.final_score is None:
            if self.stop_event.is_set() and run.state == "playing":
                run.state = "cancelled"
            self.game.finish_game()

    def _write_analysis_html(self) -> None:
        if self.solver.job_dir is None:
            return
        _write_transcript_html(
            self.transcript_path,
            self.solver.job_dir / self.analysis_html_relpath,
            f"{self.game.game_run.game_id if self.game.game_run else self.game_index} analysis",
        )

    def _read_transcript_bytes(self) -> bytes:
        try:
            return self.transcript_path.read_bytes()
        except OSError:
            return b""

    def _transcript_delta_since(self, previous_transcript: bytes) -> str:
        try:
            current_size = self.transcript_path.stat().st_size
            previous_size = len(previous_transcript)
            with self.transcript_path.open("rb") as file:
                if current_size >= previous_size:
                    current_prefix = file.read(previous_size)
                    if current_prefix == previous_transcript:
                        return file.read().decode("utf-8", errors="replace").strip()
                    file.seek(0)
                return file.read().decode("utf-8", errors="replace").strip()
        except OSError:
            return ""

    def _base_viewer_event(self, frame: Frame) -> dict[str, Any]:
        run = self.game.game_run
        raw_state = self.game.current_state.raw.state
        return {
            "board": [list(row) for row in frame.grid],
            "board_ascii": frame.ascii,
            "score": int(self.game.current_state.levels_completed),
            "state": raw_state.name,
            "level": frame.level,
            "run_status": run.state if run is not None else "playing",
        }

    def _append_initial_viewer_event(self) -> None:
        if self.viewer_events:
            return
        frame = self.current_frame()
        self.viewer_events.append(
            {
                **self._base_viewer_event(frame),
                "type": "initial",
                "title": "Initial State",
                "action_num": self.action_count,
                "analysis_step": None,
                "action_display": "RESET",
                "reward": 0.0,
            }
        )

    def _append_analysis_viewer_event(
        self, analysis_step: int, transcript: str
    ) -> None:
        frame = self.current_frame()
        self.viewer_events.append(
            {
                **self._base_viewer_event(frame),
                "type": "analysis",
                "title": f"Analysis Step {analysis_step}",
                "action_num": self.action_count,
                "analysis_step": analysis_step,
                "transcript": transcript,
            }
        )

    def _append_action_viewer_event(
        self, payload: dict[str, Any], frame: Frame
    ) -> None:
        self.viewer_events.append(
            {
                **self._base_viewer_event(frame),
                "type": "action",
                "title": f"Action {int(payload.get('action_num') or self.action_count)}",
                "action_num": int(payload.get("action_num") or self.action_count),
                "analysis_step": self.analysis_step,
                "action_name": payload.get("action_name"),
                "action_display": payload.get("action_display"),
                "reward": payload.get("reward"),
                "board_changed": payload.get("board_changed"),
                "done": payload.get("done"),
                "level_completed": payload.get("level_completed"),
                "game_over": payload.get("game_over"),
                "run_complete": payload.get("run_complete"),
                "batch_index": payload.get("batch_index"),
                "batch_size": payload.get("batch_size"),
            }
        )

    def write_viewer_payload(self) -> None:
        if self.solver.job_dir is None:
            return
        self.viewer_data_path.parent.mkdir(parents=True, exist_ok=True)
        run = self.game.game_run
        last_event_source = next(
            (
                event
                for event in reversed(self.viewer_events)
                if event.get("type") == "action"
            ),
            self.viewer_events[-1] if self.viewer_events else {},
        )
        last_event = dict(last_event_source)
        last_event.pop("board", None)
        last_event.pop("board_ascii", None)
        last_event.pop("transcript", None)
        payload = {
            "game_id": run.game_id if run is not None else str(self.game_index),
            "agent_name": self.solver.label,
            "status": run.state if run is not None else "playing",
            "pass_index": self.pass_index,
            "pass_label": str(self.pass_index),
            "eventCount": len(self.viewer_events),
            "lastEvent": last_event,
            "viewer_steps": [],
            "replay_url": self.analysis_html_relpath,
        }
        if run is not None:
            payload.update(
                {
                    "levels_completed": run.levels_completed,
                    "total_levels": run.number_of_levels,
                    "actions_per_level": list(run.actions_per_level),
                    "final_score": run.final_score,
                }
            )
        if self._viewer_events_flushed == 0:
            reset_raw_events_sidecar(self.viewer_data_path)
        append_raw_events_sidecar(
            self.viewer_data_path, self.viewer_events[self._viewer_events_flushed :]
        )
        self._viewer_events_flushed = len(self.viewer_events)
        tmp_path = self.viewer_data_path.with_suffix(
            f"{self.viewer_data_path.suffix}.tmp"
        )
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(self.viewer_data_path)

    def _normalize_actions(
        self, arguments: dict[str, Any]
    ) -> tuple[list[arcengine.ActionInput] | None, str | None]:
        has_single = bool(str(arguments.get("action", "")).strip())
        has_batch = arguments.get("actions") is not None
        if has_single and has_batch:
            return None, "Use either `action` or `actions`, not both."

        if has_batch:
            raw_actions = arguments.get("actions")
            if not isinstance(raw_actions, list):
                return None, "`actions` must be a JSON array of action objects."
            if not raw_actions:
                return None, "`actions` must contain at least one action."
        else:
            if not has_single:
                return None, "step_env requires `action` or `actions`."
            raw_actions = [
                {
                    "action": arguments.get("action"),
                    "row": arguments.get("row"),
                    "col": arguments.get("col"),
                }
            ]

        actions: list[arcengine.ActionInput] = []
        for index, raw_action in enumerate(raw_actions, start=1):
            if not isinstance(raw_action, dict):
                return None, f"Action {index} must be a JSON object."
            action_name = to_engine_action(raw_action.get("action"))
            if not action_name:
                return (
                    None,
                    f"Unknown action at index {index}: {raw_action.get('action')!r}",
                )
            action_id = arcengine.GameAction.from_name(action_name)
            data: dict[str, Any] = {}
            if action_id == arcengine.GameAction.ACTION6:
                try:
                    row = max(0, min(63, int(raw_action["row"])))
                    column = max(0, min(63, int(raw_action["col"])))
                    data = {
                        "x": column,
                        "y": row,
                    }
                except (KeyError, TypeError, ValueError):
                    return (
                        None,
                        f"MOUSE action at index {index} requires integer row and col arguments.",
                    )
            actions.append(arcengine.ActionInput(id=action_id, data=data))
        return actions, None

    def _error_payload(self, message: str) -> dict[str, Any]:
        return {
            "executed": False,
            "error": message,
            "valid_actions": to_model_actions(_engine_action_names(self.game)),
            **self.timing_payload(),
        }

    def _terminal_payload(
        self, requested_actions: list[arcengine.ActionInput]
    ) -> dict[str, Any]:
        raw_state = self.game.current_state.raw.state
        is_game_over = raw_state == arcengine.GameState.GAME_OVER
        is_win = raw_state == arcengine.GameState.WIN
        requested = [
            _format_action_display(action.id.name, dict(action.data))
            for action in requested_actions
        ]
        stop_reason = (
            "run_complete" if is_win else "game_over" if is_game_over else "stopped"
        )
        return {
            "executed": False,
            "error": "No action was executed because the current game state is terminal or stopping.",
            "action_num": self.action_count,
            "level": _level_number(self.game),
            "score": int(self.game.current_state.levels_completed),
            "state": raw_state.name,
            "valid_actions": [],
            "board_changed": False,
            "done": is_win,
            "level_completed": False,
            "game_over": is_game_over,
            "run_complete": is_win,
            "batched": len(requested_actions) > 1,
            "requested_count": len(requested_actions),
            "executed_count": 0,
            "requested_actions": requested,
            "executed_actions": [],
            "stopped_early": True,
            "stop_reason": stop_reason,
            **self.timing_payload(),
        }

    def step_env(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if str(arguments.get("query") or "").strip() == "animation":
            # Routed through step_env so no analyzer-protocol signature has to
            # change. Executes nothing and spends no action budget.
            return {
                "executed": False,
                "query": "animation",
                "record": self.animation_record if ANIMATION_ENABLED else None,
            }
        requested_actions, error = self._normalize_actions(arguments)
        if error is not None or requested_actions is None:
            return self._error_payload(error or "Could not parse action request.")
        if self.should_stop() or _is_engine_game_over(self.game):
            return self._terminal_payload(requested_actions)

        executed_payloads: list[dict[str, Any]] = []
        total_reward = 0.0
        stop_reason: str | None = None
        batch_size = len(requested_actions)
        requested_displays = [
            _format_action_display(action.id.name, dict(action.data))
            for action in requested_actions
        ]

        guard_hook = getattr(self, "action_guard_hook", None)
        blocked_detail: str | None = None
        blocked_kind: str | None = None

        # Checked AFTER the per-position guards below, so a known death or a
        # known no-op takes precedence: both name a specific action and offer an
        # override, which is more actionable than a general staleness warning.
        stale_after = (
            arguments.get("stale_after")
            if STALE_STATE_BLOCK and _guards_active(_level_number(self.game))
            else None
        )

        for batch_index, action in enumerate(requested_actions, start=1):
            if self.should_stop():
                stop_reason = "stopped"
                break
            if (
                stale_after
                and batch_index == 1
                and _stale_repeat(stale_after, action)
            ):
                # the model is re-issuing the action that just changed nothing;
                # the staleness message is the correct one here, so take it
                # before the repeat guard can claim the turn
                blocked_kind = "stale_state"
                blocked_detail = _stale_detail(action, stale_after)
                stop_reason = blocked_kind
                break
            if guard_hook is not None:
                # Per-position check: the hash is recomputed from the CURRENT
                # board, so position N of a batch is judged against the state
                # its predecessors actually produced - not the state the batch
                # started in. This is what lets a fatal move be caught no
                # matter which route reached it.
                # The engine stores mouse coordinates as x/y; every consumer
                # that identifies an action - the ledger, the action display,
                # the guard signature - works in row/col. Passing action.data
                # raw made every click share the signature "MOUSE" with no
                # coordinates, so one no-op click poisoned clicking ANYWHERE
                # from that board state. On a mouse-only game that is terminal:
                # observed as 175 replies and one executed action in a run.
                verdict = guard_hook(
                    _grid_from_state(self.game.current_state),
                    action.id.name,
                    _guard_action_data(action),
                    _level_number(self.game),
                    [
                        (later.id.name, _guard_action_data(later))
                        for later in requested_actions[batch_index - 1:]
                    ],
                )
                if verdict:
                    blocked_kind, blocked_detail = verdict
                    stop_reason = blocked_kind
                    break
            if stale_after and batch_index == 1:
                # Ordering against the repeat guards above:
                #
                # Different action - the repeat guard wins. It names the action
                # and its override is meaningful: the model may have reason to
                # believe circumstances changed since that earlier visit.
                #
                # SAME action as the one that just no-opped - this message wins,
                # and the check is placed before the guards for that case (see
                # the early branch above). Telling a model that just watched X
                # do nothing to "issue it again and it will execute" is advice
                # to repeat a failure; worse, the guard's own override is
                # already satisfied by this call, so without the swap the action
                # would simply run with no message at all.
                blocked_kind = "stale_state"
                blocked_detail = _stale_detail(action, stale_after)
                stop_reason = blocked_kind
                break
            # anything that reaches execution clears the flag: it can only
            # happen when the check above declined to fire, and leaving a stale
            # marker set behind an executed action would be a latent surprise
            stale_after = None
            if action.id.value not in self.game.current_state.available_actions:
                message = f"{_format_action_display(action.id.name, dict(action.data))} is not valid right now."
                if executed_payloads:
                    stop_reason = "invalid_action"
                    break
                return self._error_payload(message)

            try:
                payload = self._execute_action(
                    action,
                    batch_index=batch_index,
                    batch_size=batch_size,
                    flush_viewer_payload=False,
                )
            except Exception as exc:
                if executed_payloads:
                    stop_reason = "action_error"
                    break
                return self._error_payload(f"{type(exc).__name__}: {exc}")
            executed_payloads.append(payload)
            total_reward += float(payload.get("reward", 0.0) or 0.0)

            if payload.get("run_complete"):
                stop_reason = "run_complete"
                break
            if payload.get("game_over"):
                stop_reason = "game_over"
                break
            if payload.get("level_completed"):
                stop_reason = "level_completed"
                break
            if (
                BATCH_NOOP_BLOCK
                and NOOP_GUARD_BORDER_WIDTH > 0
                and batch_size >= 2
                and not payload.get("gameplay_changed", True)
            ):
                stop_reason = "no_op_action"
                break

        if not executed_payloads:
            if blocked_detail is not None:
                # Nothing ran at all: report the refusal itself rather than a
                # generic error, so the sandbox can raise the right exception.
                return {
                    **self._error_payload(blocked_detail),
                    "executed": False,
                    "stop_reason": blocked_kind,
                    "stop_detail": blocked_detail,
                    "requested_count": batch_size,
                    "executed_count": 0,
                    "stopped_early": True,
                    "requested_actions": requested_displays,
                    "skipped_actions": list(requested_displays),
                    # the refused action is position 1 here, since nothing ran.
                    # Omitting it left the next turn's opener saying "stopped
                    # before None" - the later assignment for the partial-batch
                    # case is never reached on this path.
                    "blocked_action": requested_displays[0] if requested_displays else None,
                }
            return self._error_payload("No action was executed.")

        final_payload = dict(executed_payloads[-1])
        final_payload["reward"] = total_reward
        final_payload["last_reward"] = executed_payloads[-1].get("reward", 0.0)
        final_payload["batched"] = batch_size > 1
        final_payload["requested_count"] = batch_size
        final_payload["executed_count"] = len(executed_payloads)
        final_payload["requested_actions"] = requested_displays
        final_payload["executed_actions"] = [
            str(item.get("action_display") or item.get("action_name") or "")
            for item in executed_payloads
        ]
        final_payload["board_changed"] = any(
            bool(item.get("board_changed")) for item in executed_payloads
        )
        final_payload["stopped_early"] = len(executed_payloads) < batch_size
        if stop_reason is not None:
            final_payload["stop_reason"] = stop_reason
        if final_payload["stopped_early"]:
            final_payload["skipped_actions"] = requested_displays[len(executed_payloads):]
        for item in executed_payloads:
            if item.get("game_over"):
                final_payload["fatal_action"] = str(item.get("action_display") or "")
                break
        if batch_size > 1 or final_payload.get("game_over") or blocked_kind:
            final_payload["action_trace"] = _build_action_trace(
                requested_displays, executed_payloads, blocked_kind
            )
            final_payload["gameplay_changed_per_action"] = [
                bool(item.get("gameplay_changed", True)) for item in executed_payloads
            ]
        if ACTION_ECHO:
            echo = _build_action_echo(requested_displays, executed_payloads, stop_reason)
            if echo:
                # consumed by the sandbox, which prints it and pops it: leaving
                # it in the returned dict shows the model the same line twice,
                # once on stdout and once in the payload it reads back
                final_payload["action_echo"] = echo
        if blocked_detail is not None:
            final_payload["stop_detail"] = blocked_detail
            final_payload["blocked_action"] = requested_displays[len(executed_payloads)]
        if stop_reason == "no_op_action":
            noop_index = len(executed_payloads)
            final_payload["no_op_action_index"] = noop_index
            final_payload["no_op_action"] = final_payload["executed_actions"][-1]
            skipped = requested_displays[noop_index:]
            final_payload["stop_detail"] = (
                f"Action {noop_index} of {batch_size} "
                + (
                    f"({final_payload['executed_actions'][-1]!r}) EXECUTED but produced no "
                    f"change inside the board area - only the outer {NOOP_GUARD_BORDER_WIDTH} "
                    "cells, where a timer or remaining-steps bar usually sits, could have moved. "
                    if NEW_NOOP_WORDING else
                    f"({final_payload['executed_actions'][-1]!r}) EXECUTED but changed nothing "
                    f"outside the {NOOP_GUARD_BORDER_WIDTH}-cell border (treated as HUD). "
                )
                +
                f"Actions {noop_index + 1}-{batch_size} ({skipped!r}) were SKIPPED and did "
                "not execute. Indices are 1-based positions in your submitted list. "
                f"Net effect: only the first {max(0, noop_index - 1)} executed action(s) "
                "changed the board; the final executed action moved nothing. Do NOT "
                "update your position estimate by counting submitted moves - re-locate "
                "the player from current_frame before planning. "
                "The per-action verdicts are listed in action_trace. "
                "This stop is about one action from one board state, not about batch "
                "size - batching is still the right approach, and the stop is what keeps "
                "a wrong movement model from costing the whole sequence."
            )
        self.write_viewer_payload()
        return final_payload

    def _execute_auto_reset(self) -> None:
        action = arcengine.ActionInput(id=arcengine.GameAction.RESET, data={})
        self._execute_action(
            action, batch_index=1, batch_size=1, generated_tokens=0, automatic=True
        )

    def _run_auto_probe(self) -> None:
        """Before the model's first turn, try each control once on level 1.

        See inference.agent.auto_probe. The actions go through _execute_action,
        so they land in `history` like any other (marked automatic) and count
        toward the level. The report is handed to the analyzer for its openers.
        Fails open: a probe error must never cost the game.
        """
        if _level_number(self.game) != 1 or self.should_stop() or _is_engine_game_over(self.game):
            return

        def execute(action: str, row: int | None, col: int | None) -> dict[str, Any] | None:
            engine_name = to_engine_action(action)
            if engine_name is None or self.should_stop():
                return None
            action_id = arcengine.GameAction.from_name(engine_name)
            if action_id.value not in self.game.current_state.available_actions:
                return None
            data = {"x": col, "y": row} if action_id == arcengine.GameAction.ACTION6 else {}
            payload = self._execute_action(
                arcengine.ActionInput(id=action_id, data=data),
                batch_index=1, batch_size=1, generated_tokens=0, automatic=True,
            )
            return {**payload, "grid": _grid_from_state(self.game.current_state)}

        try:
            steps = auto_probe.run_probe(
                _grid_from_state(self.game.current_state),
                to_model_actions(_engine_action_names(self.game)),
                execute,
            )
            self.analyzer.auto_probe_lines = auto_probe.render_report(steps)
            self.analyzer.auto_probe_level = _level_number(self.game)
        except Exception:
            log.warning(
                "%s: auto-probe failed; playing on without it",
                getattr(self.game.game_run, "game_id", "?"), exc_info=True,
            )

    def _execute_action(
        self,
        action: arcengine.ActionInput,
        *,
        batch_index: int,
        batch_size: int,
        generated_tokens: int | None = None,
        flush_viewer_payload: bool = True,
        automatic: bool = False,
    ) -> dict[str, Any]:
        previous_grid = _grid_from_state(self.game.current_state)
        previous_completed = int(self.game.current_state.levels_completed)
        if generated_tokens is None:
            current_tokens = _analyzer_reported_tokens(self.analyzer)
            generated_tokens = max(0, current_tokens - self.token_baseline)
            self.token_baseline = current_tokens

        new_state = self.game.execute_action(
            action, generated_tokens=generated_tokens, uncached_input_tokens=0
        )
        self.last_engine_action = action.id.name
        action_display = _format_action_display(action.id.name, dict(action.data))
        current_frame = Frame(
            grid=_grid_from_state(new_state),
            step=self.action_count,
            level=_level_number(self.game),
        )

        completed = int(new_state.levels_completed)
        reward = float(completed - previous_completed) / max(
            1.0, float(self.game.number_of_levels)
        )
        raw_state = new_state.raw.state
        new_grid = _grid_from_state(new_state)
        board_changed = previous_grid != new_grid
        gameplay_changed = _interior_gameplay_changed(previous_grid, new_grid)
        no_op = gameplay_changed is False
        animation_summary = None
        if ANIMATION_ENABLED and action.id.name != "RESET":
            # The auto-reset after a death runs through here too. Letting it
            # touch the record would clear the fatal action's animation before
            # the model ever reads it - and a death is the case where those
            # frames matter most, since the board it now sees is the restarted
            # level and the death exists nowhere else.
            chain = _animation_chain(previous_grid, new_state)
            animation_summary = summarize_animation(chain)
            # ONE record, replaced by every executed action - animated or not.
            # A no-op that animated is the case this exists for, and the guards
            # refuse the NEXT action without executing it, so the record still
            # describes the animated one when the model reads the opener.
            self.animation_record = (
                {
                    "action_display": action_display,
                    "chain": chain,
                    "summary": animation_summary,
                    "step": self.action_count,
                    "level": _level_number(self.game),
                }
                if animation_summary
                else None
            )
        level_completed = bool(
            new_state.just_won_level and raw_state != arcengine.GameState.WIN
        )
        payload = {
            "executed": True,
            "action_num": self.action_count,
            "level": _level_number(self.game),
            "score": completed,
            "reward": reward,
            "state": raw_state.name,
            "valid_actions": to_model_actions(_engine_action_names(self.game)),
            "board_changed": board_changed,
            "gameplay_changed": gameplay_changed,
            "no_op": no_op,
            "done": raw_state == arcengine.GameState.WIN,
            "level_completed": level_completed,
            "game_over": raw_state == arcengine.GameState.GAME_OVER,
            "run_complete": raw_state == arcengine.GameState.WIN,
            "action_name": action.id.name,
            "action_data": (
                _model_mouse_action_data(action.data)
                if action.id == arcengine.GameAction.ACTION6
                else dict(action.data)
            ),
            "action_display": action_display,
            "batch_index": batch_index,
            "batch_size": batch_size,
            **self.timing_payload(),
        }
        if animation_summary:
            payload["animation"] = animation_summary
            # the raw chain, for rendering an image in the opener. Dropped by
            # _compact_action_result, so it never reaches the model as text.
            payload["animation_chain"] = chain
        # Store the individual action before step_env folds results into a batch.
        # Keep only compact metadata, without grids, animation chains, or clocks.
        payload["automatic"] = automatic
        result_keys = (
            "executed", "action_num", "level", "score", "state", "valid_actions",
            "board_changed", "gameplay_changed", "no_op", "done", "level_completed",
            "game_over", "run_complete", "action_name", "action_data", "action_display",
            "animation", "automatic",
        )
        individual_result = copy.deepcopy({k: payload[k] for k in result_keys if k in payload})
        self.history_entries.append(
            HistoryEntry(action=action_display, frame=current_frame, result=individual_result)
        )
        self.write_runtime_state()
        self._append_action_viewer_event(payload, current_frame)
        if flush_viewer_payload:
            self.write_viewer_payload()
        return payload


@dataclass
class HarnessSolver(Solver):
    """Run the existing tool-using harness as a TAAF ``Solver``."""

    label: str = "HarnessSolver"
    model: str = ""
    analyzer_timeout: float | None = 120.0
    max_actions_per_game: int | None = None
    max_runtime_s_per_game: float | None = None
    # Generated tokens a game may spend before it gives up, counted from the
    # analyzer's own accumulator so it measures what the model produced rather
    # than what was billed. A wall-clock limit means something different on
    # every backend - and on a hosted API it moves hour to hour - so this is
    # what makes an offline run comparable with a Kaggle one. None disables.
    max_generated_tokens_per_game: int | None = None
    concurrency: int = 16
    save_request_logs: bool = False
    start_local_server: bool = False
    local_server_config: str = ""
    local_server_api_key_file: str = ""
    local_server_repo_dir: str = ""
    local_server_port: int | None = None
    local_server_tensor_parallel_size: int | None = None
    local_server_count: int = 1
    kaggle_enable_vllm: bool = field(default=True, repr=False)
    kaggle_wheelhouse_dataset_source: str = field(
        default=DEFAULT_VLLM_WHEELHOUSE_DATASET_SOURCE, repr=False
    )
    kaggle_model_dataset_source: str = field(
        default=DEFAULT_QWEN_MODEL_DATASET_SOURCE, repr=False
    )
    kaggle_served_model_name: str = field(default=DEFAULT_SERVED_MODEL_NAME, repr=False)
    kaggle_vllm_port: int = field(default=DEFAULT_VLLM_PORT, repr=False)
    kaggle_vllm_max_model_len: int = field(
        default=DEFAULT_VLLM_MAX_MODEL_LEN, repr=False
    )
    kaggle_vllm_tensor_parallel_size: int = field(
        default=DEFAULT_VLLM_TENSOR_PARALLEL_SIZE, repr=False
    )
    kaggle_wheelhouse_stamp_text: str = field(
        default=DEFAULT_WHEELHOUSE_STAMP_TEXT, repr=False
    )
    cancel_drain_timeout_s: float = DEFAULT_CANCEL_DRAIN_TIMEOUT_SECONDS
    analyzer_factory: AnalyzerFactory | None = field(
        default=None, repr=False, compare=False
    )
    _stop_event: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )
    _local_server_started: bool = field(
        default=False, init=False, repr=False, compare=False
    )
    _local_server_env_overrides: dict[str, str] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _local_server_cwd: str = field(default="", init=False, repr=False, compare=False)
    _local_server_api_key: str = field(
        default="", init=False, repr=False, compare=False
    )
    _local_server_base_url: str = field(
        default="", init=False, repr=False, compare=False
    )
    _local_servers: list[_LocalServerRuntime] = field(
        default_factory=list, init=False, repr=False, compare=False
    )
    _local_server_original_env: dict[str, str | None] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    # Custom pool sized to self.concurrency: asyncio.to_thread routes onto
    # Python's default executor, capped at min(32, cpu+4) — which would
    # silently cap real concurrency below self.concurrency.
    _worker_pool: ThreadPoolExecutor | None = field(default=None, init=False, repr=False, compare=False)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["analyzer_factory"] = None
        state.pop("_stop_event", None)
        state.pop("_local_server_started", None)
        state.pop("_local_server_env_overrides", None)
        state.pop("_local_server_cwd", None)
        state.pop("_local_server_api_key", None)
        state.pop("_local_server_base_url", None)
        state.pop("_local_servers", None)
        state.pop("_local_server_original_env", None)
        state.pop("_worker_pool", None)
        state.pop("_warmup_lock", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._stop_event = threading.Event()
        self._local_server_started = False
        self._local_server_env_overrides = {}
        self._local_server_cwd = ""
        self._local_server_api_key = ""
        self._local_server_base_url = ""
        self._local_servers = []
        self._local_server_original_env = {}
        self._worker_pool = None
        self._warmup_lock = threading.Lock()

    def __deepcopy__(self, memo: dict[int, Any]) -> "HarnessSolver":
        cls = type(self)
        new = cls.__new__(cls)
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            if key == "_stop_event":
                object.__setattr__(new, key, threading.Event())
            elif key == "analyzer_factory":
                object.__setattr__(new, key, value)
            elif key == "_local_servers":
                object.__setattr__(new, key, [])
            elif key == "_local_server_original_env":
                object.__setattr__(new, key, {})
            elif key == "_worker_pool":
                object.__setattr__(new, key, None)
            elif key == "_warmup_lock":
                object.__setattr__(new, key, threading.Lock())
            else:
                object.__setattr__(new, key, copy.deepcopy(value, memo))
        return new

    @property
    def kaggle_dataset_sources(self) -> list[str]:
        if not self.kaggle_enable_vllm:
            return []
        return duck_kaggle_dataset_sources(self._kaggle_vllm_config())

    @property
    def kaggle_setup_commands(self) -> list[str]:
        if not self.kaggle_enable_vllm:
            return []
        return [duck_kaggle_setup_command(self._kaggle_vllm_config())]

    @property
    def kaggle_teardown_commands(self) -> list[str]:
        if not self.kaggle_enable_vllm:
            return []
        return [duck_kaggle_teardown_command()]

    def _kaggle_vllm_config(self) -> DuckKaggleVllmConfig:
        return DuckKaggleVllmConfig(
            wheelhouse_dataset_source=self.kaggle_wheelhouse_dataset_source,
            model_dataset_source=self.kaggle_model_dataset_source,
            served_model_name=self.kaggle_served_model_name,
            vllm_port=self.kaggle_vllm_port,
            max_model_len=self.kaggle_vllm_max_model_len,
            tensor_parallel_size=self.kaggle_vllm_tensor_parallel_size,
            wheelhouse_stamp_text=self.kaggle_wheelhouse_stamp_text,
        )

    def _setup(self) -> None:
        # Created here rather than in __init__ because the notebook unpickles a
        # solver whose __dict__ predates these attributes, and __setstate__
        # applies no dataclass defaults. _setup runs after unpickling, so the
        # attributes always exist by the time a game asks for a slot.
        self._warmup_remaining = WARMUP_ACTION_GAMES
        self._warmup_lock = threading.Lock()
        if self.start_local_server:
            self._start_local_servers()
        self._worker_pool = ThreadPoolExecutor(
            max_workers=max(1, int(self.concurrency)),
            thread_name_prefix="harness-game",
        )

    def _teardown(self) -> None:
        if self._local_server_started:
            self._stop_local_servers()
        if self._worker_pool is not None:
            self._worker_pool.shutdown(wait=False)
            self._worker_pool = None

    async def _run_games(self, games: list[taaf.game.Game]) -> None:
        self._stop_event.clear()
        semaphore = asyncio.Semaphore(max(1, int(self.concurrency)))
        pass_indices_by_game_id: dict[str, int] = {}
        loop = asyncio.get_running_loop()
        pool = self._worker_pool

        async def run_one(index: int, pass_index: int, game: taaf.game.Game) -> None:
            async with semaphore:
                args = (game, index, pass_index, self._local_server_for_game_index(index))
                if pool is not None:
                    await loop.run_in_executor(pool, functools.partial(self._play_one, *args))
                else:
                    # _setup wasn't called (direct test invocation).
                    await asyncio.to_thread(self._play_one, *args)

        tasks: list[asyncio.Task[None]] = []
        for index, game in enumerate(games):
            game_id = game.game_run.game_id if game.game_run is not None else str(index)
            pass_index = pass_indices_by_game_id.get(game_id, 0)
            pass_indices_by_game_id[game_id] = pass_index + 1
            tasks.append(asyncio.create_task(run_one(index, pass_index, game)))
        try:
            await asyncio.gather(
                *(asyncio.shield(task) for task in tasks), return_exceptions=True
            )
        except asyncio.CancelledError:
            self._stop_event.set()
            await self._drain_game_tasks(tasks)
            self._finish_remaining(games)
            raise

    async def _drain_game_tasks(self, tasks: list[asyncio.Task[None]]) -> None:
        if not tasks:
            return
        timeout = max(0.0, float(self.cancel_drain_timeout_s))
        if timeout == 0.0:
            return
        done, _pending = await asyncio.wait(tasks, timeout=timeout)
        if done:
            await asyncio.gather(*done, return_exceptions=True)

    def _start_local_servers(self) -> None:
        server_count = self._resolved_local_server_count()
        started: list[_LocalServerRuntime] = []
        self._capture_local_server_process_env()
        try:
            for server_index in range(server_count):
                runtime = self._local_server_settings(
                    server_index=server_index, server_count=server_count
                )
                print(
                    "Starting local inference server inside solver setup "
                    f"(server {server_index + 1}/{server_count})"
                )
                subprocess.run(
                    ["make", "server"],
                    cwd=runtime.repo_dir,
                    env=self._local_server_env(runtime.env_overrides),
                    check=True,
                )
                if runtime.api_key_file.is_file():
                    runtime.api_key = runtime.api_key_file.read_text(
                        encoding="utf-8"
                    ).strip()
                started.append(runtime)
        except Exception:
            self._local_servers = started
            self._local_server_started = bool(started)
            with contextlib.suppress(Exception):
                self._stop_local_servers()
            raise

        self._local_servers = started
        self._local_server_started = bool(started)
        if started:
            first = started[0]
            self._local_server_cwd = str(first.repo_dir)
            self._local_server_env_overrides = first.env_overrides
            self._local_server_api_key = first.api_key
            self._local_server_base_url = first.base_url
            if first.api_key:
                os.environ["LOCAL_ANALYZER_API_KEY"] = first.api_key
                os.environ["OPENAI_API_KEY"] = first.api_key
            if first.base_url:
                os.environ["LOCAL_ANALYZER_BASE_URL"] = first.base_url
                os.environ["OPENAI_BASE_URL"] = first.base_url
                os.environ["LOCAL_ANALYZER_PROVIDER"] = "vllm"
                os.environ["OPENAI_PROVIDER"] = "vllm"

    def _stop_local_servers(self) -> None:
        runtimes = list(reversed(self._local_servers))
        if not runtimes and self._local_server_env_overrides:
            repo_dir = (
                Path(self._local_server_cwd)
                if self._local_server_cwd
                else self._local_server_repo_dir()
            )
            runtimes = [
                _LocalServerRuntime(
                    index=0,
                    repo_dir=repo_dir,
                    api_key_file=Path(
                        self._local_server_env_overrides.get("SERVER_API_KEY_FILE", "")
                    ),
                    env_overrides=self._local_server_env_overrides,
                    base_url=self._local_server_base_url,
                    api_key=self._local_server_api_key,
                )
            ]
        try:
            for runtime in runtimes:
                subprocess.run(
                    ["make", "stop-server"],
                    cwd=runtime.repo_dir,
                    env=self._local_server_env(runtime.env_overrides),
                    check=False,
                )
        finally:
            self._local_servers = []
            self._local_server_started = False
            self._restore_local_server_process_env()

    def _capture_local_server_process_env(self) -> None:
        self._local_server_original_env = {
            key: os.environ.get(key) for key in _LOCAL_SERVER_PROCESS_ENV_KEYS
        }

    def _restore_local_server_process_env(self) -> None:
        if not self._local_server_original_env:
            return
        for key, value in self._local_server_original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._local_server_original_env = {}

    def _local_server_settings(
        self, *, server_index: int, server_count: int
    ) -> _LocalServerRuntime:
        config_path = self.local_server_config.strip()
        if not config_path:
            raise ValueError(
                "local_server_config is required when start_local_server is enabled."
            )

        repo_dir = self._local_server_repo_dir()
        run_dir = (self.job_dir or Path.cwd()).resolve()
        api_key_file = self._local_server_api_key_path(
            server_index=server_index,
            server_count=server_count,
            run_dir=run_dir,
        )
        pid_path = run_dir / (
            "server.pid" if server_count <= 1 else f"server-{server_index}.pid"
        )
        log_path = run_dir / (
            "server.log" if server_count <= 1 else f"server-{server_index}.log"
        )
        port = self._local_server_port(server_index=server_index)
        base_url = f"http://127.0.0.1:{port}/v1" if port is not None else ""
        env_overrides = {
            "CONFIG_PATH": config_path,
            "SERVER_API_KEY_FILE": str(api_key_file),
            "SERVER_PID": str(pid_path),
            "SERVER_LOG": str(log_path),
            "SERVER_TAIL_ON_WAIT": "true",
            "UV_PROJECT_ENVIRONMENT": str(repo_dir / ".venv"),
        }
        venv_python = self._local_server_venv_python(repo_dir)
        if venv_python is not None:
            env_overrides["SERVER_VENV_PYTHON"] = str(venv_python)
            env_overrides["PYTHON"] = str(venv_python)
        if port is not None:
            env_overrides.update(
                {
                    "SERVER_PORT": str(port),
                    "LOCAL_ANALYZER_BASE_URL": base_url,
                    "OPENAI_BASE_URL": base_url,
                    "LOCAL_ANALYZER_PROVIDER": "vllm",
                    "OPENAI_PROVIDER": "vllm",
                }
            )
        if self.local_server_tensor_parallel_size is not None:
            env_overrides["SERVER_TENSOR_PARALLEL_SIZE"] = str(
                int(self.local_server_tensor_parallel_size)
            )
        if server_count > 1:
            env_overrides["CUDA_VISIBLE_DEVICES"] = (
                self._cuda_visible_device_for_server(server_index)
            )
        return _LocalServerRuntime(
            index=server_index,
            repo_dir=repo_dir,
            api_key_file=api_key_file,
            env_overrides=env_overrides,
            base_url=base_url,
        )

    def _resolved_local_server_count(self) -> int:
        if not self.start_local_server:
            return 0
        return max(1, int(self.local_server_count or 1))

    def _local_server_port(self, *, server_index: int) -> int | None:
        if self.local_server_port is None:
            return None
        return int(self.local_server_port) + int(server_index)

    def _local_server_api_key_path(
        self, *, server_index: int, server_count: int, run_dir: Path
    ) -> Path:
        default_name = (
            "server-api-key" if server_count <= 1 else f"server-{server_index}-api-key"
        )
        base_path = self._resolve_local_server_path(
            self.local_server_api_key_file, default=run_dir / default_name
        )
        if server_count <= 1 or not str(self.local_server_api_key_file or "").strip():
            return base_path
        suffix = base_path.suffix
        stem = base_path.name[: -len(suffix)] if suffix else base_path.name
        return base_path.with_name(f"{stem}-{server_index}{suffix}")

    def _cuda_visible_device_for_server(self, server_index: int) -> str:
        visible_devices = [
            device.strip()
            for device in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if device.strip()
        ]
        if server_index < len(visible_devices):
            return visible_devices[server_index]
        return str(server_index)

    def _local_server_repo_dir(self) -> Path:
        repo_dir = (
            Path(self.local_server_repo_dir).expanduser()
            if self.local_server_repo_dir
            else Path(__file__).parents[2]
        )
        repo_dir = repo_dir.resolve()
        if not repo_dir.is_dir():
            raise ValueError(f"local_server_repo_dir does not exist: {repo_dir}")
        return repo_dir

    def _local_server_venv_python(self, repo_dir: Path) -> Path | None:
        repo_venv_python = repo_dir / ".venv" / "bin" / "python"
        if repo_venv_python.is_file():
            return repo_venv_python
        return None

    def _resolve_local_server_path(self, raw_value: str, *, default: Path) -> Path:
        raw = str(raw_value or "").strip()
        if not raw:
            return default.resolve()
        path = Path(raw).expanduser()
        if path.is_absolute():
            return path
        return (self._local_server_repo_dir() / path).resolve()

    def _local_server_env(
        self, overrides: dict[str, str] | None = None
    ) -> dict[str, str]:
        env = os.environ.copy()
        env.update(overrides or self._local_server_env_overrides)
        return env

    def soft_time_remaining_seconds(self) -> float | None:
        if self.soft_end_time is None:
            return None
        now = (
            datetime.now(self.soft_end_time.tzinfo)
            if self.soft_end_time.tzinfo
            else datetime.now()
        )
        return max(0.0, (self.soft_end_time - now).total_seconds())

    def _local_server_for_game_index(
        self, game_index: int
    ) -> _LocalServerRuntime | None:
        if not self._local_servers:
            return None
        return self._local_servers[int(game_index) % len(self._local_servers)]

    def _claim_warmup_slot(self) -> bool:
        """Whether this game should spend one action before the analyzer is up.

        Claimed by whichever runners start first rather than by index: task
        creation order and semaphore fairness both happen to give indices 0..N,
        but neither is guaranteed by the API, and the cost of being wrong is a
        game that never fires the action the deadline depends on.
        """
        lock = getattr(self, "_warmup_lock", None)
        if lock is None:
            return False
        with lock:
            if getattr(self, "_warmup_remaining", 0) <= 0:
                return False
            self._warmup_remaining -= 1
            return True

    def _make_analyzer(
        self,
        game: taaf.game.Game,
        index: int,
        local_server: _LocalServerRuntime | None = None,
    ) -> Any:
        if self.analyzer_factory is not None:
            return self.analyzer_factory(game, index)
        return ToolAgent(
            model=self.model,
            timeout=self.analyzer_timeout,
            save_request_logs=self.save_request_logs,
            # queue position, so the untrimmed priority band has a deterministic
            # order instead of every game tying at the same value
            dispatch_index=index,
            api_key=(
                local_server.api_key
                if local_server is not None
                else self._local_server_api_key
            )
            or None,
            base_url=(
                local_server.base_url
                if local_server is not None
                else self._local_server_base_url
            )
            or None,
            provider="vllm" if local_server is not None else None,
        )

    def _play_one(
        self,
        game: taaf.game.Game,
        index: int,
        pass_index: int,
        local_server: _LocalServerRuntime | None = None,
    ) -> None:
        try:
            assert game.game_run is not None
            run = game.game_run
            run_stem = self._run_stem(run.game_id, pass_index)
            state_path = self._artifacts_dir() / f"{run_stem}_{RUNTIME_STATE_FILENAME}"
            viewer_data_path = self._artifacts_dir() / f"{run_stem}_viewer_data.json"
            transcript_path = self._transcripts_dir() / f"{run_stem}.txt"
            analysis_relpath = f"solver_analysis/{run_stem}.html"
            analyzer = self._make_analyzer(game, index, local_server)
            session = _HarnessGameSession(
                solver=self,
                game=game,
                analyzer=analyzer,
                game_index=index,
                pass_index=pass_index,
                state_path=state_path,
                transcript_path=transcript_path,
                analysis_html_relpath=analysis_relpath,
                stop_event=self._stop_event,
                viewer_data_path=viewer_data_path,
            )
            session.play()
        except Exception as exc:
            self._finish_after_error(game, exc)

    def _artifacts_dir(self) -> Path:
        root = self.job_dir or Path.cwd() / "taaf_harness_artifacts"
        path = root / "artifacts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _transcripts_dir(self) -> Path:
        root = self.job_dir or Path.cwd() / "taaf_harness_artifacts"
        path = root / "transcripts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _run_stem(self, game_id: str, index: int) -> str:
        return f"{artifact_stem(game_id)}_p{index}"

    def _finish_remaining(self, games: list[taaf.game.Game]) -> None:
        for game in games:
            run = game.game_run
            if run is not None and run.final_score is None:
                try:
                    if self._stop_event.is_set() and run.state == "playing":
                        run.state = "cancelled"
                    game.finish_game()
                except Exception:
                    pass

    def _finish_after_error(self, game: taaf.game.Game, exc: Exception) -> None:
        run = game.game_run
        if run is None or run.final_score is not None:
            return
        run.solver_note = f"error: {type(exc).__name__}: {exc}"
        if run.state == "playing":
            run.state = "crashed"
        with contextlib.suppress(Exception):
            game.finish_game()
        if run.final_score is None:
            with contextlib.suppress(Exception):
                run.final_score = run._compute_final_score()
