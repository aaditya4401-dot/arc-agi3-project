"""Auto-probe: spend a few cheap level-1 actions learning what each control does.

Runs once per game, before the model's first turn. Level k counts k times in
the game score, so level 1 is the cheapest place to spend actions - and the
transcripts show the model burning many turns rediscovering basics such as
"what does UP move" or "what does clicking that do". The probe presses each
directional/SPACE control once, clicks one object of each distinct kind, and
hands the model a short table of what each action changed.

Planning and reporting live here as plain functions; the solver supplies an
`execute` callback, so the same code runs inside the harness and in offline
tests against the public games.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

Grid = tuple[tuple[int, ...], ...]

ENABLED = os.environ.get("ARC3_AUTO_PROBE", "on").strip().lower() in {"1", "on", "true", "yes"}
MAX_ACTIONS = max(0, int(os.environ.get("ARC3_AUTO_PROBE_MAX_ACTIONS", "12")))
MAX_CLICKS = max(0, int(os.environ.get("ARC3_AUTO_PROBE_MAX_CLICKS", "6")))
# Same edge band the harness treats as HUD/timer territory.
BORDER = max(0, int(os.environ.get("ARC3_NOOP_GUARD_BORDER", "4")))

KEY_ORDER = ("UP", "DOWN", "LEFT", "RIGHT", "SPACE")
# Walls, floors and panels are rarely the thing to click.
LARGE_OBJECT_PIXELS = 400
# Pieces and buttons are usually a few cells to a few dozen.
PIECE_PIXELS = (2, 100)
REPORT_DIFF_TOKENS = 90


@dataclass
class ClickTarget:
    row: int
    col: int
    color: int
    pixels: int
    count: int  # how many objects of this kind are on the board


@dataclass
class ProbeStep:
    action: str
    before: Grid
    after: Grid
    target: Optional[ClickTarget] = None
    flags: dict[str, Any] = field(default_factory=dict)

    @property
    def row(self) -> Optional[int]:
        return self.target.row if self.target else None

    @property
    def col(self) -> Optional[int]:
        return self.target.col if self.target else None


# execute(action, row, col) -> {"grid": Grid, "game_over", "level_completed",
# "run_complete", "board_changed", "gameplay_changed", "level"} or None if the
# action could not run.
Execute = Callable[[str, Optional[int], Optional[int]], Optional[dict[str, Any]]]


def _components(grid: Grid) -> list[tuple[int, list[tuple[int, int]]]]:
    """4-connected same-color components, in reading order of their first cell."""
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    seen = [[False] * cols for _ in range(rows)]
    out = []
    for r0 in range(rows):
        for c0 in range(cols):
            if seen[r0][c0]:
                continue
            color = grid[r0][c0]
            cells = []
            stack = [(r0, c0)]
            seen[r0][c0] = True
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and not seen[nr][nc] and grid[nr][nc] == color:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            out.append((color, cells))
    return out


def click_targets(grid: Grid, limit: int, border: int = BORDER) -> list[ClickTarget]:
    """One click per distinct kind of object (same color and shape), best guesses first.

    Skips the background, very large regions, and objects lying entirely in the
    edge band where timers and HUD strips sit. Piece-sized objects come first,
    smallest first, and the first pass takes one per color so a small budget
    still covers every color on the board.
    """
    if limit <= 0 or not grid:
        return []
    rows, cols = len(grid), len(grid[0])
    background = Counter(v for row in grid for v in row).most_common(1)[0][0]
    kinds: dict[tuple, list[list[tuple[int, int]]]] = {}
    for color, cells in _components(grid):
        if color == background or len(cells) > LARGE_OBJECT_PIXELS:
            continue
        r0 = min(r for r, _ in cells); r1 = max(r for r, _ in cells)
        c0 = min(c for _, c in cells); c1 = max(c for _, c in cells)
        if r1 < border or r0 >= rows - border or c1 < border or c0 >= cols - border:
            continue
        shape = tuple(sorted((r - r0, c - c0) for r, c in cells))
        kinds.setdefault((color, shape), []).append(cells)

    candidates = []
    for (color, _), instances in kinds.items():
        cells = instances[0]
        cy = sum(r for r, _ in cells) / len(cells)
        cx = sum(c for _, c in cells) / len(cells)
        r, c = min(cells, key=lambda p: (p[0] - cy) ** 2 + (p[1] - cx) ** 2)
        candidates.append(ClickTarget(r, c, color, len(cells), len(instances)))
    lo, hi = PIECE_PIXELS
    candidates.sort(key=lambda t: (0 if lo <= t.pixels <= hi else 1, t.pixels, t.row, t.col))

    picked: list[ClickTarget] = []
    colors: set[int] = set()
    for t in candidates:
        if t.color not in colors:
            picked.append(t)
            colors.add(t.color)
    picked += [t for t in candidates if t not in picked]
    return picked[:limit]


def run_probe(
    grid: Grid,
    valid_actions: list[str],
    execute: Execute,
    *,
    max_actions: int = MAX_ACTIONS,
    max_clicks: int = MAX_CLICKS,
    border: int = BORDER,
) -> list[ProbeStep]:
    """Press each key once, click one object of each kind, then try UNDO once.

    Stops at the first game over, level completion or win: a death resets the
    level, and a completion means the model should look at the new board.
    """
    steps: list[ProbeStep] = []
    current = grid
    valid = [str(a).upper() for a in valid_actions]

    def probe(action: str, target: Optional[ClickTarget] = None) -> bool:
        nonlocal current
        if len(steps) >= max_actions:
            return False
        result = execute(action, target.row if target else None, target.col if target else None)
        if result is None:
            return True
        after = result["grid"]
        steps.append(ProbeStep(action, current, after, target, result))
        current = after
        return not (result.get("game_over") or result.get("level_completed") or result.get("run_complete"))

    for key in KEY_ORDER:
        if key in valid and not probe(key):
            return steps
    if "MOUSE" in valid:
        budget = min(max_clicks, max_actions - len(steps))
        for target in click_targets(current, budget, border):
            if not probe("MOUSE", target):
                return steps
    if "UNDO" in valid and any(s.flags.get("gameplay_changed") for s in steps):
        probe("UNDO")
    return steps


def _segment(grid: Grid, cache: dict) -> list:
    from inference.utils.grid_utils import ARC_COLOR_CHARS
    from inference.utils.segmentation import segment_layer

    if grid not in cache:
        cache[grid] = segment_layer(grid, ARC_COLOR_CHARS).get("nodes", [])
    return cache[grid]


def _mask_edge(before: Grid, after: Grid, border: int) -> Grid:
    """`after` with its edge band copied from `before`, so only interior changes remain.

    The edge band is where step counters and timer bars sit; left in, every
    move's diff would also list "resized: color c 63->62px at [63, 0]".
    """
    rows, cols = len(after), len(after[0]) if after else 0
    if border <= 0 or rows <= 2 * border or cols <= 2 * border:
        return after
    out = []
    for r in range(rows):
        if r < border or r >= rows - border:
            out.append(before[r])
        else:
            out.append(before[r][:border] + after[r][border:cols - border] + before[r][cols - border:])
    return tuple(out)


def _clean_diff_lines(lines: list[str]) -> str:
    """Join rendered diff lines into one line, dropping pointers to the frame_diff() tool."""
    parts = []
    for line in lines:
        line = re.sub(r"\s*\((?:details omitted; )?call frame_diff\(\)[^)]*\)\.?", "", line)
        line = line.removeprefix("- ").strip()
        if line:
            parts.append(line)
    if not parts:
        return ""
    header, rest = parts[0].rstrip("."), parts[1:]
    return f"{header}: " + "; ".join(rest) if rest else header


def _describe_change(step: ProbeStep, cache: dict, border: int) -> str:
    flags = step.flags
    if flags.get("game_over"):
        return "GAME OVER - this action was fatal from that board; the level restarts"
    if flags.get("run_complete"):
        return "the whole game was won"
    if flags.get("level_completed"):
        return "LEVEL COMPLETED - this action finished level 1"
    if step.before == step.after:
        return "no change anywhere"
    if len(step.before) != len(step.after) or len(step.before[0]) != len(step.after[0]):
        return "the grid changed shape"
    interior = _mask_edge(step.before, step.after, border)
    edge_changed = interior != step.after
    if interior == step.before:
        return (
            f"nothing inside the board changed; only cells within {border} of the edge did "
            "(usually a step counter or timer bar)"
        )
    from inference.agent.tool_agent import _render_auto_frame_diff
    from inference.utils.frame_diff import compute_frame_diff

    diff = compute_frame_diff(
        step.before, interior, _segment(step.before, cache), _segment(interior, cache),
        max_group_match=12,
    )
    # A moving piece reshapes the floor or wall it crosses; those entries say
    # nothing about the control and crowd out the ones that do.
    diff["resized"] = [
        e for e in diff.get("resized") or []
        if (e.get("before_pixels") or 0) <= LARGE_OBJECT_PIXELS
        and e.get("before_pixels") != e.get("after_pixels")
    ]
    cells = len(step.after) * len(step.after[0])
    text = _clean_diff_lines(_render_auto_frame_diff(diff, REPORT_DIFF_TOKENS, total_cells=cells))
    if edge_changed:
        text += " (+ edge band)"
    return text


def render_report(steps: list[ProbeStep], border: int = BORDER) -> list[str]:
    """Prompt lines describing what each probe action did, for the model's openers."""
    if not steps:
        return []
    from inference.utils.grid_utils import ARC_COLOR_CHARS

    cache: dict = {}
    lines = [
        f"Automatic probe: before your first turn the harness executed {len(steps)} probe "
        "action(s) on level 1 to show what each control does. They are in `history` and "
        "`transitions` (result `automatic=True`) and count toward this level's actions. "
        "Each line is what that action changed, starting from the board left by the "
        f"previous line; positions are [row, col]; \"(+ edge band)\" means cells within {border} "
        "of the edge (usually a step counter or timer bar) changed as well:"
    ]
    for step in steps:
        label = step.action
        if step.target is not None:
            t = step.target
            color = ARC_COLOR_CHARS[t.color] if 0 <= t.color < len(ARC_COLOR_CHARS) else t.color
            label = (
                f"MOUSE(row={t.row}, col={t.col}) on a color {color} {t.pixels}px object"
                + (f" (1 of {t.count} like it)" if t.count > 1 else "")
            )
        elif step.action == "UNDO":
            label = "UNDO (right after the line above)"
        lines.append(f"- {label}: {_describe_change(step, cache, border)}")
    lines.append(
        "Treat this as evidence, not a full explanation: effects can depend on position "
        "or state. Do not repeat these probes unless you need to re-check something."
    )
    return lines
