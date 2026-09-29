"""Your ARC-AGI-3 agent. This is the *only* file you should normally edit.

`scripts/build_notebook.py` splices the contents of this file into the
Kaggle submission notebook, so your local dev loop and your Kaggle
submission stay in lock-step:

    [edit my_agent.py] → [make play-local] → [make submit]

Strategy: systematic exploration instead of random button mashing.

  1. Every screen we see is a "state" (a hash of the 64×64 grid).
  2. In each state the candidate moves are the simple actions the game
     allows, plus one click (ACTION6) per visible object.
  3. We try every untried move once, remember where it led, and when the
     current screen has nothing left to try we walk the shortest known path
     to a screen that still does. Moves that change nothing are never
     repeated.
  4. When a level is completed the map is wiped and exploration restarts,
     because the next level is a new puzzle.

No learning, no LLM: it's a baseline that beats random on games whose
screens are deterministic, and a clean place to plug smarter ideas in.

Contract (enforced by the ARC-AGI-3-Agents framework):
  - Subclass `agents.agent.Agent`.
  - Class must be named `MyAgent` (the notebook's __init__.py registers it).
  - Implement `is_done(frames, latest_frame) -> bool`.
  - Implement `choose_action(frames, latest_frame) -> GameAction`.
"""
from __future__ import annotations

import hashlib
import random
from collections import Counter, deque
from typing import Any, Optional

import numpy as np
from arcengine import FrameData, GameAction, GameState

# When run inside the ARC-AGI-3-Agents framework (locally or on Kaggle)
# the `agents` package is on sys.path, so this import resolves.
from agents.agent import Agent

# A move is ("simple", action_id) or ("click", x, y).
Move = tuple

MAX_CLICK_TARGETS = 32


def grid_of(frame: FrameData) -> np.ndarray:
    """The last 64×64 grid of a frame (an action can emit several)."""
    return np.asarray(frame.frame[-1], dtype=np.int16)


def state_key(grid: np.ndarray) -> str:
    return hashlib.blake2b(grid.tobytes(), digest_size=12).hexdigest()


def click_targets(grid: np.ndarray) -> list[tuple[int, int]]:
    """One click point per connected same-colour blob that isn't background.

    Smallest blobs first: buttons and pieces tend to be small, walls and
    floors big.
    """
    background = Counter(grid.flatten().tolist()).most_common(1)[0][0]
    h, w = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    blobs: list[tuple[int, int, int]] = []  # (size, x, y)
    for y0 in range(h):
        for x0 in range(w):
            if seen[y0, x0] or grid[y0, x0] == background:
                continue
            colour = grid[y0, x0]
            cells = []
            queue = deque([(y0, x0)])
            seen[y0, x0] = True
            while queue:
                y, x = queue.popleft()
                cells.append((y, x))
                for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and grid[ny, nx] == colour:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
            # Click the member cell closest to the blob's centre.
            cy = sum(c[0] for c in cells) / len(cells)
            cx = sum(c[1] for c in cells) / len(cells)
            y, x = min(cells, key=lambda c: (c[0] - cy) ** 2 + (c[1] - cx) ** 2)
            blobs.append((len(cells), x, y))
    blobs.sort()
    return [(x, y) for _, x, y in blobs[:MAX_CLICK_TARGETS]]


class MyAgent(Agent):
    """Explores each level's state graph breadth-first. See module docstring."""

    # Upper bound on actions per game; the framework also enforces global limits.
    MAX_ACTIONS = 400

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        random.seed(hash(self.game_id) % 1_000_000)
        self._reset_memory()
        self._level = 0
        self._last: Optional[tuple[str, Move]] = None  # (state, move) just taken

    def _reset_memory(self) -> None:
        self.untried: dict[str, list[Move]] = {}
        self.graph: dict[str, dict[Move, str]] = {}

    @property
    def name(self) -> str:
        return f"{super().name}.explorer.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER) or latest_frame.is_empty():
            # A death is an edge worth remembering: don't take that move again.
            if self._last and latest_frame.state is GameState.GAME_OVER:
                prev, move = self._last
                self.graph.setdefault(prev, {})[move] = "GAME_OVER"
            self._last = None
            return GameAction.RESET

        if latest_frame.levels_completed != self._level:
            self._level = latest_frame.levels_completed
            self._reset_memory()
            self._last = None

        grid = grid_of(latest_frame)
        here = state_key(grid)
        if here not in self.untried:
            self.untried[here] = self._moves_for(grid, latest_frame.available_actions)
            random.shuffle(self.untried[here])
            self.graph.setdefault(here, {})

        if self._last is not None:
            prev, move = self._last
            self.graph[prev][move] = here

        move = self._next_move(here)
        self._last = (here, move)
        return self._to_action(move)

    def _moves_for(self, grid: np.ndarray, available: list[int]) -> list[Move]:
        ids = [a for a in (available or range(1, 8)) if a != 0]
        moves: list[Move] = [("simple", a) for a in ids if a != 6]
        if 6 in ids:
            moves += [("click", x, y) for x, y in click_targets(grid)]
        return moves

    def _next_move(self, here: str) -> Move:
        if self.untried[here]:
            return self.untried[here].pop()
        path = self._path_to_frontier(here)
        if path:
            return path[0]
        # Everything reachable is explored: take any move that changes the screen.
        useful = [m for m, s in self.graph[here].items() if s not in (here, "GAME_OVER")]
        return random.choice(useful or list(self.graph[here]) or [("simple", 1)])

    def _path_to_frontier(self, start: str) -> list[Move]:
        """Shortest known move sequence to a state that still has untried moves."""
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            state, path = queue.popleft()
            if path and self.untried.get(state):
                return path
            for move, nxt in self.graph.get(state, {}).items():
                if nxt not in visited and nxt != "GAME_OVER":
                    visited.add(nxt)
                    queue.append((nxt, path + [move]))
        return []

    @staticmethod
    def _to_action(move: Move) -> GameAction:
        if move[0] == "click":
            action = GameAction.ACTION6
            action.set_data({"x": int(move[1]), "y": int(move[2])})
            action.reasoning = {"why": "explore: click object", "x": move[1], "y": move[2]}
        else:
            action = GameAction.from_id(move[1])
            action.reasoning = f"explore: action {move[1]}"
        return action
