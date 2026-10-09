"""Run the solver's auto-probe on the public games, offline, without the model.

Plays the probe's actions on each game's level 1 with the real game engine and
prints the report the model would receive, plus totals. Needs the games in
environment_files/ (run `make play-local STEPS=1` once to download them).

    .venv/bin/python scripts/test_auto_probe.py            # all games
    .venv/bin/python scripts/test_auto_probe.py ft09,sk48  # some games
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solver" / "ARC3-Inference"))
os.environ.setdefault("EXPOSE_UNDO", "on")  # as in the competition notebook

import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

from inference.agent import auto_probe  # noqa: E402
from inference.agent.action_names import to_engine_action, to_model_actions  # noqa: E402


def to_grid(frame) -> auto_probe.Grid:
    return tuple(tuple(int(v) for v in row) for row in frame.frame[-1])


def interior_changed(a: auto_probe.Grid, b: auto_probe.Grid, border: int) -> bool:
    return any(ra[border:-border] != rb[border:-border] for ra, rb in zip(a[border:-border], b[border:-border]))


def model_actions(frame) -> list[str]:
    return to_model_actions(GameAction.from_id(int(i)).name for i in frame.available_actions)


def main() -> None:
    wanted = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None
    arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ROOT / "environment_files"))
    games = sorted(e.game_id for e in arc.get_environments())
    totals = {"games": 0, "actions": 0, "deaths": 0, "completed": 0, "seconds": 0.0}
    for game_id in games:
        if wanted and game_id.split("-")[0] not in wanted:
            continue
        env = arc.make(game_id.split("-")[0])
        frame = env.reset()
        state = {"frame": frame}

        def execute(action, row, col):
            engine = to_engine_action(action)
            if engine is None:
                return None
            game_action = GameAction.from_name(engine)
            if game_action.value not in state["frame"].available_actions:
                return None
            data = {"x": col, "y": row} if game_action == GameAction.ACTION6 else {}
            before = state["frame"]
            after = env.step(game_action, data=data)
            state["frame"] = after
            a, b = to_grid(before), to_grid(after)
            return {
                "grid": b,
                "board_changed": a != b,
                "gameplay_changed": interior_changed(a, b, auto_probe.BORDER),
                "game_over": after.state == GameState.GAME_OVER,
                "run_complete": after.state == GameState.WIN,
                "level_completed": after.levels_completed > before.levels_completed and after.state != GameState.WIN,
            }

        start = time.time()
        steps = auto_probe.run_probe(to_grid(frame), model_actions(frame), execute)
        report = auto_probe.render_report(steps)
        took = time.time() - start
        totals["games"] += 1
        totals["actions"] += len(steps)
        totals["deaths"] += sum(bool(s.flags.get("game_over")) for s in steps)
        totals["completed"] += sum(bool(s.flags.get("level_completed")) for s in steps)
        totals["seconds"] += took
        print(f"\n######## {game_id}  valid={model_actions(frame)}  probe actions={len(steps)}  ({took:.1f}s)")
        print("\n".join(report[1:-1]))

    n = max(1, totals["games"])
    print(
        f"\n== {totals['games']} games: {totals['actions']} probe actions "
        f"({totals['actions'] / n:.1f}/game), {totals['deaths']} game overs, "
        f"{totals['completed']} level-1 completions, {totals['seconds'] / n:.1f}s/game"
    )


if __name__ == "__main__":
    main()
