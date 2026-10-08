# Changes

## Credits

- **The Duck harness**: Jeroen Cottaar and Tufa Labs — https://github.com/Tufalabs/duck-harness
  (no LICENSE file in the repo; the package metadata declares MIT).
- **Milestone 2 solution**: Daniel Franzen, built on the Duck — https://github.com/1zuki/arc-agi-3-solution
  (Apache License 2.0, copy in `solver/LICENSE`).

`solver/` is a copy of Franzen's repository at commit `8b2d83a`. Its agent code matches what
his Kaggle notebook ([arc-agi-3-milestone-2-solution](https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution))
runs. Version 3 of that notebook (4 practice passes on all 25 public games) is saved unchanged as
`solver/base-notebook.ipynb`. `scripts/build_solver_notebook.py` swaps its patch cell for one built
from `solver/`.

## Our changes

| Date | File | Change | Result |
|---|---|---|---|
| 2026-10-05 | `solver/ARC3-Inference/inference/agent/prompts.py` | System prompt: explain that the score is (human/agent actions)², that probes, deaths, UNDO and RESET all count, and that the agent should stop probing once it understands the level | **Dropped, never run.** The baseline already uses 0.68x human actions on the levels it completes |
| 2026-10-07 | `solver/ARC3-Inference/inference/agent/prompts.py` | System prompt "Game overview": a short list of goal types and mechanics that recur across the public games (find the reference panel first, matching colors, select-then-place, HUD progress, level 1 as tutorial), phrased as hypotheses to test | **28.06** leaderboard (baseline 20.60 in our run; Franzen reported 27.89 for the same baseline, so not yet proven) |
| 2026-10-07 | `scripts/build_solver_notebook.py` | The practice run (not scored) defaults to 2 games x 1 pass x 10 min instead of 25 x 4 (~8 h). `--full-practice` restores it | — |
| 2026-10-08 | `solver/ARC3-Inference/inference/agent/prompts.py` | System prompt: levels are weighted by their number, so level 1 (under 4% of a 7-level game) is the cheap place to explore every action and object type; be efficient from level 2 on | Not measured yet (built on the cheat sheet) |
