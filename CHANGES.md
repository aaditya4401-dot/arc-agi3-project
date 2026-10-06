# Changes

## Credits

- **The Duck harness**: Jeroen Cottaar and Tufa Labs — https://github.com/Tufalabs/duck-harness
  (no LICENSE file in the repo; the package metadata declares MIT).
- **Milestone 2 solution**: Daniel Franzen, built on the Duck — https://github.com/1zuki/arc-agi-3-solution
  (Apache License 2.0, copy in `solver/LICENSE`).

`solver/` is a copy of Franzen's repository at commit `8b2d83a`. Its agent code matches what
his Kaggle notebook ([arc-agi-3-milestone-2-solution](https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution))
runs. That notebook is saved unchanged as `solver/base-notebook.ipynb`.

## Our changes

| Date | File | Change | Result |
|---|---|---|---|
| 2026-10-05 | `solver/ARC3-Inference/inference/agent/prompts.py` | System prompt: explain that the score is (human/agent actions)², that probes, deaths, UNDO and RESET all count, and that the agent should stop probing once it understands the level | Not measured yet |
