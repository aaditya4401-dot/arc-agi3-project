# Start here

This folder is the official ARC Prize 2026 Kaggle starter kit
(github.com/arcprize/ARC-AGI-3-Kaggle-Starter), plus three changes:

1. `agent/my_agent.py` is an **explorer** agent instead of random clicks.
   It remembers every screen it has seen, tries each action (and a click on
   each visible object) once per screen, and walks back to screens that still
   have untried moves. Memory is wiped each time a level is cleared.
2. `dev_games/` holds two tiny local games (a maze, and a click-the-lights
   puzzle) so the agent can be tested with no internet at all.
3. `make play-dev` runs the agent on those local games. The notebook now
   targets CPU (`scripts/build_notebook.py`), since nothing here needs a GPU.

## The loop

    make setup        # once: Python 3.12 venv + framework
    make play-dev     # offline smoke test on dev_games/
    make play-local   # the real competition games (downloads them once)
    make submit       # needs .kaggle/access_token + your username in notebooks/kernel-metadata.json

Last local result (`--max-steps 400`): maze solved in 273 actions (random: 0
levels in 80), lights solved in 9 actions.
