"""Build a Kaggle notebook that runs OUR copy of the solver (`solver/`).

Franzen's notebook (`solver/base-notebook.ipynb`) ships Tufa's original Duck
code untouched, plus one big patch (cell 2) that turns it into his solver.
On Kaggle the patch is applied with `git apply` before anything runs.

This script rebuilds that patch so it turns the Duck into our `solver/`
instead, and writes `notebooks/solver-kernel/solver-submission.ipynb`, next to
the kernel-metadata.json (inputs + RTX Pro 6000) that `kaggle kernels push` needs. Every other cell
(model, serving, settings) stays exactly as Franzen had it, except that by
default the practice run ("Save & Run All", not scored) is cut from ~8 hours
(25 games x 4 passes) to ~15 minutes (2 games x 1 pass). The scored
competition run is never changed. Pass --full-practice to keep the long one.

Two options for experiments:
  --franzen-only   keep Franzen's own patch (no changes from solver/), so an
                   experiment is "Franzen + one thing" against "Franzen"
  --set KEY=VALUE  add/override a harness setting in the notebook's settings
                   block (cell 4, `setup_env`); repeatable

Every build also adds a server watchdog (--no-watchdog to leave it out): if
the model server dies mid-run it is restarted, and games wait for it instead
of giving up. See WATCHDOG_CODE.

    [edit solver/...] → python scripts/build_solver_notebook.py → upload to Kaggle

Needs the original Duck repo at `reference/duck-harness` (git clone of
https://github.com/Tufalabs/duck-harness).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DUCK_REPO = ROOT / "reference" / "duck-harness"
DUCK_COMMIT = "7652836"  # the Duck commit Franzen's patch is based on
PATCHED_DIR = "ARC3-Inference/inference"  # Franzen's patch only touches this folder
BASE_NOTEBOOK = ROOT / "solver" / "base-notebook.ipynb"
OUT_NOTEBOOK = ROOT / "notebooks" / "solver-kernel" / "solver-submission.ipynb"
PATCH_CELL = 2
PATCH_HEADER = "%%writefile /kaggle/harness-changes.patch\n"
SETTINGS_CELL = 16  # bm.n_passes, games, time per game
ENV_CELL = 4  # setup_env: the ARC3_* / LOCAL_ANALYZER_* harness settings

# Quick practice: only the non-competition branch of the settings cell changes.
PUBLIC_GAMES = [
    "ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52",
    "lp85", "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26", "sc25", "sk48",
    "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30",
]
QUICK_PRACTICE_GAMES = ["ft09", "lp85"]  # public games the agent usually solves fast
QUICK_PRACTICE_EDITS = [
    ("bm.n_passes = 4\n", "bm.n_passes = 1  # quick practice\n"),
    (
        "demo_excluded_games = [] if TRUE_SUBMISSION else []\n",
        "demo_excluded_games = [] if TRUE_SUBMISSION else "
        f"{[g for g in PUBLIC_GAMES if g not in QUICK_PRACTICE_GAMES]!r}  # quick practice\n",
    ),
    (
        "        bm.solver.max_runtime_s_per_game = 532*60 * bm.solver.concurrency // 110\n",
        "        bm.solver.max_runtime_s_per_game = {minutes}*60  # quick practice\n",
    ),
]


SERVER_CELL = 12  # Franzen's launcher: builds `args`/`env`, starts `proc`, waits for /health
SERVER_CELL_MARKER = "proc = subprocess.Popen(args, env=env"

# Games retry a dead endpoint (10 s backoff + HTTP retries), then give up after
# ARC3_MAX_ANALYZER_FAILURES in a row: Franzen's 10 is ~4 minutes. A restart
# takes ~10 minutes with the model files re-read in parallel and ~30 without
# (crash drill, 2026-10-15). 110 is ~45 minutes, still bounded so a server
# that never returns cannot spin a game for hours.
WATCHDOG_ENV = ["ARC3_MAX_ANALYZER_FAILURES=110"]

# Appended to the launcher cell, so it runs after the server is up. The 2026-10-14
# practice run lost its server to a SIGKILL 2h45m in (host-RAM spike) and every
# game gave up within minutes; nothing restarted it.
WATCHDOG_CODE = '''

# ---- server watchdog (added by scripts/build_solver_notebook.py) ----
import signal, threading

WATCHDOG_MAX_RESTARTS = 3

def _restart_server(reason):
    global proc
    print(f"[watchdog] model server {reason}; restarting", flush=True)
    try:  # the launcher starts it in its own session: take every child with it
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        pass
    proc.wait()
    time.sleep(15)  # let the GPU memory of the dead processes be released
    # The first launch reads the model in ~5 min because cell 6's precache pulls
    # the files into the page cache in parallel; alone, the loader reads them one
    # by one from slow storage (~24 min in the crash drill). Warm them again.
    try:
        threading.Thread(target=precache, args=(MODEL_DIR, DRAFT_MODEL_DIR),
                         kwargs={"threads": 8}, name="watchdog-precache", daemon=True).start()
    except NameError:
        print("[watchdog] precache() not defined; relaunching without it", flush=True)
    with open(LOG, "ab", buffering=0) as logf:
        proc = subprocess.Popen(args, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    print(f"[watchdog] relaunched as pid {proc.pid}", flush=True)
    started = time.time()
    while time.time() - started < 30 * 60 and proc.poll() is None:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{SERVED_MODEL_PORT}/health", timeout=5) as r:
                if r.status == 200:
                    print(f"[watchdog] server healthy again after {int(time.time() - started)}s", flush=True)
                    return
        except Exception:
            pass
        time.sleep(5)

def _server_watchdog():
    restarts = 0
    while restarts < WATCHDOG_MAX_RESTARTS:
        time.sleep(10)
        if proc.poll() is None:
            continue
        restarts += 1
        _restart_server(f"exited with code {proc.returncode} (restart {restarts}/{WATCHDOG_MAX_RESTARTS})")
    print("[watchdog] restart limit reached; no longer watching", flush=True)

threading.Thread(target=_server_watchdog, name="server-watchdog", daemon=True).start()
print("[watchdog] watching the model server", flush=True)
'''


# Practice-only crash drill: SIGKILL the server N minutes after launch so the
# watchdog's restart can be watched on real hardware. Never in a scored rerun.
DRILL_CODE = '''

# ---- crash drill (practice runs only; added by build_solver_notebook --drill-kill-after) ----
def _crash_drill(minutes):
    time.sleep(minutes * 60)
    if TRUE_SUBMISSION:
        return
    print(f"[drill] killing the model server (pid {proc.pid}) to test the watchdog", flush=True)
    os.killpg(proc.pid, signal.SIGKILL)

if not TRUE_SUBMISSION:
    threading.Thread(target=_crash_drill, args=({minutes},), name="crash-drill", daemon=True).start()
'''


def apply_watchdog(cell: dict) -> None:
    src = "".join(cell["source"])
    if SERVER_CELL_MARKER not in src:
        raise SystemExit(f"Cell {SERVER_CELL} is no longer the server launcher; can't add the watchdog")
    cell["source"] = src.rstrip("\n") + "\n" + WATCHDOG_CODE


def apply_drill(cell: dict, minutes: float) -> None:
    cell["source"] = "".join(cell["source"]).rstrip("\n") + "\n" + DRILL_CODE.replace("({minutes},)", f"({minutes},)")


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


def build_patch(solver_ref: str | None = None) -> str:
    if not (DUCK_REPO / ".git").exists():
        raise SystemExit(
            f"Original Duck repo not found at {DUCK_REPO}.\n"
            "  git clone https://github.com/Tufalabs/duck-harness.git reference/duck-harness"
        )
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        # Start from the original Duck code...
        archive = subprocess.run(
            ["git", "archive", DUCK_COMMIT, PATCHED_DIR],
            cwd=DUCK_REPO, check=True, capture_output=True,
        ).stdout
        subprocess.run(["tar", "-x"], cwd=tmp, input=archive, check=True)
        git("init", "-q", cwd=tmp)
        git("add", "-A", cwd=tmp)
        git("-c", "user.name=build", "-c", "user.email=build@local",
            "commit", "-qm", "duck", cwd=tmp)

        # ...replace it with our solver code (the working tree, or a past commit), and diff.
        shutil.rmtree(tmp / PATCHED_DIR)
        if solver_ref:
            snap = tmp / "_snapshot"
            snap.mkdir()
            archive = subprocess.run(
                ["git", "archive", solver_ref, f"solver/{PATCHED_DIR}"],
                cwd=ROOT, check=True, capture_output=True,
            ).stdout
            subprocess.run(["tar", "-x"], cwd=snap, input=archive, check=True)
            shutil.move(str(snap / "solver" / PATCHED_DIR), str(tmp / PATCHED_DIR))
            shutil.rmtree(snap)
        else:
            shutil.copytree(
                ROOT / "solver" / PATCHED_DIR, tmp / PATCHED_DIR,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        git("add", "-A", cwd=tmp)
        return git("diff", "--cached", cwd=tmp)


def apply_quick_practice(cell: dict, minutes: int) -> None:
    src = "".join(cell["source"])
    for old, new in QUICK_PRACTICE_EDITS:
        if src.count(old) != 1:
            raise SystemExit(f"Settings cell changed upstream; can't find exactly one {old.strip()!r}")
        src = src.replace(old, new.replace("{minutes}", str(minutes)))
    cell["source"] = src


ENV_APPLY_LINE = "os.environ.update({k: str(v) for k,v in setup_env.items()})"


def apply_env_overrides(cell: dict, overrides: list[str]) -> None:
    """Update setup_env just before it is exported, so overrides win.

    Inserting into the dict literal is not enough: the cell later runs
    `if USE_PRIORITY_SCHEDULING: setup_env.update({... 'ARC3_MAX_ACTIVE_STREAMS': 10 ...})`,
    which would silently undo an override of any key it sets.
    """
    src = "".join(cell["source"])
    if src.count(ENV_APPLY_LINE) != 1:
        raise SystemExit(f"Can't find exactly one {ENV_APPLY_LINE!r} in cell {ENV_CELL}")
    lines = ["########## experiment overrides (build_solver_notebook --set) ##########", "setup_env.update({"]
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        lines.append(f"    {key.strip()!r}: {value.strip()!r},")
    lines += ["})", ""]
    cell["source"] = src.replace(ENV_APPLY_LINE, "\n".join(lines) + ENV_APPLY_LINE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full-practice", action="store_true",
                        help="Keep the ~8 h practice run (25 games x 4 passes) instead of ~15 min.")
    parser.add_argument("--franzen-only", action="store_true",
                        help="Keep Franzen's own harness patch instead of building one from solver/.")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Add/override a harness setting in setup_env (repeatable).")
    parser.add_argument("--solver-ref", metavar="COMMIT",
                        help="Build the patch from solver/ as of this git commit instead of the working tree.")
    parser.add_argument("--no-watchdog", action="store_true",
                        help="Leave out the model-server watchdog.")
    parser.add_argument("--practice-minutes", type=int, default=10,
                        help="Minutes per game in the quick practice run (default 10).")
    parser.add_argument("--drill-kill-after", type=float, metavar="MINUTES",
                        help="Practice runs only: SIGKILL the model server this long after launch.")
    args = parser.parse_args()

    nb = json.loads(BASE_NOTEBOOK.read_text())
    cell = nb["cells"][PATCH_CELL]
    if not "".join(cell["source"]).startswith(PATCH_HEADER):
        raise SystemExit(f"Cell {PATCH_CELL} of {BASE_NOTEBOOK.name} is not the patch cell.")
    if args.franzen_only:
        patch = "".join(cell["source"])[len(PATCH_HEADER):]
    else:
        patch = build_patch(args.solver_ref)
        # Franzen's cell has no trailing newline; keep the same shape.
        cell["source"] = PATCH_HEADER + patch.rstrip("\n")
    if not args.full_practice:
        apply_quick_practice(nb["cells"][SETTINGS_CELL], args.practice_minutes)
    overrides = list(args.set)
    if not args.no_watchdog:
        apply_watchdog(nb["cells"][SERVER_CELL])
        if args.drill_kill_after:
            apply_drill(nb["cells"][SERVER_CELL], args.drill_kill_after)
        # explicit --set values come last, so they win
        overrides = WATCHDOG_ENV + overrides
    if overrides:
        apply_env_overrides(nb["cells"][ENV_CELL], overrides)

    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None

    OUT_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    files = [l.split(" b/")[-1] for l in patch.splitlines() if l.startswith("diff --git")]
    practice = "full (~8 h)" if args.full_practice else f"quick ({'+'.join(QUICK_PRACTICE_GAMES)}, ~15 min)"
    print(f"[build_solver_notebook] Wrote {OUT_NOTEBOOK.relative_to(ROOT)} "
          f"(patch: {'Franzen original' if args.franzen_only else 'solver/@' + (args.solver_ref or 'working tree')}, {len(files)} files; "
          f"practice run: {practice}; watchdog: {'off' if args.no_watchdog else 'on'}; "
          f"overrides: {len(args.set)})")


if __name__ == "__main__":
    main()
