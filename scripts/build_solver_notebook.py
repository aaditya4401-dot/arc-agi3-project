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
SETTINGS_CELL = 16

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
        "        bm.solver.max_runtime_s_per_game = 10*60  # quick practice\n",
    ),
]


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


def build_patch() -> str:
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

        # ...replace it with our solver code, and diff.
        shutil.rmtree(tmp / PATCHED_DIR)
        shutil.copytree(
            ROOT / "solver" / PATCHED_DIR, tmp / PATCHED_DIR,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        git("add", "-A", cwd=tmp)
        return git("diff", "--cached", cwd=tmp)


def apply_quick_practice(cell: dict) -> None:
    src = "".join(cell["source"])
    for old, new in QUICK_PRACTICE_EDITS:
        if src.count(old) != 1:
            raise SystemExit(f"Settings cell changed upstream; can't find exactly one {old.strip()!r}")
        src = src.replace(old, new)
    cell["source"] = src


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full-practice", action="store_true",
                        help="Keep the ~8 h practice run (25 games x 4 passes) instead of ~15 min.")
    args = parser.parse_args()

    patch = build_patch()
    nb = json.loads(BASE_NOTEBOOK.read_text())
    cell = nb["cells"][PATCH_CELL]
    if not "".join(cell["source"]).startswith(PATCH_HEADER):
        raise SystemExit(f"Cell {PATCH_CELL} of {BASE_NOTEBOOK.name} is not the patch cell.")
    # Franzen's cell has no trailing newline; keep the same shape.
    cell["source"] = PATCH_HEADER + patch.rstrip("\n")
    if not args.full_practice:
        apply_quick_practice(nb["cells"][SETTINGS_CELL])

    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None

    OUT_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    files = [l.split(" b/")[-1] for l in patch.splitlines() if l.startswith("diff --git")]
    practice = "full (~8 h)" if args.full_practice else f"quick ({'+'.join(QUICK_PRACTICE_GAMES)}, ~15 min)"
    print(f"[build_solver_notebook] Wrote {OUT_NOTEBOOK.relative_to(ROOT)} "
          f"(patch: {len(files)} files, {len(patch.splitlines())} lines; practice run: {practice})")


if __name__ == "__main__":
    main()
