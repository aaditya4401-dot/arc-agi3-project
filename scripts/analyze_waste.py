"""Where does an agent waste actions? Usage: python scripts/analyze_waste.py <run>/artifacts

Reads the per-run *_events.jsonl files a TAAF/Duck run writes, sorts every action
into ok / no-op / HUD-only / death / reset, and compares completed levels against the
human baselines in environment_files/ (run `make play-local STEPS=1` once to fetch them).
"""
import glob, json, hashlib, statistics as st
from collections import Counter, defaultdict
import numpy as np

import sys
RUN = sys.argv[1] if len(sys.argv) > 1 else "reference/duck-harness/example-run/artifacts"
BASE = {m["game_id"]: m["baseline_actions"] for m in (json.load(open(p)) for p in glob.glob("environment_files/*/*/metadata.json"))}
BORDER = 4

def h(b): return hashlib.md5(b.tobytes()).hexdigest()

def level_score(base, acts):
    return min(115.0, (base / acts) ** 2 * 100) if acts else 0.0

tot = Counter()             # action categories over all actions
lev_rows = []               # one row per (run, level attempted)
batch_sizes = Counter()
turn_sizes = []

for f in sorted(glob.glob(f"{RUN}/*_events.jsonl")):
    gid = f.split("/")[-1].split("_p")[0]
    evs = [json.loads(l) for l in open(f)]
    prev = None
    level = 1
    seg = Counter()               # categories in current level
    seen = {}                     # (board_hash, action) -> outcome  ('noop'|'hud'|'death'|'ok')
    attempt_actions = 0           # actions since level start / last reset
    turns = Counter()
    for e in evs:
        if e["type"] == "initial":
            prev = np.array(e["board"], dtype=np.int8); continue
        if e["type"] != "action": continue
        cur = np.array(e["board"], dtype=np.int8)
        act = e.get("action_display", "?")
        turns[e.get("analysis_step")] += 1
        if e.get("batch_index") == 1: batch_sizes[e.get("batch_size", 1)] += 1
        key = (h(prev), act) if prev is not None else None
        diff = (prev != cur) if prev is not None and prev.shape == cur.shape else None
        if e.get("game_over"):
            cat = "death"
        elif act == "RESET":
            cat = "reset"
        elif diff is not None and not diff.any():
            cat = "noop"
        elif diff is not None and not diff[BORDER:-BORDER, BORDER:-BORDER].any() and not e.get("level_completed"):
            cat = "hud_only"
        else:
            cat = "ok"
        known = seen.get(key)
        if known in ("noop", "hud_only", "death") and cat in ("noop", "hud_only", "death"):
            seg["repeat_known_bad"] += 1
        if key: seen[key] = cat
        seg[cat] += 1; seg["all"] += 1
        attempt_actions += 1
        if cat == "death":
            seg["lost_in_deaths"] += attempt_actions; attempt_actions = 0
        if act == "RESET": attempt_actions = 0
        if e.get("level_completed") or e.get("run_complete"):
            lev_rows.append(dict(game=gid, level=level, done=True, **seg))
            tot.update(seg); seg = Counter(); seen = {}; attempt_actions = 0
            level += 1
        prev = cur
    if seg["all"]:
        lev_rows.append(dict(game=gid, level=level, done=False, **seg)); tot.update(seg)
    turn_sizes += list(turns.values())

A = tot["all"]
print(f"runs: {len(glob.glob(f'{RUN}/*_events.jsonl'))}   actions: {A}")
print("\n== What every action did ==")
for k in ["ok", "noop", "hud_only", "death", "reset"]:
    print(f"  {k:10} {tot[k]:6}  {100*tot[k]/A:5.1f}%")
print(f"  of the no-op/HUD/death moves, exact repeats of an already-known bad move: {tot['repeat_known_bad']} ({100*tot['repeat_known_bad']/A:.1f}% of all)")
print(f"  moves inside attempts that ended in game over: {tot['lost_in_deaths']} ({100*tot['lost_in_deaths']/A:.1f}%)")

done = [r for r in lev_rows if r["done"]]
open_ = [r for r in lev_rows if not r["done"]]
print("\n== Completed vs unfinished levels ==")
print(f"  levels completed: {len(done)}   actions spent on them: {sum(r['all'] for r in done)} ({100*sum(r['all'] for r in done)/A:.0f}%)")
print(f"  levels left unfinished: {len(open_)}   actions spent on them: {sum(r['all'] for r in open_)} ({100*sum(r['all'] for r in open_)/A:.0f}%)  <- score 0")

ratios, sc_now, sc_clean, sc_clean_death = [], [], [], []
for r in done:
    b = BASE[r["game"]][r["level"] - 1]
    a = r["all"]
    waste = r.get("noop", 0) + r.get("hud_only", 0) + r.get("repeat_known_bad", 0) * 0  # repeats already inside noop/hud/death
    ratios.append(a / b)
    sc_now.append(level_score(b, a))
    sc_clean.append(level_score(b, max(1, a - r.get("noop", 0) - r.get("hud_only", 0))))
    sc_clean_death.append(level_score(b, max(1, a - r.get("noop", 0) - r.get("hud_only", 0) - r.get("lost_in_deaths", 0))))
print("\n== Completed levels: agent actions / human actions ==")
q = np.percentile(ratios, [25, 50, 75, 90])
print(f"  median {q[1]:.1f}x   (25%: {q[0]:.1f}x, 75%: {q[2]:.1f}x, 90%: {q[3]:.1f}x)")
print(f"  within 1x human: {sum(x<=1 for x in ratios)}   1-2x: {sum(1<x<=2 for x in ratios)}   2-4x: {sum(2<x<=4 for x in ratios)}   >4x: {sum(x>4 for x in ratios)}")
print(f"  mean level score now: {st.mean(sc_now):.1f}/100")
print(f"  ...if no-op + HUD-only moves were removed: {st.mean(sc_clean):.1f}/100")
print(f"  ...and also moves lost in game-over attempts: {st.mean(sc_clean_death):.1f}/100")

print("\n== Batching ==")
n = sum(batch_sizes.values())
print(f"  action calls: {n}   single-move calls: {100*batch_sizes[1]/n:.0f}%   median moves per AI turn: {st.median(turn_sizes)}")

print("\n== Per game (completed levels): median actions/human, waste share ==")
by = defaultdict(list)
for r in lev_rows: by[r["game"]].append(r)
for g, rs in sorted(by.items()):
    d = [r for r in rs if r["done"]]
    a = sum(r["all"] for r in rs)
    w = sum(r.get("noop", 0) + r.get("hud_only", 0) for r in rs)
    dd = sum(r.get("lost_in_deaths", 0) for r in rs)
    med = st.median([r["all"] / BASE[g][r["level"] - 1] for r in d]) if d else float("nan")
    print(f"  {g:14} completed {len(d):3}  median {med:5.1f}x  noop+hud {100*w/a:4.0f}%  in-deaths {100*dd/a:4.0f}%")

print("\n== Which levels get completed ==")
c = Counter(r["level"] for r in done)
for lv in sorted(c): print(f"  level {lv}: completed {c[lv]} times out of 500 runs")
