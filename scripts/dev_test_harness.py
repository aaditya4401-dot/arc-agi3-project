"""Dev-test harness settings offline: the real game loop against a fake model server.

Plays a real public game with Franzen's real harness (TAAF session, ToolAgent,
Python sandbox), but the model is a tiny local HTTP server that answers
instantly with a canned "inspect, then press a random arrow" tool call, and a
canned text reply when the harness asks for a summary. No GPU, no Qwen.

What this can catch: crashes, settings that never take effect, and how much
model-server work a setting adds (requests, prompt tokens, prefix-cache reuse).
What it cannot tell: whether the real model plays better.

Every run uses the competition notebook's settings (cell 4 of
solver/base-notebook.ipynb), with the memory sizes scaled down so trimming
happens within a short game, plus whatever --set adds.

    .venv/bin/python scripts/dev_test_harness.py --label base
    .venv/bin/python scripts/dev_test_harness.py --label compaction \
        --set ARC3_SUMMARY_INTERVAL_TOKENS=6000 --set ARC3_DRAIN_STOP_AT_SUMMARIES=1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "solver" / "base-notebook.ipynb"
SETTINGS_CELL = 4
SUMMARY_MARKER = "Summarize everything above"

# The competition window is 128k with a 58k drain; a fake game is ~60 turns,
# so scale memory down ~4x to see several trims.
SCALED = {
    "LOCAL_ANALYZER_CONTEXT_WINDOW": 32 * 1024,
    "LOCAL_ANALYZER_MAX_OUTPUT": 2 * 1024,
    "ARC3_CONTEXT_DRAIN_TOKENS": 14 * 1024,
}
ARROWS = ["UP", "DOWN", "LEFT", "RIGHT"]


def notebook_settings() -> dict[str, str]:
    """The setup_env dict from the competition notebook, evaluated."""
    src = "".join(json.loads(NOTEBOOK.read_text())["cells"][SETTINGS_CELL]["source"])
    start = src.index("setup_env = {")
    end = src.index("\n}\n", start) + 2
    scope = {"SERVER_BASE_URL": "", "SERVED_MODEL_NAME": "flashnext"}
    exec(src[start:end], scope)
    return {k: str(v) for k, v in scope["setup_env"].items()}


# ---------------------------------------------------------------- fake model

class FakeModel(BaseHTTPRequestHandler):
    log_path: Path
    rng = random.Random(0)
    lock = threading.Lock()

    def log_message(self, *_):  # silence http.server
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        messages = body.get("messages") or []
        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), {})
        is_summary = SUMMARY_MARKER in json.dumps(last_user.get("content"))
        with self.lock:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"summary": is_summary, "messages": [json.dumps(m, sort_keys=True) for m in messages]}) + "\n")
        acted_last = bool(messages) and messages[-1].get("role") == "tool" and '"executed": true' in str(messages[-1].get("content"))
        if is_summary:
            message = {"role": "assistant", "content": "SUMMARY: arrows move the orange/blue pair one cell; the yellow HUD bar shrinks each move; goal not found yet. " * 12}
            finish = "stop"
        elif acted_last:
            message = {"role": "assistant", "content": "Moved; will re-check next turn.", "reasoning_content": "The move executed. " * 40}
            finish = "stop"
        else:
            with self.lock:
                move = self.rng.choice(ARROWS)
            code = (
                "seg = current_frame.segmentation\n"
                "print(len(seg['nodes']), current_frame.level, current_frame.step)\n"
                f"r = action(['{move}'])\n"
                "print(r.get('board_changed'), r.get('level_completed'))\n"
            )
            message = {
                "role": "assistant", "content": "",
                "reasoning_content": "Let me look at the board and try a move to learn the mechanics. " * 30,
                "tool_calls": [{"id": f"call_{time.time_ns()}", "type": "function",
                                "function": {"name": "python", "arguments": json.dumps({"code": code})}}],
            }
            finish = "tool_calls"
        chars = sum(len(m) for m in json.dumps(messages))
        reply = {
            "id": "fake", "object": "chat.completion", "model": "flashnext",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": chars // 3, "completion_tokens": 400, "total_tokens": chars // 3 + 400},
        }
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


# ---------------------------------------------------------------- one run

def run_game(game: str, max_actions: int, out_dir: Path, port: int) -> dict:
    """Runs inside a child process whose environment already holds the settings."""
    sys.path[:0] = [str(ROOT / "solver" / "ARC3-Inference"), str(ROOT / "solver" / "tufa-arc-agi-framework" / "src")]
    # taaf.diagnostics (plots, movies, stats) needs imageio/scipy; the game loop does not
    sys.modules["taaf.diagnostics"] = types.ModuleType("taaf.diagnostics")
    import arc_agi
    import taaf.game_api
    from inference.agent.tool_agent import ToolAgent
    from inference.framework import solver as S

    spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE,
                                    environments_dir=str(ROOT / "environment_files"))
    g = taaf.game_api.GameAPI(env_name=game, arcade_spec=spec)
    g.start_game()
    solver = S.HarnessSolver(model="local", max_runtime_s_per_game=900, max_actions_per_game=max_actions)
    solver.job_dir = out_dir
    analyzer = ToolAgent(model="local", timeout=60, dispatch_index=0, api_key="x",
                         base_url=f"http://127.0.0.1:{port}/v1", provider="vllm")
    session = S._HarnessGameSession(
        solver=solver, game=g, analyzer=analyzer, game_index=0, pass_index=0,
        state_path=out_dir / "state.json", transcript_path=out_dir / "transcript.txt",
        analysis_html_relpath="a.html", stop_event=threading.Event(), viewer_data_path=out_dir / "viewer.json",
    )
    start = time.time()
    session.play()
    run = g.game_run
    return {"state": run.state, "note": run.solver_note, "actions": session.action_count,
            "seconds": round(time.time() - start, 1)}


def analyse(log_path: Path, result: dict, transcript: Path) -> dict:
    reqs = [json.loads(l) for l in log_path.read_text().splitlines()]
    total = reused = 0
    prev: list[str] = []
    for r in reqs:
        cur = r["messages"]
        n = 0
        while n < min(len(prev), len(cur)) and prev[n] == cur[n]:
            n += 1
        total += sum(len(m) for m in cur)
        reused += sum(len(m) for m in cur[:n])
        prev = cur
    # Did a summary survive at the head of the history after a trim?
    head_summaries = sum(
        1 for r in reqs if len(r["messages"]) > 2 and SUMMARY_MARKER in r["messages"][1]
    )
    text = transcript.read_text(errors="replace") if transcript.exists() else ""
    return {
        **result,
        "requests": len(reqs),
        "summary_requests": sum(r["summary"] for r in reqs),
        "requests_starting_with_a_summary": head_summaries,
        "prompt_chars_sent_k": round(total / 1000),
        "cache_reuse_pct": round(100 * reused / max(1, total), 1),
        "new_prefill_chars_k": round((total - reused) / 1000),
        "tracebacks_in_transcript": len(re.findall(r"Traceback", text)),
    }


# ---------------------------------------------------------------- driver

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--label", required=True)
    p.add_argument("--game", default="ls20")
    p.add_argument("--actions", type=int, default=60)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--outage", metavar="AFTER_REQ:SECONDS",
                   help="Take the fake server down after N requests for S seconds (server-crash drill).")
    p.add_argument("--child", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.child:
        out_dir, port = Path(args.child), int(os.environ["DEV_TEST_PORT"])
        print("RESULT " + json.dumps(run_game(args.game, args.actions, out_dir, port)))
        return

    out_dir = Path(tempfile.mkdtemp(prefix=f"devtest-{args.label}-"))
    FakeModel.log_path = out_dir / "requests.jsonl"
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeModel)
    port = server.server_port
    threading.Thread(target=server.serve_forever, daemon=True).start()
    servers = [server]
    if args.outage:
        after, seconds = (float(x) for x in args.outage.split(":"))

        def drill():
            while not FakeModel.log_path.exists() or len(FakeModel.log_path.read_text().splitlines()) < after:
                time.sleep(0.05)
            servers[0].shutdown(); servers[0].server_close()
            print(f"   [drill] fake server DOWN after {int(after)} requests, for {seconds:.0f}s", flush=True)
            time.sleep(seconds)
            servers[0] = ThreadingHTTPServer(("127.0.0.1", port), FakeModel)
            threading.Thread(target=servers[0].serve_forever, daemon=True).start()
            print("   [drill] fake server UP again", flush=True)

        threading.Thread(target=drill, daemon=True).start()

    env = dict(os.environ)
    env.update(notebook_settings())
    env.update({k: str(v) for k, v in SCALED.items()})
    env.update({"LOCAL_ANALYZER_BASE_URL": f"http://127.0.0.1:{port}/v1",
                "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
                "ARC3_HTTP_RETRY_INITIAL_SECONDS": "0", "ARC3_AUTO_PROBE": "off",
                "DEV_TEST_PORT": str(port)})
    for item in args.set:
        key, _, value = item.partition("=")
        env[key] = value
    proc = subprocess.run([sys.executable, __file__, "--label", args.label, "--game", args.game,
                           "--actions", str(args.actions), "--child", str(out_dir)],
                          env=env, capture_output=True, text=True)
    servers[0].shutdown()
    line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
    if line is None:
        print(proc.stdout[-3000:], proc.stderr[-6000:], sep="\n")
        raise SystemExit(f"[{args.label}] child run failed (exit {proc.returncode}); files in {out_dir}")
    warnings = [l for l in proc.stderr.splitlines() if re.search(r"Traceback|Error|WARNING", l)]
    report = analyse(FakeModel.log_path, json.loads(line[7:]), out_dir / "transcript.txt")
    report["stderr_warning_lines"] = len(warnings)
    print(f"[{args.label}] " + json.dumps(report))
    for w in warnings[:8]:
        print("   ", w[:200])
    print(f"   files: {out_dir}")


if __name__ == "__main__":
    main()
