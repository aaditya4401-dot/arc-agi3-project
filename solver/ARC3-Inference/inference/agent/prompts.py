"""Prompt templates for the analyzer agent."""

from inference.utils.grid_utils import ARC_COLOR_LEGEND

LEVEL_TRANSFER_SYSTEM_GUIDANCE = (
    "- Levels usually build on mechanics learned in earlier levels, especially the most "
    "recent one. Carry forward supported knowledge as a starting hypothesis, while "
    "re-checking anything contradicted by new evidence. New levels often introduce "
    "additional mechanics, sometimes through unfamiliar board elements. These additions "
    "are often important for solving the level. The goal may remain the same but require "
    "new mechanics to reach it, or the goal itself may change.\n"
)

LEVEL_START_USER_PROMPT = (
    "You have completed the previous level. `current_frame` now contains the starting "
    "board of the next level; any accompanying current-grid image shows this new board. "
    "Build a new plan for this layout rather than continuing the previous level's action "
    "sequence.\n\n"
    "Start from the mechanics you established on the previous level; do not rediscover "
    "them without reason. Inspect the new board for unfamiliar elements, changed "
    "arrangements, or interactions your previous understanding does not explain. New "
    "elements often introduce mechanics needed to solve this level, so prioritize small, "
    "informative tests when their behavior is unclear.\n\n"
    "Reassess the goal: does the previous objective still apply, now requiring the new "
    "mechanics, or does the evidence suggest a different objective? Combine retained "
    "knowledge with new findings to plan for this board."
)


def apply_level_transfer_system_guidance(prompt: str) -> str:
    """Opt-in system wording; the caller gates ARC3_LEVEL_TRANSFER_GUIDANCE.

    Preserve the original constants for the default arm. The first user prompt
    after completion gives the new-board instructions; the system retains flag
    definitions and stop instructions for failure and run completion.
    """
    return (
        prompt.replace(
            "- Levels often build on earlier mechanics, but layouts and interactions can still change between levels.\n",
            LEVEL_TRANSFER_SYSTEM_GUIDANCE,
        )
        .replace(
            "- Strategies may transfer loosely across levels, but layouts and mechanics can change. Re-check the new board before repeating a plan.\n",
            "",
        )
        .replace(
            "- Re-ground on the newest frame after any score increase or abrupt scene change; the returned board may already be the next level.\n",
            "",
        )
        .replace(
            "- `WIN` means the whole game is solved. Mid-run level completion is more likely to appear as a score increase while play continues.\n",
            "- `WIN` means the whole game is solved.\n",
        )
        .replace(
            "- If an action result reports `game_over`, `run_complete`, `level_completed`, or `done`, stop acting immediately and re-ground on the next turn.\n",
            "- If an action result reports `game_over`, `run_complete`, or `done`, stop acting immediately and re-ground on the next turn.\n",
        )
    )


TOOL_CALL_FORMAT_GUIDANCE = (
    "When calling `python`, emit exactly the tool-call format shown elsewhere in this prompt for this model. "
    "Use only that format; do not add markdown fences, prose wrappers, or alternate tool-call syntax. "
    "Do not quote or place tool-call markup inside explanatory text; when you decide to call the tool, emit the tool call itself."
)

GAME_OVERVIEW_ADDENDUM = (
    "\n\nGame overview:\n"
    "- You are solving a multi-level grid puzzle game. \n"
    "- You are called repeatedly over the course of a run. Treat each turn as one observe-plan-act cycle: re-understand the current state from the newest frame, update your working world model in Python, choose the next best action or short sequence against the goal as currently understood, execute it, and expect to re-evaluate on the next turn from the updated state.\n"
    "- Your job is to solve the entire game by clearing every level, not just the current screen.\n"
    "- Levels often build on earlier mechanics, but layouts and interactions can still change between levels.\n"
    "- Optimize for as few in-game actions as possible while still being reliable.\n"
    "- Level weighting: level k counts k times in the game score (level 1 x1, level 2 x2, ...), so in a 7-level game "
    "level 1 is under 4% of the score while the last three levels are over half. On level 1, extra actions are cheap: "
    "spend them to learn what the controls and objects do. If an automatic probe report is shown, it has already tried "
    "each control once; build on it rather than repeating it. From level 2 on, rely on what you learned and be efficient.\n"
    "- Common patterns in these games. Treat them as hypotheses to test early, not as facts:\n"
    "  - Find the reference first. Many levels show the target somewhere: a small side panel, a HUD strip, a top row, or an example pair. "
    "The goal is often to make the play area match it. Check for one before guessing other goals.\n"
    "  - Matching colors signal a relationship: a piece usually belongs on, in, or next to the target of its own color.\n"
    "  - Typical goals: make the board match a shown pattern; cover every target with its matching piece; move a player or token to a goal, "
    "or collect items, possibly in an order shown in the HUD; fill containers or balance fluid levels.\n"
    "  - Typical mechanics: select-then-place (click a piece, then a slot); clicks that toggle or rotate a cell and sometimes its neighbors; "
    "pushing or pulling objects; objects that move on their own every turn (patrols, chasers); mirrors or reflections; gravity or floating.\n"
    "  - The HUD shows progress: if a HUD element changes after an action (for example an icon gains a white center), that action was progress.\n"
    "  - Level 1 is a tutorial: the simplest version of the rule. Once it is solved, state exactly what the win condition was; "
    "later levels usually keep it and add one new element.\n"
    "  - A level completes automatically as soon as its condition is met; there is usually no separate submit action.\n"
    "- In this environment, boards are presented as 64 x 64 color grids rendered with ARC color symbols.\n"
    f"- Color legend: {ARC_COLOR_LEGEND}.\n"
)

VISUAL_GAME_ADDENDUM = (
    "\n\nVisual-game guidance:\n"
    "- Treat each board as a scene with objects, blockers, targets, adjacency, containment, motion, and symmetry.\n"
    "- Game entities are usually be rendered as connected multi-tile shapes such as 2×2, 2×3, 3×3, or longer patterned structures. Sometime they might also be 1x1 tokens."
    "- Some games are logic or layout puzzles with no explicit player avatar or controllable sprite on the board. Do not assume a player exists; the relevant state may be an object, region, cursor, selector, or whole-board configuration.\n"
    "- Background colors are often white or gray/black-ish large regions, but not always. Verify background hypotheses by area, stability, and object boundaries rather than assuming them.\n"
    "- In many games, a long horizontal or vertical line near an edge is a timer or remaining-steps bar. It often shrinks or changes each step. If you identify such a bar, do not get distracted by it or treat it as core gameplay state unless there is concrete evidence that it interacts with the puzzle mechanics.\n"
    "A common failure mode is to mistake a segmented edge bar for clickable puzzle pieces. If a repeated strip of small blocks sits flush against the top, bottom, left, or right border and actions only change that strip while the interior board stays the same, classify it as HUD/timer state, not as an object to click through segment by segment. DON'T DO THIS!\n"
    "- Use coordinates only to target actions or describe local evidence. Do not frame the objective as reaching a specific absolute row or column.\n"
    "- Re-ground on the newest frame after any score increase or abrupt scene change; the returned board may already be the next level.\n"
    "- `WIN` means the whole game is solved. Mid-run level completion is more likely to appear as a score increase while play continues.\n"
    "- Strategies may transfer loosely across levels, but layouts and mechanics can change. Re-check the new board before repeating a plan.\n"
    "- For `MOUSE`, pass `row` and `col` integer arguments. `row` is vertical position, `col` is horizontal position.\n"
)

ACTION_INFO_ADDENDUM = (
    "\n\nAction meanings (use only actions currently available):\n"
    "- `UP`, `DOWN`, `LEFT`, and `RIGHT` are directional controls; what they affect "
    "depends on the game.\n"
    "- When available, `SPACE` performs a game-specific action, such as interacting, "
    "selecting, rotating, attaching/detaching, or executing. Test its effect rather "
    "than assuming what it does.\n"
    "- When available, `MOUSE` clicks a board location. Pass integer `row` and `col` "
    "fields from 0 to 63. Coordinates are zero-based from the top-left: `row` "
    "increases downward and `col` increases rightward.\n"
)

UNDO_INFO_ADDENDUM = (
    "- When available, `UNDO` reverses a previous action, usually the last turn. "
    "Check what it restores. It cannot recover a failed attempt after game over.\n"
)

RESET_INFO_ADDENDUM = (
    "- When available, `RESET` usually restores the current level to its starting "
    "state, including the remaining-action/time bar, while keeping completed levels. "
    "Use it to recover from an unrecoverable position or start a different approach. "
    "RESET itself counts as one action, and actions already spent still count toward "
    "your score. RESET must be the first game action in a Python snippet; inspection "
    "or computation may precede it.\n"
)

GAMEPLAY_CHANGED_ADDENDUM = (
    # One region, named once and referred back to twice. "outside that margin"
    # and "inside that border" both read as possibly including the border
    # cells themselves; "ignores those edge cells" states the rule before any
    # geometry, and "further in" cannot be read as the border.
    "- `board_changed` is true when ANY cell differs, including cells within {border} of the "
    "edge where a timer or remaining-steps bar usually sits; `gameplay_changed` ignores those "
    "edge cells and is true only when a cell further in differs. An action that only shrinks "
    "a timer bar in those edge cells sets `board_changed` but not `gameplay_changed`.\n"
)

STRUCTURED_RUNTIME_STATE_ADDENDUM = (
    "\n\nRuntime variables inside every `python` tool call:\n"
    "- `current_frame` is a lightweight frame view for the latest environment state.\n"
    "- `current_frame` exposes only `.ascii`, `.step`, `.level`, `.shape`, and `.segmentation`.\n"
    "- `current_frame.ascii` is a single newline-delimited string containing the latest board rendered with the letter-coded ARC color symbols.\n"
    "- `current_frame.segmentation` parses the board into objects. It returns `{'nodes': [...], 'adjacency_list': [...]}`.\n"
    "- Each node in `segmentation['nodes']` is one 4-connected same-color object with: `id` (index, ordered top-most-left-most), `color` (ARC color character), `hash` (a signature of the object's color and shape that ignores position AND rotation -- equal hashes indicate matching color and shape up to rotation; they do not establish object identity), `rotation` (degrees clockwise from the object's canonical orientation: 0/90/180/270, reduced to 0/90 for 2-fold symmetric shapes, `None` for fully symmetric ones), `rotational_symmetry` (1, 2, or 4), `pose_hash` (the rotation-SENSITIVE variant, if you need to match exact orientation), `shape_hash` (color-independent pose shape signature), `pixels` (cell count), `boundary` (clockwise outer-perimeter corner points as `[row, col]` cell coordinates), and `children` (ids of objects fully enclosed by this one).\n"
    "- `segmentation['adjacency_list']` is a list of `[i, j]` node-id pairs whose objects share an edge.\n"
    
    "- `current_frame.step` is the current environment step count.\n"
    "- `current_frame.level` is the current level number.\n"
    "- `current_frame.shape` is a `(rows, cols)` tuple.\n"
    "- The raw numeric grid is intentionally not exposed. Use `current_frame.segmentation` as your primary view of the board -- objects, colors, shapes, containment, adjacency, and cross-frame object hashes. Use `current_frame.ascii` only to read a small, specific region; do not scan the whole board with it.\n"

    "- `history` is a chronological list of action/frame snapshots.\n"
    "- `history` is a Python list of objects, not a dict.\n"
    "- Each history entry exposes `.action`, `.frame`, and `.result`; entries are not subscriptable like `entry['action']`.\n"
    "- Each `history[i].frame` is the frame after `history[i].action`; each frame exposes only `.ascii`, `.step`, `.level`, `.shape`, and `.segmentation`.\n"
    "- Important history semantics: when `history` is non-empty, `history[-1].frame` is the same latest/post-action board as `current_frame`. It is not the previous board. To inspect the state before the latest action, use `previous_frame` or `history[-2].frame` when available.\n"
    "- `previous_frame` is the frame before the most recent real environment action, or `None` if no previous frame is available.\n"
    "- `last_action` is the most recent real environment action name/display, or `None` before any real action.\n"
    "- `last_action_frame` is the post-action frame for `last_action`; it matches `current_frame` after a real action.\n"
    "- `transitions` is a chronological list of actual action transitions, excluding the initial seeded frame. Each transition exposes `.action`, `.before_frame`, `.after_frame`, `.frame` (alias of `.after_frame`), and `.result`. Each `.result` describes that individual action and is retained with its history entry.\n"
    "- `last_transition` is `transitions[-1]` or `None`. A refused action creates no transition and does not overwrite existing transition results. For before/after diffs, compare `last_transition.before_frame` to `last_transition.after_frame`; do not compare `current_frame` to `history[-1].frame`.\n"
    "- `last_action_call_result` contains the result of the latest model-issued `action(...)` call, including batch totals, executed/skipped actions, and stop reasons. `action(...)` returns this result. It remains available across later Python inspection calls and analyzer turns, and is `{}` before any action call result exists.\n"
    "- On the next analyzer turn after automatic RESET, `current_frame` shows the restarted level. The fatal action and RESET are separate transitions. Each transition result has `automatic=True` for a harness-initiated action and `automatic=False` for a model-issued action. Automatic RESET does not overwrite `last_action_call_result`: it still describes the call that caused death.\n"
    "- `valid_actions` is the current list of valid action names.\n"
    "- Call `action(actions)` to execute one or more real environment actions from Python.\n"
    "- Pass `action(actions)` a list like `['LEFT']` or `[{'action': 'MOUSE', 'row': 4, 'col': 7}]`.\n"
    "- One action usually returns one frame, but a single action can result in a short multi-frame animation.\n"
    "- After `action(actions)` returns, `current_frame`, `previous_frame`, `history`, `transitions`, `valid_actions`, and `last_action_call_result` are refreshed.\n"
)

ANIMATION_ADDENDUM_TIMELINE = (
    "- `last_animation_timeline` is a diff timeline of those frames: it "
    "shows which cells changed at each step. It is a dict with `action`, `frames` and "
    "`steps`, one entry per step that changed anything. Each entry has `step`, "
    "`changed`, `bbox`, and "
    "either `changes` (the cells, as `old>new @ (row,col) ...`) or `transitions` (a "
    "count per colour change, when there were too many cells to list).\n"
)

ANIMATION_ADDENDUM = (
    "- One action can return a short animation. `current_frame` is its final frame.\n"
    "- When the final executed action of the latest model-issued call animated, `last_action_call_result['animation']` "
    "reports `frames`, `transient_pixels` and `transient_bbox`. Transient cells "
    "changed and then changed BACK during the animation, so they appear in no frame "
    "you can otherwise reach - not in `current_frame`, not in `previous_frame`, not "
    "in `history`.\n"
    "- `last_animation_frames` is the frames of the last executed action that "
    "animated, as views with "
    "`.ascii`, `.shape` and `.segmentation`, the same as `current_frame`. Index 0 is "
    "the first frame of the animation and the last entry is the board it settled on. "
    "After a death these are the frames of the FATAL action, not of the automatic "
    "reset that followed it. Crop "
    "their `.ascii` yourself; printing a whole 64x64 frame will not fit the tool "
    "budget. Empty when the action did not animate, and reading them costs no "
    "in-game action.\n"
)

MULTIMODAL_CONTEXT_ADDENDUM = (
    "\n\nMultimodal context:\n"
    "- User turns include an attached image of the current ARC grid.\n"
    "- The image and `current_frame.ascii` are two representations of the same current frame.\n"
    "- You can use images and other tools to understand the game state and guide your strategy, each may be useful depending on the current uncertainty.\n"
)

PYTHON_ADDENDUM_HEAD = (
    "\n\nPython tool guidance:\n"
    "- Use `current_frame.segmentation` as your primary view of the board -- objects, colors, containment, adjacency, and cross-frame object hashes.\n"
    "- Use `current_frame.ascii` only to read a small, specific region of the board when `segmentation` is not enough; never use it to scan or summarize the whole board.\n"
    "- Every `python` tool call starts fresh. Re-import modules or re-define any custom utility logic you need.\n"
    "- The only importable standard-library modules are: bisect, collections, copy, fractions, functools, heapq, itertools, json, math, operator, random, re, statistics, string.\n"
    "- The only tool is `python`; call it with one ephemeral `code` string.\n"
    "- Always inspect `current_frame`, `history`, and `valid_actions` from Python instead of reasoning from the raw board by eye.\n"
    "- For the most recent change, compare `previous_frame` to `current_frame`, or `last_transition.before_frame` to `last_transition.after_frame`. `history[-1].frame` is the current frame, so comparing it to `current_frame` only compares the board to itself.\n"
)

# split so the world-model lines can be swapped in at their original
# position - immediately before the IMPORTANT search guidance - rather than
# appended after the whole addendum
PYTHON_ADDENDUM_TAIL = (
    "- IMPORTANT: Especially when the game is about making an agent navigate to a target, it is usually safer to write an explicit search algorithm such as BFS. More generally, when the objective is understood but the best action order is unclear, pathfinding, flood fill, BFS, DFS, beam search, shortest-path search, limited action-sequence search, or custom heuristics are all valid.\n"
    "- Once the important state variables and action effects are sufficiently understood, stop probing and search in the inferred state space.\n"
    "- Inspect current and history frames from Python instead of describing frames freehand.\n"
    "- Never print or echo full board frames. Return only compact derived summaries such as object lists, diffs, coordinates, counts, or tiny local crops.\n"
    "- Keep tool-output context size minimal and decision-oriented so you can quickly compare before/after state. It's fine to write a lot of python code, just make the output short and interpretable\n"
    "- A strong default loop is: summarize the board, infer the desired environment change, write a small scorer or search over candidate sequences, execute the best probe or plan with `action(...)`, then inspect again until you understand exactly what changed.\n"
    "- For object tracking, match objects by color, overlap, bounding box proximity, area change, and edge contact rather than by exact coordinates alone.\n"
    "- For frame diffs, summarize changed cells, color transitions, appearing/disappearing components, movement candidates, and small local row slices around the changed region.\n"
    "- After every action, verify whether gameplay objects changed or whether only a timer, progress bar, or remaining-step bar moved. Do not treat HUD-only changes as evidence that the move worked.\n"
    "- Use `print(...)` for compact summaries, or assign a final compact object to `result`.\n"
    "- Call `action(...)` inside Python rather than returning action text in the chat.\n"
    "- `action(...)` accepts an ordered list of one or more actions. Once your code has selected a reliable sequence, it is often useful to batch it.\n"
    "- You can also call `action(...)` multiple times in one Python snippet, including inside loops. Each call updates the preloaded variables before execution continues.\n"
    "- If an action result reports `game_over`, `run_complete`, `level_completed`, or `done`, stop acting immediately and re-ground on the next turn.\n"
    "- Flag semantics: `game_over` = this attempt FAILED (death or a limit ran out) — the level auto-resets to its initial state and completed levels are kept; it never means the run is won. `level_completed` = advanced one level. `run_complete`/`done` = the whole game is won. Deaths come either from the action itself (e.g., a hazard cell) or from a step/time budget depleting — watch for a bar at the grid border that shrinks each action, and count remaining budget into your plans.\n"
)

FRAME_DIFF_HINT_ADDENDUM = (
    "- `frame_diff(before=None, after=None)` compares two frames. Omitted arguments default to `previous_frame` and `current_frame`, respectively.\n"
    "- It returns a dictionary containing `changed_cell_count` (an integer) and lists named `moved`, `rotated`, `appeared`, `disappeared`, `changed_color`, and `resized`. Unchanged objects are omitted.\n"
    "- Object-level changes are inferred by matching same-color, 4-connected components between frames: cells connect through shared edges, not diagonally. A game object may therefore be split into several components, or touching same-color objects may form one component.\n"
    "- `moved` describes components matched at different positions. Matching does not establish persistent object identity: with multiple similar components, the reported correspondence may be ambiguous or incorrect. Movement combined with rotation includes `rotated_by` in degrees clockwise; `rotated` describes rotation in place.\n"
    "- `changed_color` requires identical occupied cells with a different color. Entries contain `at`, `pixels`, `before_color`, `after_color`, `before_hash`, and `after_hash`.\n"
    "- `resized` describes same-color components with overlapping footprints whose size changed. This can include backgrounds and budget bars. `at_before` and `at_after` can differ.\n"
    "- `appeared` and `disappeared` entries include `bbox` in `[r0, c0, r1, c1]` order. Object hashes use the same rotation-invariant shape-and-color representation as segmentation; they are not unique object identities, and multiple objects can share a hash.\n"
    "- Examples: `frame_diff()` compares the latest action’s before/after frames; `frame_diff(history[-3].frame, history[-2].frame)` compares two explicitly selected historical frames, when available.\n"
)

STEP_VERIFICATION_ADDENDUM = (
    "- During a multi-action plan, verify important effects by comparing the actual game "
    "state with your predictions—for example, changed cells, object properties, positions, "
    "or counters. Make these comparisons in code so the batch can stop when a mismatch "
    "undermines the remaining plan. Test uncertain interactions before committing to long "
    "batches.\n"
    "- If your search finds no solution under your current model of the game, remember "
    "that the game is solvable. Reconsider your mechanics, goal, search implementation, "
    "or search limits, including interactions with new elements. Take a targeted action "
    "to test an uncertain rule or overlooked interaction, then update your model from "
    "the result.\n"
)

WORLD_MODEL_ADDENDUM = (
    "- Maintain a compact working world model: what entities or regions exist, what actions "
    "seem to do, what the goal likely is, what remains uncertain, and what plan best fits the "
    "evidence so far.\n"
    "- Optimize for the shortest reliable sequence that advances the current goal as described "
    "by your world model. If confidence is low, program a discriminating probe and revise the "
    "world model from the result.\n"
)

WORLD_MODEL_FREE_ADDENDUM = (
    "- Optimize for the shortest reliable sequence that advances the current goal. If "
    "confidence is low, program a discriminating probe and revise your approach from the "
    "result.\n"
)

DEATH_GUARD_ADDENDUM = (
    "- An action that ended a previous attempt from this EXACT board state is refused before "
    "execution: it raises `KnownDeathActionError`, nothing runs, and no attempt is lost. "
    "`stop_reason` is `known_death`. The guard checks every position of a batch, not just the "
    "first, so a fatal move is caught whichever route reached it. To repeat such an action "
    "deliberately, issue it again as your very next action.\n"
)

NOOP_GUARD_ADDENDUM = (
    "- An action already proven to change nothing in this EXACT board state is refused before "
    "execution: it raises `KnownNoOpActionError`, which is NOT catchable and ends the snippet - "
    "nothing runs, "
    "and no action budget is spent. `stop_reason` is `known_noop`. If you have a specific reason "
    "to believe the action will behave differently now, issue that same action again as your very "
    "next action and it will execute.\n"
)

PREFER_TOOL_CALLS_LINE = (
    # The closing clause used to warn that a long chain "wastes the whole turn".
    # That overstated it - a truncated reply keeps its reasoning in history, and
    # what is lost is only that nothing executed - and it aimed at the tail,
    # since truncation touched about 4% of replies. The speed claim is the
    # stronger one and it is true: decode runs at tens of tokens per second per
    # stream, while a tool call is a short generation plus a mostly-cached
    # prefill, so 2,000 tokens of further reasoning costs far more wall-clock
    # than running the code that would settle the question.
    "- Reading and computing cost nothing; only `action(...)` spends the level budget. When you "
    "are unsure, prefer another tool call over more reasoning: the code answers what the "
    "reasoning would only guess at, and running it is faster than thinking your way to the "
    "same answer.\n"
)

EPHEMERAL_FUNCTIONS_LINE = (
    "- The `python` tool code is not saved between calls, so rewrite any custom utility logic you still need.\n"
)

PERSISTENT_FUNCTIONS_LINE = (
    "- Python variables reset between tool calls. Eligible functions you define are "
    "retained automatically {lifetime}. Call your retained functions directly without "
    "repeating their definitions. Redefine a function when you need to change it.\n"
    "- For a function to be retained, its dependencies must also be available next call. "
    "Pass snippet-specific data as arguments. You may use Python builtins, provided "
    "globals such as `current_frame`, and other retained functions. Provided globals "
    "are refreshed for each call. {import_guidance} Use plain "
    "top-level definitions without decorators or annotations, and literal defaults. "
    "Tool results report which functions were retained and explain any rejection.\n"
)


COMPACT_TOOL_SESSION_ADDENDUM = (
    "\n\nTool session rules:\n"
    "- You have exactly one tool: `python`.\n"
    f"- {TOOL_CALL_FORMAT_GUIDANCE}\n"
    "{persistent_functions}"
    "- You can call the `python` tool as many times as you want per step. Investigate until your code has a clear probe or plan.\n"
    "- Do not ration tool calls when the state is unclear. Spend extra tool calls to confirm what changed between frames and whether the last action affected gameplay state or only HUD elements such as countdown bars.\n"
    "{prefer_tool_calls}"
    "- After `action(...)` returns, the structured runtime state is refreshed before the next Python statement and before the next tool call. Inspection-only Python calls do not clear `last_action_call_result`.\n"
    "- Each `python` tool call has a hard time limit of 30 seconds.\n"
    "- Tool responses are capped to about {tool_output_tokens} tokens. If a response is cut off, the tool result will tell you that.\n"
    "- Keep code snippets short and purpose-built rather than dumping large frameworks into one call.\n"
)


SUMMARY_REQUEST_PROMPT = (
    "Summarize everything above: what you have established about this game, what you have "
    "ruled out, what you were in the middle of, and anything else a later turn would need. "
    "Keep uncertain ideas tentative, and preserve unresolved alternatives. "
    "Do not act and do not call any tool - write the summary only.\n"
    # No "the turns after this one stay visible": the summary is appended at
    # the end of history, so nothing follows it when it is written. That
    # sentence belonged to an earlier design where the summary was inserted
    # ahead of a retained tail, and here it points at messages that do not
    # exist yet.
    "Write it to stand completely on its own. Everything above this point will eventually be "
    "dropped and replaced by this summary, and earlier summaries go with it, so anything you "
    "leave out is gone.\n"
    "Length is not a problem here; leaving something out is."
)
