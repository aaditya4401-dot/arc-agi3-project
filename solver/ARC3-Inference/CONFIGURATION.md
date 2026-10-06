# Duck harness: cumulative change inventory

This inventory describes the current harness changes relative to **Tufa Labs’ Duck harness, commit `7652836056c59e044f093e3c13ed7438c814169e` (`7652836`)**. Full credit belongs to Jeroen Cottaar and Tufa Labs for the original solver and harness.

**Scope:** the current cumulative source changes in `ARC3-Inference`, including six new production modules/data files and changes to thirteen existing production, build, and viewer files. README edits, analysis scripts, notebooks, model weights, and SGLang server patches are outside this inventory. Audited on 29 September 2026.

This is an implementation inventory, not a claim that every option improved score or was enabled in the final submission. It includes optional experiments still present in the code. Defaults below are the effective harness defaults when the environment variable is absent; a notebook can override them. Set configuration before importing/starting the harness, since some settings are read at import time.

## 1. Action exposure and action explanations

### Undo

Added a complete model-facing route from `UNDO` to engine `ACTION7`, covering action lists, validation, execution, and prompts. The model can recover from mistakes in games that provide this action.

| Setting | Default | Behavior |
|---|---|---|
| `EXPOSE_UNDO` | `tufa` | `tufa` preserves the original inconsistent wiring: ACTION7 can be listed but cannot be executed. `off` hides ACTION7/UNDO and rejects them. `on` lists UNDO and accepts either UNDO or ACTION7 for execution. |

This setting is an enum: **use `on`, not `1`**. Invalid values fall back to `tufa`. Exposure does not make undo available in a game/state that does not support it.

### Model-initiated reset

Added explicit RESET advertising, guidance, and snippet-level restrictions. The intended use is recovery from an unrecoverable position without spending actions merely to exhaust the bar.

| Setting | Default | Behavior |
|---|---|---|
| `EXPOSE_RESET` | `off` | Enables explicit RESET presentation and the first-action restriction. Accepts `on`, `1`, `true`, or `yes`. Off preserves legacy RESET handling; it is not a blanket ban on every legacy RESET path. |

When exposed, RESET must be the first action of the Python snippet. A batch that reaches RESET after another action is stopped before executing it. Subsequent actions after a successful initial RESET remain subject to the normal guards. RESET counts as an action. Guidance says it usually restores the current level, including its bar, to the starting state.

A model-requested reset receives reset-specific follow-up information rather than being represented as a death. The previous action’s animation is preserved across reset, including automatic reset after death. This lets the model inspect what happened before recovery.

### Action information in the system prompt

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_ACTION_INFO` | Off | Adds compact descriptions of directional controls, SPACE, and MOUSE; includes UNDO when undo exposure is on and RESET when reset exposure is on. |

The descriptions clarify that action semantics depend on the game. MOUSE uses integer `row` and `col` fields: rows increase downward, columns increase rightward, measured from the top-left. The normal model interface rejects legacy `x`/`y` fields. The explanation is in the system prompt rather than repeatedly appended to user prompts.

## 2. Prompt changes and optional behavioral guidance

| Setting | Default | Change |
|---|---|---|
| `ARC3_LEVEL_TRANSFER_GUIDANCE` | Off | Revises system and level-up user guidance: carry forward useful mechanics from the preceding level; inspect new elements and test their role; expect new mechanics; reconsider how the goal must be reached. The level-up message identifies the returned frame/image as the next level’s starting state. |
| `ARC3_FRAME_DIFF_HINT` | Off | Explains how to call `frame_diff`, its defaults and returned fields. Describes same-color four-connected components and warns that matching is heuristic, so a reported movement does not establish persistent object identity. It is informational rather than a requirement to use the helper. |
| `ARC3_STEP_VERIFICATION_HINT` | Off | Replaces the player-specific verification hint with general guidance to compare observations with expected effects during multi-step code. If search fails, the game is still solvable; reconsider mechanics, goal, search implementation, search limits, and interactions with new elements. |
| `ARC3_NEW_CHANGED_PROMPTS` | Off | Uses more precise language about no change in the board’s gameplay area, rather than equating every HUD-only change with useful progress. Also changes related guard feedback. |
| `ARC3_REPORT_GAMEPLAY_CHANGED` | Off | Exposes `gameplay_changed` and `no_op` in model-facing action results. The harness can calculate these internally even when reporting is off. |
| `ARC3_EXPLAIN_GAMEPLAY_CHANGED` | Inherits `ARC3_REPORT_GAMEPLAY_CHANGED` | Adds the explanation distinguishing full-board changes from changes inside the gameplay area. Can be explicitly enabled independently. |
| `ARC3_OPENER_COMMIT_HINT` | Off | Adds encouragement to commit to an action sequence when the mechanics are sufficiently understood. |
| `ARC3_PREFER_TOOL_CALLS` | Off | Adds guidance about using Python inspection/computation rather than spending excessive deliberation without testing. |
| `ARC3_DEDUPE_MULTICALL_LINE` | Off | Removes duplicated prompt wording about multiple action calls. This is text deduplication, not deduplication of tool execution. |
| `ARC3_SYSTEM_PROMPT_PREFIX` | Empty | Prepends a custom system-prompt prefix for experiments. |

With level-transfer guidance enabled, the old advice about re-grounding after a score increase is replaced: the model is explicitly informed of level transitions rather than being asked to infer them from score. Transition/terminal behavior is explained alongside the corresponding runtime information.

Several consistency fixes are unconditional: documentation now includes frame `.shape`; hashes are not described as unique object identities; and normal prompt guidance describes automatic reset after death as occurring before the next analyzer turn.

## 3. Segmentation and structured frame differences

### Richer shape metadata — unconditional

Extended the existing segmentation representation with rotation-aware shape information. Components expose canonical hashes and orientation-related metadata, including `shape_hash`, `pose_hash`, `rotation`, and `rotational_symmetry`. Canonicalization accounts for rotations, and color-independent orientation is calculated consistently across differently colored copies of a shape.

Segmentation still uses same-color **four-connectivity**. Diagonal contacts, multicolor objects, and touching objects can therefore produce components that differ from the game’s conceptual objects. Hashes describe shapes/appearances; they are not unique object IDs.

### `frame_diff` — unconditional Python helper

Added `frame_diff(before=None, after=None)` to the sandbox. Its defaults compare `previous_frame` with `current_frame`. It returns a compact structured account of changed cells and candidate object changes: movement, rotation, appearance, disappearance, recoloring, and resizing. Different-sized frames return an explicit full-redraw/shape-mismatch result.

Matching uses component information and assignment heuristics. Multiple similar objects, merges, splits, and segmentation changes can make the correspondence ambiguous. Automatic prompt rendering bounds expensive matching for large groups of identical components; direct Python inspection remains available independently of the automatic hint.

| Setting | Default | Change |
|---|---|---|
| `ARC3_AUTO_FRAME_DIFF` | Off | Inserts a compact structured/textual frame-difference summary in the user prompt. Separate from availability of the Python helper. |
| `ARC3_AUTO_FRAME_DIFF_BUDGET` | `300` | Approximate token budget for the automatic text diff; rendering falls back from individual changes to grouped changes and then counts. |
| `ARC3_LEVEL_INVENTORY` | Off | Compares object inventories at the starts of successive levels, highlighting newly appearing types. Independent of ordinary per-action diffs and diff images. |
| `ARC3_GAMEOVER_DIFF` | Off | Adds a textual comparison involving the fatal frame after game over. |

## 4. Images and animations

### Diff images

Added images that show changed cells in their true palette colors while dimming unchanged cells to an off-palette dark navy. This avoids confusing the background marker with a real board color. Invalid comparisons, including incompatible frame sizes, do not produce a misleading diff image.

| Setting | Default | Change |
|---|---|---|
| `ARC3_DIFF_IMAGE` | Off | Attaches ordinary board-difference images. |
| `ARC3_GAMEOVER_DIFF_IMAGE` | Off | Attaches death-related comparison images in the first post-death opener. |
| `ARC3_RESUME_DIFF_IMAGE` | Off | Reattaches the current turn’s diff image to a resumed turn, useful if earlier images have been hidden. |
| `ARC3_RESUME_PROMPT_IMAGES` | On | Allows images on yield/resumption messages. Turning it off removes those repeated attachments. |
| `MULTIMODAL_RESAMPLE` | `nearest` | Selects the image resize filter: nearest, bilinear, bicubic, or lanczos. |
| `ARC3_HISTORY_IMAGE_KEEP` | Unset: unlimited | Optionally retains images only in the latest N image-bearing messages sent to the model. Zero strips all such images. This filters outgoing context and its estimate rather than deleting stored history. |

The original `MULTIMODAL_CONTEXT=current_grid` and `MULTIMODAL_UPSCALE` settings remain the basic image controls. The latter’s source default is 16; running at 10× is a notebook configuration choice, not a new fixed default. Filtering historical images changes request prefixes and can affect prefix reuse.

### Intermediate animation frames

Added capture of intermediate frames returned by an action, consecutive-frame deduplication, transient-pixel detection, and compact animation metadata. This recovers information that disappears before the final board is returned.

| Setting | Default | Change |
|---|---|---|
| `ARC3_ANIMATION` | Off | Enables animation capture, summaries, and model inspection support. |
| `ARC3_ANIMATION_TIMELINE` | On, under animation support | Provides the compressed timeline accessible through Python. Does not automatically paste the full timeline into the user prompt. |
| `ARC3_ANIMATION_IMAGE` | `off` | `peak` shows one real frame with many transient cells; `composite` combines transient information across time. Ordinary animation imagery is attached for transient activity when the gameplay area did not change. |
| `ARC3_ANIMATION_DEATH_IMAGE` | `off` | Separate `peak`/`composite` option for the fatal action’s animation. |

The image-mode variables are enums, not booleans: an invalid value such as `1` raises a configuration error. Image attachments require the multimodal image path to be enabled.

The model can inspect `last_animation_frames` and `last_animation_timeline`. These are loaded lazily through a read-only environment query, without spending a game action. The prompt points out transient activity so that the model knows intermediate information is available.

A composite is explicitly not a real single board state. It retains the last non-final transient colors, with brightness encoding duration. A peak image is an actual intermediate frame, which avoids combining moments but may omit important parts of a longer animation.

Animation correctness fixes preserve fatal-action information through automatic RESET, preserve preceding animation through explicit RESET, and invalidate cached Python views after actions, including aliases and guard refusals. A subsequent ordinary action updates or clears the animation record instead of leaving an unrelated old animation visible.

## 5. Action guards and failure prevention

### Optional guards

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_BATCH_NOOP_BLOCK` | Off | Stops the remaining actions of a batch after an executed action fails to change the interior gameplay area. |
| `ARC3_STALE_STATE_BLOCK` | Off | Blocks a later `action(...)` call within the same snippet when a preceding action was ineffective, preventing continued execution from an unchecked assumption. |
| `ARC3_NOOP_REPEAT_GUARD` | Off | Recognizes previously ineffective state/action pairs and refuses a repeat. A deliberate immediate reissue can override the refusal. |
| `ARC3_DEATH_REPEAT_GUARD` | Off | Recognizes recorded fatal state/action continuations and refuses a repeat, with a deliberate reissue override. |
| `ARC3_DEATH_GUARD_LOOKAHEAD` | Off | Extends fatal-continuation checking to a proposed batch before earlier actions in that batch are spent. Used with the death guard. |
| `ARC3_REPEAT_STATE_GUARD` | Off | Stops an action repeated from the same state within one snippet, catching cycles caused by faulty loops. A new snippet starts a fresh check. |
| `ARC3_DEATH_LEDGER` | Off | Adds advisory information about previously fatal routes and exposes the structured ledger for inspection. |
| `ARC3_GUARDS_FROM_LEVEL` | `1` | Starts the per-action stale/repeat/death guards at this level. Does not defer the batch no-op guard or terminal latch. |
| `ARC3_NOOP_GUARD_BORDER` | `4` | Excludes the outer border when deciding whether gameplay changed, so HUD/bar changes do not count as progress. Zero disables the border-dependent batch no-op behavior. |
| `ARC3_STATE_IDENTITY_BORDER` | `0` | Separately controls the border ignored when identifying states for repeat/fatal checks. |

Death recording is enabled whenever either the ledger display or the death guard is enabled. The guard works independently of the advisory display.

Mouse action signatures translate engine coordinates into the same row/column convention used by guard comparisons, fixing missed matches between equivalent clicks.

### Unconditional terminal and result handling

A terminal latch prevents further actions within the same snippet after a death, level completion, or finished run. This protects against executing a plan on a replaced board. Guard refusals return clear stop metadata, including what actually executed and what was skipped; a refusal with zero executed actions does not invent a transition.

Known-no-op, known-death, repeated-state, and terminal-stop exceptions derive from `BaseException`, so a model-written broad `except Exception` cannot silently swallow them and continue a broken loop.

### Optional output guard

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_BOARD_DUMP_GUARD` | Off | Compresses long full-width ASCII board dumps in tool output and adds a reminder to print useful computed facts or small regions. Short crops remain available. |

## 6. More faithful Python state and action results

These changes are unconditional unless a setting is shown.

- **Per-action history metadata:** executed actions retain their individual compact result metadata in history and `transitions[i].result`, rather than assigning the batch result only to the final transition and leaving older transitions empty.
- **Separate batch result:** the global **`last_action_call_result` replaces `last_action_result`**. It describes the most recent model `action(...)` call, including a batch or refusal. This is an intentional model-facing API rename, without an old-name compatibility alias.
- **Stable inspection:** `last_action_call_result` survives intervening inspection-only snippets and analyzer turns. It is empty before the first applicable call.
- **Correct death/reset distinction:** automatic RESET has its own transition metadata marked automatic. It does not overwrite the model’s fatal action-call result with a successful reset result.
- **Correct before/after access:** prompts and tool metadata explain that `history[-1].frame` is current state. Use `previous_frame` or the transition’s before/after frames for differences.
- **Compact results:** retained per-action metadata excludes raw boards and full animation chains.
- **REPL-style last expressions:** the final expression of a snippet is evaluated and printed when non-None; it can also populate the result when no explicit result was supplied.
- **Sandbox compatibility:** added `difflib` to the allowed standard-library modules and more ordinary exception types to the available builtins, avoiding unnecessary snippet failures on harmless imports or exception handlers.

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_ACTION_ECHO` | Off | Prints a compact account of action calls/results into tool output. Avoids returning the same echo again as duplicated result data. |
| `ARC3_MIDDLE_TRUNCATION` | Off | When tool output is too long, retains its beginning and ending around a visible omission marker, helping preserve exceptions or conclusions at the end. |

The original `LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS` controls output length. Increasing that budget is a configuration choice; middle truncation changes which parts survive. Output truncation uses a stable character-based approximation rather than fluctuating with the context estimator’s learned token ratio.

## 7. Retaining model-written functions

Added optional retention of eligible Python function definitions between otherwise ephemeral tool processes. This is source-based function restoration into fresh runtime globals, not a persistent Python kernel: ordinary variables, mutable simulation state, and arbitrary objects are not automatically carried over.

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_PERSISTENT_FUNCTIONS` | Off | Enables function collection, validation, restoration, and feedback. |
| `ARC3_PERSISTENT_FUNCTIONS_SCOPE` | `levelup` | `turn`: retain within an analyzer turn; `levelup`: retain until a level change; `game`: retain throughout the game. |
| `ARC3_PERSISTENT_FUNCTIONS_IMPORTS` | Off | Also retains supported explicit import dependencies needed by retained definitions. |
| `ARC3_PERSISTENT_FUNCTIONS_REPAIR_HINTS` | Off | Enables extended instructions for repairing rejected function retention. |

The retention mechanism checks top-level definitions and their dependencies, including dependencies on other retained functions. Unsupported definitions or unavailable globals are rejected rather than silently restored in an invalid form. Imports inside a function remain a way to make the function self-contained without enabling import retention.

Prompt wording explicitly calls these functions **the model’s own retained definitions**, avoiding the impression that they are built-in harness helpers. With game scope, guidance reminds the model to recheck level-specific assumptions. When retention is disabled, the ordinary ephemeral-session instructions remain in effect.

Robustness fixes retain valid functions even when a snippet ends with a stale-state or other handled guard exception, rather than treating every such interruption as loss of definitions. Unexpected retention bookkeeping errors degrade to dropping functions and giving feedback; they must not replace the original output/error, replay actions, or halt the game.

Feedback deduplicates rejection notices within a result and avoids repeatedly reporting unchanged successful definitions as new. Import-retention instructions are gated with the import feature. Extended repair instructions are separately gated; the short baseline advice about passing snippet-specific data as arguments or redefining it remains.

## 8. Context estimation, rolling history, and pruning

### More realistic context estimates

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_TEXT_TOKEN_CHARS` | `3` | Initial/fallback characters-per-token estimate. |
| `ARC3_CALIBRATE_TEXT_TOKENS` | On | Calibrates the text estimate from backend prompt-usage reports, accounting separately for images. |
| `ARC3_TEXT_TOKEN_CHARS_MIN` | `1.0` | Lower bound on calibrated characters per token. |
| `ARC3_TEXT_TOKEN_CHARS_MAX` | `3.3` | Upper bound on calibrated characters per token. |
| `ARC3_IMAGE_TOKEN_ESTIMATE` | On | Estimates image cost from decoded PNG dimensions rather than charging for the base64 string as text. |
| `ARC3_IMAGE_TOKENS_FLAT` | `0` | A positive value overrides dimension-based estimation with a fixed cost per image. |

The dimension estimate uses 32-pixel patches plus two image marker tokens. For example, a 256×256 image estimates 64 patch tokens plus two markers. It is an estimator, not a guarantee that every backend/vision encoder uses exactly this tokenization.

### Hysteresis and configurable history limits

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_CONTEXT_DRAIN_TOKENS` | `0` | After reaching the context limit, removes additional old history in blocks, leaving headroom and reducing the frequency of prefix-changing trims. |
| `ARC3_CONTEXT_DRAIN_FLOOR` | Off | Stops optional extra draining before crossing its target when the mandatory context limit has already been satisfied. |
| `ARC3_HISTORY_ASSISTANT_TURNS` | `30` | Configures the separate assistant-message history cap. Zero or negative disables it. Counts assistant messages, not game actions or whole analyzer turns. |
| `ARC3_HISTORY_TURN_DRAIN` | `0` | Adds hysteresis to the assistant-message cap. |
| `ARC3_HISTORY_DRAIN_COALESCE` | Off | Coordinates token- and message-count drains to avoid closely spaced independent trims. |

The original `LOCAL_ANALYZER_CONTEXT_WINDOW` and `LOCAL_ANALYZER_MAX_OUTPUT` still determine the ordinary prompt budget, reserving generation space plus a safety margin. A large context setting does not override a smaller assistant-message cap or a smaller server context limit.

### Optional control-message pruning

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_PRUNE_CONTROL_CONTEXT` | Off | Master switch for pruning supported old harness-control messages. |
| `ARC3_PRUNE_NUDGES` | Off | Independently prunes obsolete nudge messages. |
| `ARC3_PRUNE_RESUME_PROMPTS` | Off | Independently prunes old resumption messages. |
| `ARC3_PRUNE_DEAD_STUBS` | Off | Independently prunes obsolete dead-reasoning placeholders. |
| `ARC3_STUB_DEAD_REASONING` | Off | Replaces unproductive reasoning-only replies with compact placeholders in retained context. |

Internal tags distinguish control messages without leaking those tags into requests. Pruning repairs the trailing user-message structure and records that the context has changed, so scheduling/prefix-state bookkeeping remains consistent.

## 9. Rolling summaries

Added periodic model-written summaries as a separate memory mechanism from the structured world-model sections.

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_SUMMARY_INTERVAL_TOKENS` | `0`: disabled | Requests another summary after this much newly appended context has accumulated since the preceding attempt. |
| `ARC3_SUMMARY_REPLACES_HISTORY` | Off | Off appends summaries in the stream; on replaces older history after a successful summary while retaining the summary context. |
| `ARC3_DRAIN_STOP_AT_SUMMARIES` | Off | Allows draining to stop at a useful summary boundary once the mandatory context constraint is met. |
| `ARC3_HIDE_FOLLOWUP_SUMMARIES` | Off | Filters completed summary exchanges from outgoing context according to eviction state, without deleting stored history. |
| `ARC3_SUMMARY_TURN_CONTEXT` | Off | Adds recent action/current-state information to the summary-writing request. |
| `ARC3_SUMMARY_ENABLE_THINKING` | Off | Controls thinking for the summary request separately from ordinary gameplay requests. |
| `ARC3_SUMMARY_MAX_GEN` | Normal positive output cap, otherwise `8192` | Bounds summary generation even when normal generation is configured as unlimited. |

### Summary correctness fixes

- The interval measures cumulative **new context**, including generated text, inserted prompts, tool outputs, and estimated image costs. It is neither generated-token-only accounting nor the size of the surviving rolling window. Re-sent history, static system/tool-schema text, and the summary exchange itself do not advance this interval.
- Every summary attempt resets the interval baseline, successful or not. A failed summary can be skipped and tried again after another interval; trimming cannot leave an old absolute threshold permanently unreachable.
- The visibility filter preserves a live request to write a summary. It only hides completed summary exchanges, so it cannot silently remove the request before the model answers it.
- Successful history replacement sets both the context-trim flag and the history-eviction flag. Scheduling, visibility filtering, and memory-rebuild behavior therefore see the same event.
- Summary calls receive the available timeout. Empty/failed responses are skipped; useful truncated summaries are marked incomplete. Tool calls produced during summary writing are not executed.
- Summary wording retains useful current facts and uncertainty/alternative hypotheses rather than demanding a complicated new memory taxonomy.

Summary output length is a separate request limit; it does not change the normal gameplay generation reservation. Normal context-overflow recovery trims and retries ordinary requests. A failed summary request can instead be skipped until its next interval.

## 10. Structured memory / world-model options

The original structured memory mechanism remains, with configurable subsets, lifecycle behavior, and more robust extraction.

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_MEMORY_SECTIONS` | All sections | Selects the carried memory sections. `slim` keeps world model and plan; `medium` also keeps the action model; `off` disables this structured-memory path. |
| `ARC3_WM_SECTION_EXPLANATION` | `tufa` | `full` expands the section descriptions, update semantics, and examples; default keeps the original style. |
| `ARC3_WM_WIPE_ON_LEVEL` | `1` | Clears appropriate memory at level changes; can be disabled to retain it. |
| `ARC3_WM_WIPE_ON_GAME_OVER` | `1` | Clears memory after death; can be disabled to preserve learned mechanics. |
| `ARC3_WM_REVISE_NUDGES` | Off | Requests revision when memory is retained across a level transition or death. |
| `ARC3_WM_REBUILD_NUDGE` | Off | Requests rebuilding memory after it has been cleared. |
| `ARC3_WM_NUDGE_TURNS` | `0`: disabled | Nudges after this many analysis steps without a memory update. |
| `ARC3_TOLERANT_SECTION_HEADERS` | Off | Recognizes decorated or slightly rephrased headers, including Markdown, “revised,” “updated,” and short annotations. |
| `ARC3_ALIAS_RESPECTS_STORED` | On | Prevents an alias-derived section from unintentionally replacing an existing canonical section. |
| `ARC3_MEMORY_SECTION_MAX_CHARS` | `8000` | Caps individual stored sections; zero disables the cap. |
| `ARC3_DEGENERACY_RATIO` | `0.01` | Detects extremely repetitive text by compression ratio; zero disables the check. |
| `ARC3_DEGENERACY_MIN_CHARS` | `500` | Minimum text length for the repetition check. |

Section updates accompanying tool calls are processed rather than silently lost. Oversized or degenerate memory writes cannot repeatedly inject an enormous corrupted block into later prompts. Degenerate assistant output also has a recovery path instead of being treated as useful context.

Disabling structured memory removes its storage/parsing and most associated prompt scaffolding; this is separate from rolling summaries and retained Python functions. It is not a guarantee that no legacy prompt sentence ever uses the words “world model.”

## 11. Priority scheduling and compute allocation

### Admission control and cache-aware scheduling

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_MAX_ACTIVE_STREAMS` | `0`: disabled | Limits concurrently admitted game streams through a shared priority gate. |
| `ARC3_PRIORITY_REFRESH_QUEUE` | Off | Recomputes waiting priorities against a common current clock before admission. Pace or tail fade also activates refreshed ranking. |
| `ARC3_DIAG_CONCURRENCY` | Off | Prints periodic concurrency/gate diagnostics. |

The gate can admit a new game when another finishes, rather than leaving backend capacity idle. It preserves an admitted stream across tool calls and resumptions and can hand over when context trimming makes a new prefill necessary. This requires the runner to have enough game sessions available; it does not itself replace every outer worker/concurrency limit.

Priority bands protect prefix reuse. Already warm continuation requests receive a 1,000,000 band above requests needing prefill, and the untrimmed continuation path uses its higher 2,000,000-based ordering. Score-based priority operates within that cache-aware policy. Request priority is also sent to compatible backends; backend support and the local admission gate are distinct mechanisms.

### Base scoring model

The normal score term estimates the value of finishing the current level plus future progress, discounted by effort already spent:

```
efficiency = (h / (h + actions_spent_on_level))²
A = current_level × efficiency
B = min(tail_cap, tail_base + current_level − 1)
C = w × 0.5^((actions/action_scale)²)
  + (1 − w) × 0.5^((tokens/token_scale)²)
priority_score = (A + B) × C
```

C is a heuristic based on measured completion-hazard decay, not an exact conditional probability. The implementation scales the score for integer priorities and combines it with the continuation bands.

| Setting | Default | Role |
|---|---|---|
| `ARC3_PRIORITY_HUMAN_ACTIONS` | `25` | h in the action-efficiency estimate. |
| `ARC3_PRIORITY_ACTION_SCALE` | `115` | Action falloff scale. |
| `ARC3_PRIORITY_TOKEN_SCALE` | `62000` | Generated-token falloff scale. |
| `ARC3_PRIORITY_ACTION_WEIGHT` | `0.25` | Action contribution to C; tokens contribute the remaining 0.75. |
| `ARC3_PRIORITY_TAIL_BASE` | **`2`** | Starting future-level tail. |
| `ARC3_PRIORITY_TAIL_CAP` | **`4`** | Maximum ordinary tail. |

The effective harness defaults are 2/4 even though the standalone helper’s function signature has 6/6 defaults. The wrapper supplies the environment-derived values. The successful 5/5 competition configuration is an explicit override.

**Counter correction is unconditional.** Completed-action accounting now takes the real engine action number into the call summary’s `end_action_num`; priority reads that corrected summary. There is no remaining `ARC3_PRIORITY_FIX_ACTIONS` switch.

### Pace adjustment

| Setting | Default | Role |
|---|---|---|
| `ARC3_PRIORITY_PACE` | Off | Adjusts expected cost using the speed of earlier completed levels in this game. |
| `ARC3_PRIORITY_PACE_REFERENCE` | Bundled reference JSON | Optional replacement reference data. |

The implementation smooths relative completion costs, shrinks estimates toward neutral when little evidence is available, and clips the multiplier to 0.5–2. It adjusts the token falloff scale and discounts value by expected cost. It does not require knowledge of the game’s final number of levels.

### Tail fading and endgame

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_PRIORITY_TAIL_FADE` | Off | Linearly reduces B near the deadline while retaining C. |
| `ARC3_PRIORITY_TAIL_FADE_FRACTION` | Unset | A value in (0, 1] selects the final fraction of total runtime for fading; 0.2 means the final 20%. |
| `ARC3_PRIORITY_TAIL_FADE_MINUTES` | `90` | Fallback fade duration when no valid fraction is configured. |
| `ARC3_ENDGAME_START_MINUTES` | `0`: disabled | Enables endgame after this many elapsed minutes. Ignored when tail fade is enabled. |

The solver configures a shared clock from session start and the applicable finite per-game/soft deadline. Fade needs a usable finite deadline. A log line announces fade start and its remaining duration.

**Tail fade takes precedence over endgame.** When `ARC3_PRIORITY_TAIL_FADE` is enabled, the endgame switch is disabled even if `ARC3_ENDGAME_START_MINUTES` is set. Scheduling uses `(A + faded_B) × C` throughout the run. With tail fade disabled, enabling endgame switches to `level × (0.25 + efficiency) / pace`, dropping both B and C. Without queue refreshing, endgame requests also receive a 500,000 phase offset. Refreshed ranking evaluates candidates consistently instead of mixing priorities computed in different phases.

### Optional remaining-level lookup and score normalization

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_PRIORITY_TAIL_LOOKUP` | Off | `remaining` selects the heuristic 8/7/5/0 tail by remaining levels; `1` selects the empirical 80k-token continuation table. Off keeps the constant/capped tail. |
| `ARC3_PRIORITY_TAIL_EFFICIENCY` | `0.8` | Multiplies empirical lookup-table future reward by assumed future action efficiency. Has no effect with `remaining` or with lookup disabled. |
| `ARC3_PRIORITY_SCORE_NORMALIZATION` | Off | Normalizes A only across games with different total level counts. B and the large cache-priority bands are unchanged. Independent of lookup. |

The total level count comes from harness metadata when available, is clipped to 6–10, and falls back to 10 when unavailable/invalid.

`ARC3_PRIORITY_TAIL_LOOKUP=remaining` chooses B according to the number of levels left **after completing the current level**:

| Remaining levels | B |
|---|---:|
| 3 or more | 8 |
| 2 | 7 |
| 1 | 5 |
| 0 | 0 |

These are final heuristic tail values, without an efficiency multiplier. This is the lookup mode used in the open-sourced submission, together with A-only score normalization.

`ARC3_PRIORITY_TAIL_LOOKUP=1` instead selects the empirical table estimating future weighted progress within an 80,000-token continuation budget. It does not bake in the separate efficiency factor. It was constructed from empirical level-completion data, with pooling/extrapolation where late-level data were sparse. Ordinary tail fade applies to either lookup result.

Normalization multiplies A by `55 / (N(N+1)/2)`, using a ten-level game as the reference. B is not normalized. The lookup and normalization switches remain off by default; the notebook enables the submission settings explicitly.

## 12. Yielding, reasoning, and provider compatibility

### Resumption behavior

| Setting | Default | Behavior |
|---|---|---|
| `LOCAL_ANALYZER_YIELD_TOKENS` | `0`: disabled | Adds a generated-token threshold for yielding an analyzer turn. Checked between completions; it is not a hard mid-stream token cutoff. |
| `ARC3_YIELD_RESUME_PROMPT` | `full` | Selects the full opener, a shorter continuation prompt, or `state_only`. |
| `ARC3_YIELD_RESUME_TONE` | `commit` | Selects commit-oriented, `neutral`, or `minimal` resumption wording. |
| `ARC3_YIELD_ON_TIMEOUT` | On | Allows a read timeout to yield while preserving useful history rather than losing the turn’s work. |

These extend the original time-based yielding/tool-step controls. They do not mean that every individual generation is interrupted exactly at the yield threshold. Image behavior on resumed turns is controlled separately, as listed above.

### Reasoning and tool-call compatibility

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_REASONING_HISTORY_KEY` | `reasoning` | Configures the reasoning field(s) retained in request history, including `reasoning_content` for compatible servers/models. |
| `ARC3_PRESERVE_THINKING` | `1` | Sends `preserve_thinking` in chat-template kwargs. False disables it; an empty value omits it. Preserves other template options such as enable-thinking. |
| `ARC3_REASONING_EFFORT_LADDER` | Empty: disabled | After output-length truncation, moves through configured reasoning-effort values, usually toward less reasoning. Resets after an action executes. |
| `ARC3_TOOLCALL_TEXT_FALLBACK` | On | Recovers recognizable tool calls emitted as text/markup when the backend did not produce structured tool-call objects. |
| `LOCAL_ANALYZER_TOOL_CHOICE` | `auto` | Overrides tool choice; empty, `omit`, or `none` omits the field while still sending tools. Other values pass through. |

The last point is deliberately different from sending the API value `tool_choice="none"`: this harness uses that environment value to omit the field for backend compatibility.

### OpenRouter support

Added streaming response assembly, including incremental text, reasoning, tool-call fragments, usage, and provider metadata, plus provider-selection controls and an API-key fallback.

| Setting | Default | Behavior |
|---|---|---|
| `OPENROUTER_STREAM` | Streaming enabled for OpenRouter | Can disable automatic streaming for this provider. |
| `OPENROUTER_PROVIDER_ORDER` | Empty | Optional ordered provider preferences. |
| `OPENROUTER_ALLOW_FALLBACKS` | Off when an order is supplied | Controls fallback outside the requested provider order. |
| `ARC3_OPENROUTER_PROVIDER` | Empty | Request-level provider pinning; takes precedence over general provider-order preferences. |
| `OPENROUTER_API_KEY` | Empty | Additional credential source for OpenRouter. |

These are separate from the existing local/OpenAI-compatible base URL, provider, model, temperature, top-p, top-k, and enable-thinking settings. Changing the served model is a deployment choice, not part of the cumulative harness source patch.

## 13. Request recovery, runtime budgets, and compatibility fixes

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_HTTP_RETRIES` | `3` | In-request retry count for transient HTTP/connection failures. Zero disables retries; negative allows unlimited retries. |
| `ARC3_HTTP_RETRY_BASE_SECONDS` | `5` | Base retry delay. |
| `ARC3_HTTP_RETRY_MAX_SECONDS` | `5` | Backoff ceiling, with jitter and separate handling of server Retry-After. |
| `ARC3_HTTP_RETRY_INITIAL_SECONDS` | `0` | Optional startup grace period for the first request while the model server is becoming available. |
| `ARC3_ANALYZER_RETRY_BACKOFF_SECONDS` | `10` | Delay between outer analyzer retries. |
| `ARC3_MAX_ANALYZER_FAILURES` | `10` | Limits consecutive outer analyzer failures; negative means unlimited, zero stops on the first failure. |
| `ARC3_WARMUP_ACTION_GAMES` | `0` | Lets the first N game sessions execute a counted automatic RESET before waiting for analyzer readiness. |

Additional recovery work:

- Recognizes SGLang’s context-overflow error wording as well as existing compatible-server forms. An overlong ordinary request now triggers history reduction and retry instead of endlessly resending the same oversized prompt.
- Preserves usable analysis/tool history across retries and timeout yields, and avoids treating a new unrelated prompt as the continuation of a failed call.
- Separates transient HTTP retries from context-overflow recovery and read-timeout handling.
- Passes the available timeout into summary generation and provides a finite 8192-token fallback where an unlimited output setting would otherwise make summary generation unbounded.
- Adds a per-game generated-token budget through `--max-generated-tokens-per-game`, the solver’s `max_generated_tokens_per_game`, and Makefile/config plumbing via `MAX_GENERATED_TOKENS_PER_GAME` / `environment.max_generated_tokens_per_game`.
- Uses compatibility fallbacks for newly optional solver state, allowing older pickled solver objects to run without assuming newly added attributes already exist.
- Releases admission resources on completion/failure and supplies the scheduler’s shared deadline information from the solver.

These changes do not increase the server’s true context limit. Correctly matching the harness context setting to the serving configuration is still necessary for efficient operation.

## 14. Diagnostics, transcript viewer, and build plumbing

### Viewer and request accounting — unconditional

- Associates analysis steps with exact recorded request snapshots, including multimodal-aware matching, rather than relying only on reconstructed text.
- Separates the full model context from the current turn’s local transcript in both normal and lazily loaded views.
- Aggregates request count, prompt tokens, and generated tokens across the requests/attempts belonging to an analysis step.
- Makes token badges available in the lightweight initial listing without expensive transcript/segmentation hydration.
- Displays pending transcript state explicitly while a step is still running.
- Combines retry attempts so earlier attempts and executed tool calls are not hidden by the last attempt.
- Prevents the latest-board-state card from duplicating the prior turn’s transcript or token totals.
- Shows world-model updates as separate information and improves their extraction/display.
- Updates runtime-global wording to match `last_action_call_result` and the current tool interface.

### Makefile — unconditional

Configuration-derived variables now use explicit undefined checks and immediate assignment, reducing repeated/recursive shell configuration evaluation while respecting external overrides. This spans game selection, budgets, concurrency, model/server settings, deployment settings, and related paths. Generated-token budget arguments are passed through to the runner.

These are build/run reliability changes, not new gameplay policies.

## 15. Publication scope and exclusions

For the public writeup, the principal additions can be grouped as:

1. Better action access and explanations: UNDO, RESET, and action information.
2. Cross-level learning and clearer observation semantics.
3. Richer perception: structured diffs, rotation-aware shapes, and intermediate animations.
4. Protection against ineffective batches, repeated mistakes, and stale plans.
5. Reusable model-written Python functions with controlled lifetime.
6. Larger-context operation: estimation, blockwise trimming, summaries, and recovery.
7. Cache-aware admission and progress-based allocation of computation.
8. Provider compatibility, diagnostics, and operational fixes.

Describe which of these were enabled in a particular submission separately. The inventory deliberately retains optional unsuccessful or inconclusive experiments that are still in the code. It does not attribute a measured score gain to a single mechanism.

The SGLang Mamba/prefix-cache patches, end-of-prefill LRU refresh, bounded parallel prefetcher, wheelhouse builder, model resharding, quantization choice, and serving performance configuration belong in a separate **serving and deployment** section. They are not changes in this harness diff.

## Source map

All paths below are relative to `ARC3-Inference/`.

| Area | Main source files |
|---|---|
| Prompts, context, summaries, memory, request loop | `inference/agent/prompts.py`, `inference/agent/tool_agent.py` |
| Action names and exposure | `inference/agent/action_names.py` |
| Action execution, guard integration, budgets, resets | `inference/framework/solver.py` |
| Runtime state and Python interface | `inference/agent/runtime_state.py`, `inference/agent/python_tool_sandbox.py` |
| Guard bookkeeping | `inference/agent/noop_repeat_guard.py` — new |
| Segmentation and differences | `inference/utils/segmentation.py`, `inference/utils/frame_diff.py` — latter new |
| Animations and images | `inference/utils/animation.py` — new; `inference/agent/vision_context.py` |
| Function retention | `inference/utils/retained_functions.py` — new |
| Priority gate, formulas, references | `inference/agent/priority_scheduler.py`, `inference/agent/priority_pace_reference.json` — new; integration in `tool_agent.py` and `solver.py` |
| Provider transport | `inference/utils/openai_compat.py` |
| Runner/build | `inference/framework/run.py`, `Makefile` |
| Viewer | `viewer/data.py`, `viewer/index.html` |
