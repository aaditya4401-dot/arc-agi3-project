"""Known-no-op repeat guard.

Records ``(level, interior_state_hash, action_signature)`` triples that were
observed to change nothing outside the board border, and refuses to execute
them again in a bit-identical state.

Two deliberate choices:

* The state key uses ``_interior_state_hash`` - the same border-cropped hash
  the death ledger uses - so "the same state" means the same thing everywhere
  in the harness, and a ticking HUD timer does not make every state unique.

* Blocking is overridable. A blocked action arms a single slot; if the model
  immediately re-issues that same action, it executes. This avoids the
  self-sealing failure of a hard block: once an action is on the list, a hard
  block guarantees no contradicting evidence can ever arrive, so a mechanic
  gated on hidden state (a counter, a cooldown, an off-screen timer) would be
  foreclosed forever. Each insistence costs exactly one action and the entry
  survives, so the model can always insist again.

Known limitation: the harness does not receive per-action frame counts, so an
action whose effect appears only in intermediate animation frames (a rejected
click, a bounce) looks identical to one that did nothing, and will be
recorded. The override path is the only recourse there, which is why this
guard defaults to off.
"""
from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any, Callable


_DISPLAY_COORDS_RE = re.compile(
    r"^([A-Z0-9_]+)\(\s*(?:ROW\s*=\s*)?(-?\d+)\s*,\s*(?:COL\s*=\s*)?(-?\d+)\s*\)$"
)


def action_signature(action: Any) -> str:
    """Stable signature for one action. Mouse coordinates are part of the
    identity: a click is only "the same action" at the same cell.

    Accepts both shapes the harness uses - the dict form the sandbox passes
    (`{"action": "MOUSE", "row": 4, "col": 7}`) and the display form the death
    ledger stores (`"MOUSE(row=4, col=7)"`) - and normalizes them to the same
    string, so a verdict recorded through one path is found through the other.
    Callers must pass MODEL-facing names (UP/DOWN/...), not engine names
    (ACTION1/ACTION2/...).
    """
    if isinstance(action, dict):
        name = str(action.get("action") or action.get("name") or "").strip().upper()
        row = action.get("row")
        col = action.get("col")
        if row is not None or col is not None:
            return f"{name}({row},{col})"
        return name
    text = " ".join(str(action or "").split()).upper()
    match = _DISPLAY_COORDS_RE.match(text)
    if match:
        return f"{match.group(1)}({match.group(2)},{match.group(3)})"
    return text


class NoopRepeatGuard:
    """Per-level store of state+action combinations known to do nothing."""

    def __init__(self, *, max_states_per_level: int = 512, max_actions_per_state: int = 16) -> None:
        self._max_states = max(1, int(max_states_per_level))
        self._max_actions = max(1, int(max_actions_per_state))
        self._states: "OrderedDict[str, OrderedDict[str, None]]" = OrderedDict()
        self._level: Any = None
        # One slot per guard kind. A shared slot let whichever guard ran first
        # consume the other's override: the death check runs before the no-op
        # check, cleared the arm on every non-match, and re-armed it, so the
        # no-op guard's "issue it again and it will execute" never fired. A
        # mouse-only game observed 291 replies and one executed action.
        self._armed_noop: tuple[str, str] | None = None
        self._armed_death: tuple[str, str] | None = None

    # -- level scoping -------------------------------------------------------
    def note_level(self, level: Any) -> None:
        """Drop everything when the level changes: a new layout means old
        state hashes describe boards that no longer exist. Deaths and resets
        deliberately do NOT clear the store - the board returns to the level
        start, so recorded no-ops still hold and carry across attempts."""
        if level != self._level:
            self._level = level
            self._states.clear()
            self._armed_noop = None
            self._armed_death = None

    # -- recording -----------------------------------------------------------
    def observe(self, state_hash: str, action_sig: str, *, gameplay_changed: Any) -> None:
        if not state_hash or not action_sig:
            return
        if gameplay_changed is False:
            entry = self._states.get(state_hash)
            if entry is None:
                entry = OrderedDict()
                self._states[state_hash] = entry
                while len(self._states) > self._max_states:
                    self._states.popitem(last=False)
            entry[action_sig] = None
            while len(entry) > self._max_actions:
                entry.popitem(last=False)
        elif gameplay_changed is True:
            # Evidence contradicts a recorded no-op: drop it. Reachable via the
            # override path, where a blocked action is executed after all.
            entry = self._states.get(state_hash)
            if entry is not None and action_sig in entry:
                del entry[action_sig]
                if not entry:
                    del self._states[state_hash]

    # -- blocking ------------------------------------------------------------
    def should_block(self, state_hash: str, action_sig: str) -> bool:
        """True when this combination is a known no-op AND the model has not
        just been told so. Arms the override slot when it blocks."""
        if not state_hash or not action_sig:
            return False
        key = (state_hash, action_sig)
        if self._armed_noop == key:
            # the model saw the refusal and insisted: let it through, and
            # re-arm nothing so a third attempt is blocked again
            self._armed_noop = None
            return False
        entry = self._states.get(state_hash)
        if not entry or action_sig not in entry:
            self._armed_noop = None
            return False
        self._armed_noop = key
        return True

    def note_executed(self, state_hash: str, action_sig: str) -> None:
        """Any executed action that is not the armed one clears the slot, so
        the override only applies to an immediate repeat."""
        key = (state_hash, action_sig)
        if self._armed_noop and self._armed_noop != key:
            self._armed_noop = None
        if self._armed_death and self._armed_death != key:
            self._armed_death = None

    # -- death verdicts ------------------------------------------------------
    def should_block_death(
        self,
        state_hash: str,
        action_sig: str,
        is_known_fatal: "Callable[[str, str], bool]",
    ) -> bool:
        """Same block-then-insist contract as `should_block`, but the verdict
        comes from the death ledger rather than this store.

        The ledger already keeps, per recorded attempt, an `actions` list and
        an index-aligned `hashes` list of interior state hashes taken BEFORE
        each action - so "this action, from this exact board, killed us" is a
        lookup, not a sequence match. Matching per position rather than by
        sequence prefix is what makes an interleaved or differently-routed
        approach to the same fatal move still get caught, while a batch whose
        earlier actions changed the board correctly does NOT match.

        Deaths are more likely than no-ops to hinge on state the board does
        not show (a hazard's facing, a timer), so blocking must stay
        overridable: under permadeath a hard block could foreclose the only
        path through a level with no way to ever discover the error.
        """
        if not state_hash or not action_sig:
            return False
        key = (state_hash, action_sig)
        if self._armed_death == key:
            self._armed_death = None
            return False
        if not is_known_fatal(state_hash, action_sig):
            # deliberately does NOT clear the slot: an action this guard has no
            # verdict on is none of its business, and clearing here would drop
            # an override armed for a different action
            return False
        self._armed_death = key
        return True
