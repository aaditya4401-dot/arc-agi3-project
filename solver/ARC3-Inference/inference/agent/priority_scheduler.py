"""Pure scheduler inputs and scoring; never reads hidden game metadata."""
from __future__ import annotations

import math
from dataclasses import dataclass


# Expected future level-weight sum, conditional on completing the current level.
# 80k tokens are shared by ALL future levels; efficiency is NOT baked in.
# Source: arc3-levelup-chances, pooled_late_provisional, N6..10_H80000 / 0.8.
# KM duration distributions: separate levels 1-4, pooled observed levels 5-8
# for levels 5 onward. Levels 9-10 are extrapolated. Costs rounded up to 250
# tokens. 336 historical passes / 1,417 reached levels; not a contest fit.
# Tuples are indexed by current_level - 1. The final level has no future tail.
TAIL_LOOKUP_80K: dict[int, tuple[float, ...]] = {
    6: (3.9082328014, 5.0756939323, 5.3532912917, 5.6080111343,
        4.0898867400, 0.0),
    7: (3.9262240771, 5.1669653335, 5.6943514417, 6.6500041602,
        6.6562876217, 4.7715345300, 0.0),
    8: (3.9291260268, 5.1855629875, 5.7851444897, 7.0053702980,
        7.8471367942, 7.7045641090, 5.4531823200, 0.0),
    9: (3.9295002558, 5.1885168225, 5.8035897108, 7.0958580947,
        8.2469236991, 9.0442694281, 8.7528405964, 6.1348301100, 0.0),
    10: (3.9295397677, 5.1888936715, 5.8065473797, 7.1136691211,
         8.3474656955, 9.4884771003, 10.2414020621, 9.8011170838,
         6.8164779000, 0.0),
}


# Heuristic continuation value by levels remaining AFTER the current level:
# 3 or more -> 8; 2 -> 7; 1 -> 5; 0 -> 0. These are final B values,
# so neither score normalization nor the empirical efficiency discount applies.
TAIL_LOOKUP_REMAINING: dict[int, tuple[float, ...]] = {
    n: (8.0,) * (n - 3) + (7.0, 5.0, 0.0) for n in range(6, 11)
}


def parse_total_levels(value: object) -> int | None:
    """Validate public level-count metadata, then clip to the table domain."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or number <= 0 or not number.is_integer():
        return None
    return min(10, max(6, int(number)))


@dataclass
class ProgressPace:
    completed: int = 0
    log_cost_ratio: float = 0.0

    def record_completion(self, tokens_used: float, reference_tokens: float) -> None:
        observation = math.log(max(1.0, tokens_used) / max(1.0, reference_tokens))
        self.log_cost_ratio = (
            observation if self.completed == 0
            else 0.6 * self.log_cost_ratio + 0.4 * observation
        )
        self.completed += 1

    def cost_multiplier(self) -> float:
        confidence = self.completed / (self.completed + 3.0)
        return math.exp(min(math.log(2.0), max(math.log(0.5),
                                             confidence * self.log_cost_ratio)))


@dataclass(frozen=True)
class PrioritySnapshot:
    level: int
    actions: int
    tokens: float
    cost_multiplier: float = 1.0
    # Appended to preserve four-argument callers and older pickled snapshots.
    total_levels: int | None = None


def priority_value(
    state: PrioritySnapshot, *, endgame: bool = False,
    tail_fraction: float | None = None, human_actions: float = 25.0,
    action_scale: float = 115.0, token_scale: float = 62000.0,
    action_weight: float = 0.25, tail_base: float = 6.0, tail_cap: float = 6.0,
    normalize_score: bool = False, tail_lookup: bool | str = False,
    tail_efficiency: float = 0.8,
) -> int:
    """No phase offset: all candidates must be evaluated at the same time.

    tail_fraction=None retains the endgame rule. Supplying a fraction
    keeps the likelihood proxy active and fades the tail instead of switching.
    The likelihood proxy is heuristic, not a calibrated completion probability.

    tail_lookup=True uses the empirical 80k table; "remaining" uses the
    fixed 8/7/5/0 table without an efficiency discount. False keeps the capped
    tail. normalize_score scales only A, never B or the endgame's 0.25 floor.
    """
    level = max(1, state.level)
    total_levels = parse_total_levels(getattr(state, "total_levels", None)) or 10
    actions, tokens = max(0, state.actions), max(0.0, state.tokens)
    pace = min(2.0, max(0.5, state.cost_multiplier))
    efficiency = (human_actions / (human_actions + actions)) ** 2
    current_value = level * efficiency
    if normalize_score:
        # Normalize immediate reward A to a ten-level reference (sum 1..10).
        # B is a separately chosen continuation value, not normalized reward.
        current_value *= 55.0 / (total_levels * (total_levels + 1) / 2)
    if endgame and tail_fraction is None:
        # Preserve the original arithmetic when normalization is disabled.
        value = (
            level * 0.25 + current_value if normalize_score
            else level * (0.25 + efficiency)
        ) / pace * 100
    else:
        if tail_lookup == "remaining":
            index = min(total_levels, level) - 1
            tail = TAIL_LOOKUP_REMAINING[total_levels][index]
        elif tail_lookup:
            future_efficiency = (
                min(1.0, max(0.0, tail_efficiency))
                if math.isfinite(tail_efficiency) else 0.8
            )
            index = min(total_levels, level) - 1
            tail = TAIL_LOOKUP_80K[total_levels][index] * future_efficiency
        else:
            tail = min(tail_cap, tail_base + level - 1)
        if tail_fraction is not None:
            tail *= min(1.0, max(0.0, tail_fraction))
        chance = (
            action_weight * 0.5 ** ((actions / action_scale) ** 2)
            + (1.0 - action_weight) * 0.5 ** ((tokens / (token_scale * pace)) ** 2)
        )
        value = (current_value + tail) * chance / pace * 100
    return max(1, int(value))
