"""Optional multimodal context helpers for ARC analyzer prompts."""
from __future__ import annotations

import base64
import io
from collections import defaultdict
import os
from typing import Any

from PIL import Image

from inference.agent.runtime_state import Frame


ARC_COLOR_MAP: dict[int, tuple[int, int, int]] = {
    0: (255, 255, 255),
    1: (204, 204, 204),
    2: (153, 153, 153),
    3: (102, 102, 102),
    4: (51, 51, 51),
    5: (0, 0, 0),
    6: (229, 58, 163),
    7: (255, 123, 204),
    8: (249, 60, 49),
    9: (30, 147, 255),
    10: (136, 216, 241),
    11: (255, 220, 0),
    12: (255, 133, 27),
    13: (146, 18, 49),
    14: (79, 204, 48),
    15: (163, 86, 214),
}


def multimodal_context() -> str:
    return os.environ.get("MULTIMODAL_CONTEXT", "").strip().lower()


def current_grid_image_enabled() -> bool:
    return multimodal_context() == "current_grid"


_RESAMPLE_FILTERS = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def multimodal_resample() -> Image.Resampling:
    """Upscale filter for grid images. Nearest neighbor (the default) keeps
    cells as flat exact-palette blocks; interpolating filters produce softer
    edges that sit closer to a vision encoder's natural-image training
    distribution, which has measurably helped on some models. Model-dependent:
    test against the model actually served."""
    raw = os.environ.get("MULTIMODAL_RESAMPLE", "").strip().lower()
    return _RESAMPLE_FILTERS.get(raw, Image.Resampling.NEAREST)


def current_grid_image_upscale() -> int:
    raw = os.environ.get("MULTIMODAL_UPSCALE", "").strip()
    if not raw:
        return 16
    try:
        return max(1, int(raw))
    except ValueError:
        return 16


def frame_to_png_data_url(frame: Frame, *, upscale: int | None = None) -> str:
    rows = len(frame.grid)
    cols = max((len(row) for row in frame.grid), default=0)
    if rows <= 0 or cols <= 0:
        raise ValueError("Cannot render an empty grid as an image.")

    scale = current_grid_image_upscale() if upscale is None else max(1, int(upscale))
    image = Image.new("RGB", (cols, rows), ARC_COLOR_MAP[0])
    pixels = image.load()
    for row_idx, row in enumerate(frame.grid):
        for col_idx in range(cols):
            value = row[col_idx] if col_idx < len(row) else 0
            pixels[col_idx, row_idx] = ARC_COLOR_MAP.get(int(value), ARC_COLOR_MAP[0])
    if scale > 1:
        image = image.resize((cols * scale, rows * scale), multimodal_resample())

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def current_grid_image_part(frame: Frame | None) -> dict[str, Any] | None:
    if frame is None or not current_grid_image_enabled():
        return None
    return {
        "type": "image_url",
        "image_url": {
            "url": frame_to_png_data_url(frame),
        },
    }


# Off-palette neutral for unchanged cells in diff images. The ARC palette's
# indices 0-5 span a white-to-black grayscale ramp, so no gray (or black, or
# white) can be used without colliding with real cell values; a dark
# desaturated navy is unambiguous.
DIFF_UNCHANGED_RGB: tuple[int, int, int] = (16, 24, 48)


_ANIMATION_IMAGE_MODES = ("off", "peak", "composite")


def _animation_image_mode(name: str) -> str:
    """One of off, peak, composite. Anything else raises.

    These two are the only enum knobs among a set of otherwise boolean ARC3_*
    variables, so `=1` is the natural thing to write - and silently meant off,
    which cost a whole run to notice. Failing at startup is better than a
    feature that is configured and quietly absent.
    """
    mode = os.environ.get(name, "off").strip().lower()
    if mode not in _ANIMATION_IMAGE_MODES:
        raise ValueError(
            f"{name}={mode!r} is not valid. Use one of "
            f"{', '.join(_ANIMATION_IMAGE_MODES)} - this knob is not a boolean."
        )
    return mode


def animation_death_image_mode() -> str:
    """off | peak | composite, for the fatal action of a game over.

    Separate from ARC3_ANIMATION_IMAGE because the two fire on different
    conditions and a game-over turn already carries two images, so a third is
    a cost worth deciding on its own.
    """
    return _animation_image_mode("ARC3_ANIMATION_DEATH_IMAGE")


def animation_image_mode() -> str:
    """off | peak | composite.

    Only ever attached when the action left the board area unchanged, which is
    exactly when the board image the model receives is uninformative: the
    action did something, and no frame it can reach shows what.
    """
    return _animation_image_mode("ARC3_ANIMATION_IMAGE")


def animation_composite_part(
    chain: list[tuple[tuple[int, ...], ...]],
    transient_cells: list[tuple[int, int]],
    *,
    border: int,
) -> dict[str, Any] | None:
    """Every transient cell in the last colour it held before reverting.

    Not a board state: cells from different moments composited onto one grid.
    The alternative - a single real frame - misses a sweep that touches
    different regions at different times, which has no peak moment.

    Everything else is dimmed to the diff image's off-palette navy, so the
    image carries only what no reachable frame contains. That includes cells
    that changed just once (visible in current_frame anyway) and the border,
    where the budget bar ticks on every action and would otherwise be the
    brightest thing in the picture.
    """
    if not chain or not transient_cells:
        return None
    if not current_grid_image_enabled():
        return None
    final = chain[-1]
    rows = len(final)
    cols = max((len(row) for row in final), default=0)
    if rows <= 0 or cols <= 0:
        return None

    wanted = set(transient_cells)
    last_seen: dict[tuple[int, int], int] = {}
    held: dict[tuple[int, int], int] = defaultdict(int)
    for grid in chain[:-1]:
        for row_idx, row in enumerate(grid):
            final_row = final[row_idx] if row_idx < len(final) else ()
            for col_idx, value in enumerate(row):
                cell = (row_idx, col_idx)
                if cell not in wanted:
                    continue
                if row_idx < border or row_idx >= rows - border:
                    continue
                if col_idx < border or col_idx >= cols - border:
                    continue
                final_value = final_row[col_idx] if col_idx < len(final_row) else None
                if value == final_value:
                    continue
                value = int(value)
                if last_seen.get(cell) != value:
                    # a cell that flickers between colours restarts the count,
                    # so the duration belongs to the colour actually painted
                    held[cell] = 0
                # later frames overwrite earlier ones, so this ends up holding
                # the LAST non-final colour the cell showed
                last_seen[cell] = value
                held[cell] += 1
    if not last_seen:
        return None

    # Brightness carries DURATION: a cell blended toward its true colour in
    # proportion to how many frames it held it. This separates two mechanics
    # that are otherwise identical in a flat composite - a 4x4 block falling
    # down a column touches each cell for a frame or two and renders uniformly
    # dim, while a column filling from the top holds its highest cell for the
    # whole animation and its lowest for a moment, so it renders as a gradient
    # pointing back at the source.
    span = max(1, len(chain) - 1)
    image = Image.new("RGB", (cols, rows), DIFF_UNCHANGED_RGB)
    pixels = image.load()
    for (row_idx, col_idx), value in last_seen.items():
        colour = ARC_COLOR_MAP.get(value, ARC_COLOR_MAP[0])
        # floored so a single-frame cell stays legible rather than vanishing
        weight = max(0.35, min(1.0, held[(row_idx, col_idx)] / span))
        pixels[col_idx, row_idx] = tuple(
            int(round(base + weight * (target - base)))
            for base, target in zip(DIFF_UNCHANGED_RGB, colour)
        )
    scale = current_grid_image_upscale()
    if scale > 1:
        image = image.resize((cols * scale, rows * scale), multimodal_resample())
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}


def animation_peak_part(
    chain: list[tuple[tuple[int, ...], ...]],
    transient_cells: list[tuple[int, int]],
) -> tuple[dict[str, Any] | None, int]:
    """The real frame showing the most transient cells at once, and its index.

    Unlike the composite this is a state the game actually displayed, so
    nothing in it is fabricated - at the cost of missing anything that was
    hidden at a different moment.
    """
    if not chain or not transient_cells or len(chain) <= 2:
        return None, -1
    if not current_grid_image_enabled():
        return None, -1
    final = chain[-1]
    wanted = set(transient_cells)
    best_index, best_count = -1, -1
    for index, grid in enumerate(chain[1:]):
        count = 0
        for row_idx, col_idx in wanted:
            row = grid[row_idx] if row_idx < len(grid) else ()
            final_row = final[row_idx] if row_idx < len(final) else ()
            value = row[col_idx] if col_idx < len(row) else None
            final_value = final_row[col_idx] if col_idx < len(final_row) else None
            if value != final_value:
                count += 1
        if count > best_count:
            best_index, best_count = index, count
    if best_index < 0 or best_count <= 0:
        return None, -1

    grid = chain[1:][best_index]
    rows = len(grid)
    cols = max((len(row) for row in grid), default=0)
    image = Image.new("RGB", (cols, rows), ARC_COLOR_MAP[0])
    pixels = image.load()
    for row_idx, row in enumerate(grid):
        for col_idx in range(cols):
            value = row[col_idx] if col_idx < len(row) else 0
            pixels[col_idx, row_idx] = ARC_COLOR_MAP.get(int(value), ARC_COLOR_MAP[0])
    scale = current_grid_image_upscale()
    if scale > 1:
        image = image.resize((cols * scale, rows * scale), multimodal_resample())
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
        best_index,
    )


def diff_image_part(
    previous_frame: Frame | None, current_frame: Frame | None
) -> dict[str, Any] | None:
    """Render a change-highlight image: cells that differ between the two
    frames appear in their (new) true palette color; unchanged cells are
    dimmed to an off-palette dark navy. Returns None when disabled, frames
    are missing, or shapes differ (a full redraw makes a diff meaningless)."""
    if previous_frame is None or current_frame is None:
        return None
    if not current_grid_image_enabled():
        return None
    prev_grid, cur_grid = previous_frame.grid, current_frame.grid
    rows = len(cur_grid)
    cols = max((len(row) for row in cur_grid), default=0)
    if rows <= 0 or cols <= 0:
        return None
    if len(prev_grid) != rows or max(
        (len(row) for row in prev_grid), default=0
    ) != cols:
        return None

    scale = current_grid_image_upscale()
    image = Image.new("RGB", (cols, rows), DIFF_UNCHANGED_RGB)
    pixels = image.load()
    for row_idx in range(rows):
        prev_row = prev_grid[row_idx]
        cur_row = cur_grid[row_idx]
        for col_idx in range(cols):
            prev_value = prev_row[col_idx] if col_idx < len(prev_row) else 0
            cur_value = cur_row[col_idx] if col_idx < len(cur_row) else 0
            if prev_value != cur_value:
                pixels[col_idx, row_idx] = ARC_COLOR_MAP.get(
                    int(cur_value), ARC_COLOR_MAP[0]
                )
    if scale > 1:
        image = image.resize((cols * scale, rows * scale), multimodal_resample())

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{encoded}",
        },
    }
