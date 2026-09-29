"""Local test game: click every light to switch it off.

Exercises ACTION6 (click at x, y). Written for offline agent testing only;
it is not one of the competition games.
"""
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

LIGHT = Sprite(pixels=[[4]], name="light", tags=["light"])

levels = [
    Level(
        sprites=[LIGHT.clone().set_position(x, y) for x, y in [(1, 1), (6, 2), (3, 5)]],
        grid_size=(8, 8),
    ),
    Level(
        sprites=[
            LIGHT.clone().set_position(x, y)
            for x, y in [(0, 0), (7, 0), (2, 3), (5, 4), (1, 7), (6, 6)]
        ],
        grid_size=(8, 8),
    ),
]


class Lt01(ARCBaseGame):
    def __init__(self) -> None:
        super().__init__(
            game_id="lt01",
            levels=levels,
            camera=Camera(background=0, letter_box=3),
            available_actions=[6],
        )

    def step(self) -> None:
        if self.action.id == GameAction.ACTION6:
            pos = self.camera.display_to_grid(self.action.data["x"], self.action.data["y"])
            if pos is not None:
                hit = self.current_level.get_sprite_at(pos[0], pos[1], tag="light")
                if hit is not None:
                    self.current_level.remove_sprite(hit)
            if not self.current_level.get_sprites_by_tag("light"):
                self.next_level()
        self.complete_action()
