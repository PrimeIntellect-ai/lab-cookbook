from typing import Literal

import verifiers.v1 as vf

Mode = Literal["single", "multi"]
Shape = Literal["circle", "square", "triangle", "star"]
Color = Literal["red", "blue", "green", "yellow"]
Pattern = Literal["solid", "striped", "dotted"]


class Tile(vf.StrictBaseModel):
    shape: Shape
    color: Color
    pattern: Pattern


class ShapeDetectiveTaskData(vf.TaskData):
    answer: str
    mode: Mode
    target_tile: Tile
