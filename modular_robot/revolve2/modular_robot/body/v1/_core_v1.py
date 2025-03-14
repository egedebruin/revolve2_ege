from pyrr import Vector3

from .._right_angles import RightAngles
from ..base import Core


class CoreV1(Core):
    """The core module of a v1 modular robot."""

    def __init__(self, rotation: float | RightAngles):
        """
        Initialize this object.

        :param rotation: The modules' rotation.
        """
        super().__init__(
            rotation=rotation,
            bounding_box=Vector3([0.06485, 0.06485, 0.06485]),
            mass=0.088,
            child_offset=0.06485 / 2.0,
            sensors=[],
        )
