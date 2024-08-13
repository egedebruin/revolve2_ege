"""Standard terrains."""

import math

import numpy as np
import numpy.typing as npt
from noise import pnoise2
from pyrr import Vector3, Quaternion

from revolve2.modular_robot_simulation import Terrain
from revolve2.simulation.scene import Pose
from revolve2.simulation.scene.geometry import GeometryHeightmap, GeometryPlane
from revolve2.simulation.scene.vector2 import Vector2


def flat(size: Vector2 = Vector2([20.0, 20.0])) -> Terrain:
    """
    Create a flat plane terrain.

    :param size: Size of the plane.
    :returns: The created terrain.
    """
    return Terrain(
        static_geometry=[
            GeometryPlane(
                pose=Pose(),
                mass=0.0,
                size=size,
            )
        ]
    )


def flat_thin(length: float = 20.0) -> Terrain:
    size = Vector2([3.0, length])
    heights = []
    for i in range(10):
        row_height = []
        for j in range(10):
            row_height.append(0)
        heights.append(row_height)

    return Terrain(
        static_geometry=[
            GeometryHeightmap(
                pose=Pose(position=Vector3([0, size[1] - 1, 0])),
                mass=0.0,
                size=Vector3([size[0], size[1], 1.0]),
                base_thickness=0.1,
                heights=heights,
            )
        ]
    )


def gradient(size: Vector2 = Vector2([20.0, 20.0]), angle=0.0) -> Terrain:
    """
    Create a flat plane terrain.

    :param size: Size of the plane.
    :param angle: Angle of plane
    :returns: The created terrain.
    """
    return Terrain(
        static_geometry=[
            GeometryPlane(
                pose=Pose(orientation=Quaternion(value=[0, angle, 0, 1])),
                mass=0.0,
                size=size,
            )
        ]
    )


def hills(length: float = 10.0, height=0.0, num_edges=50) -> Terrain:
    size = Vector2([3.0, length])
    height = height/100
    heights = []
    for i in range(num_edges):
        row_height = []
        for j in range(num_edges):
            if j > num_edges * 0.9:
                row_height.append(0)
                continue
            if j % (num_edges/20) == 0.0:
                row_height.append(height)
            else:
                row_height.append(0)
        heights.append(row_height)

    return Terrain(
        static_geometry=[
            GeometryHeightmap(
                pose=Pose(position=Vector3([0, size[1] - 1, 0])),
                mass=0.0,
                size=Vector3([size[0], size[1], 100]),
                base_thickness=0.1,
                heights=heights,
            )
        ]
    )


def steps(length: float = 10.0, height=0.0, num_edges=30) -> Terrain:
    size = Vector2([3.0, length])
    height = height/100
    heights = []
    for _ in range(num_edges):
        row_height = []
        current_height = 0
        for j in range(num_edges):
            if j < num_edges * 0.03:
                row_height.append(current_height)
                continue
            if j % (num_edges / 40) == 0.0:
                current_height += height
            row_height.append(current_height)
        row_height = row_height[::-1]
        heights.append(row_height)

    return Terrain(
        static_geometry=[
            GeometryHeightmap(
                pose=Pose(position=Vector3([0, size[1] - 1, 0])),
                mass=0.0,
                size=Vector3([size[0], size[1], 100]),
                base_thickness=0.1,
                heights=heights,
            )
        ]
    )


def thin_crater(
    size: tuple[float, float],
    ruggedness: float,
    curviness: float,
    granularity_multiplier: float = 1.0,
    wanted_size: float = 3.0
) -> Terrain:
    r"""
    Create a crater-like terrain with rugged floor using a heightmap.

    It will look like::

        |            |
         \_        .'
           '.,^_..'

    A combination of the rugged and bowl heightmaps.

    :param size: Size of the crater.
    :param ruggedness: How coarse the ground is.
    :param curviness: Height of the edges of the crater.
    :param granularity_multiplier: Multiplier for how many edges are used in the heightmap.
    :param wanted_size: Wanted width of the crater.
    :returns: The created terrain.
    """
    NUM_EDGES = 100  # arbitrary constant to get a nice number of edges

    num_edges = (
        int(NUM_EDGES * size[0] * granularity_multiplier),
        int(NUM_EDGES * size[1] * granularity_multiplier),
    )

    rugged = rugged_heightmap(
        size=size,
        num_edges=num_edges,
        density=1.5,
    )
    bowl = bowl_heightmap(num_edges=num_edges)

    max_height = ruggedness + curviness
    if max_height == 0.0:
        heightmap = np.zeros(num_edges)
        max_height = 1.0
    else:
        heightmap = (ruggedness * rugged + curviness * bowl) / (ruggedness + curviness)

    for i in range(num_edges[0]):
        for j in range(num_edges[1]):
            if j > num_edges[1] * 0.9:
                heightmap[i, j] = 0
            if i == (num_edges[0]/2 - (wanted_size/2)/size[0] * num_edges[0]) or i == (num_edges[0]/2 +(wanted_size/2)/size[0] * num_edges[0]):
                heightmap[i, j] = 2

    return Terrain(
        static_geometry=[
            GeometryHeightmap(
                pose=Pose(position=Vector3([0, size[1] - 1, 0])),
                mass=0.0,
                size=Vector3([size[0], size[1], max_height]),
                base_thickness=0.1,
                heights=heightmap,
            )
        ]
    )

def crater(
    size: tuple[float, float],
    ruggedness: float,
    curviness: float,
    granularity_multiplier: float = 1.0,
) -> Terrain:
    r"""
    Create a crater-like terrain with rugged floor using a heightmap.

    It will look like::

        |            |
         \_        .'
           '.,^_..'

    A combination of the rugged and bowl heightmaps.

    :param size: Size of the crater.
    :param ruggedness: How coarse the ground is.
    :param curviness: Height of the edges of the crater.
    :param granularity_multiplier: Multiplier for how many edges are used in the heightmap.
    :returns: The created terrain.
    """
    NUM_EDGES = 100  # arbitrary constant to get a nice number of edges

    num_edges = (
        int(NUM_EDGES * size[0] * granularity_multiplier),
        int(NUM_EDGES * size[1] * granularity_multiplier),
    )

    rugged = rugged_heightmap(
        size=size,
        num_edges=num_edges,
        density=1.5,
    )
    bowl = bowl_heightmap(num_edges=num_edges)

    max_height = ruggedness + curviness
    if max_height == 0.0:
        heightmap = np.zeros(num_edges)
        max_height = 1.0
    else:
        heightmap = (ruggedness * rugged + curviness * bowl) / (ruggedness + curviness)

    return Terrain(
        static_geometry=[
            GeometryHeightmap(
                pose=Pose(),
                mass=0.0,
                size=Vector3([size[0], size[1], max_height]),
                base_thickness=0.1 + ruggedness,
                heights=heightmap,
            )
        ]
    )


def rugged_heightmap(
    size: tuple[float, float],
    num_edges: tuple[int, int],
    density: float = 1.0,
) -> npt.NDArray[np.float_]:
    """
    Create a rugged terrain heightmap.

    It will look like::

        ..^.__,^._.-.

    Be aware: the maximum height of the heightmap is not actually 1.
    It is around [-1,1] but not exactly.

    :param size: Size of the heightmap.
    :param num_edges: How many edges to use for the heightmap.
    :param density: How coarse the ruggedness is.
    :returns: The created heightmap as a 2 dimensional array.
    """
    OCTAVE = 10
    C1 = 4.0  # arbitrary constant to get nice noise

    return np.fromfunction(
        np.vectorize(
            lambda y, x: pnoise2(
                x / num_edges[0] * C1 * size[0] * density,
                y / num_edges[1] * C1 * size[1] * density,
                OCTAVE,
            ),
            otypes=[float],
        ),
        num_edges,
        dtype=float,
    )


def bowl_heightmap(
    num_edges: tuple[int, int],
) -> npt.NDArray[np.float_]:
    r"""
    Create a terrain heightmap in the shape of a bowl.

    It will look like::

        |         |
         \       /
          '.___.'

    The height of the edges of the bowl is 1.0 and the center is 0.0.

    :param num_edges: How many edges to use for the heightmap.
    :returns: The created heightmap as a 2 dimensional array.
    """
    return np.fromfunction(
        np.vectorize(
            lambda y, x: (x / num_edges[0] * 2.0 - 1.0) ** 2
            + (y / num_edges[1] * 2.0 - 1.0) ** 2
            if math.sqrt(
                (x / num_edges[0] * 2.0 - 1.0) ** 2
                + (y / num_edges[1] * 2.0 - 1.0) ** 2
            )
            <= 1.0
            else 0.0,
            otypes=[float],
        ),
        num_edges,
        dtype=float,
    )
