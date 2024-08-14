"""Evaluator class."""

import math
import config

import numpy as np
import numpy.typing as npt

from sine_brain import SineBrain

from revolve2.ci_group import fitness_functions, terrains
from revolve2.ci_group.simulation_parameters import make_standard_batch_parameters
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body.base import ActiveHinge, Body
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkStatic, CpgNetworkStructure
from revolve2.modular_robot_simulation import (
    ModularRobotScene,
    Terrain,
    simulate_scenes,
)
from revolve2.simulators.mujoco_simulator import LocalSimulator


class Evaluator:
    """Provides evaluation of robots."""

    _simulator: LocalSimulator
    _terrain: Terrain
    _cpg_network_structure: CpgNetworkStructure
    _body: Body
    _output_mapping: list[tuple[int, ActiveHinge]]
    _active_hinges: list[ActiveHinge]

    def __init__(
        self,
        headless: bool,
        num_simulators: int,
        cpg_network_structure: CpgNetworkStructure,
        body: Body,
        output_mapping: list[tuple[int, ActiveHinge]],
        active_hinges: list[ActiveHinge]
    ) -> None:
        """
        Initialize this object.

        :param headless: `headless` parameter for the physics simulator.
        :param num_simulators: `num_simulators` parameter for the physics simulator.
        :param cpg_network_structure: Cpg structure for the brain.
        :param body: Modular body of the robot.
        :param output_mapping: A mapping between active hinges and the index of their corresponding cpg in the cpg network structure.
        """
        self._simulator = LocalSimulator(
            headless=headless, num_simulators=num_simulators
        )
        # self._terrain = terrains.flat()
        self._terrain = terrains.hills(height=0.3)
        self._cpg_network_structure = cpg_network_structure
        self._body = body
        self._output_mapping = output_mapping
        self._active_hinges = active_hinges

    def evaluate(
        self,
        solutions: list[npt.NDArray[np.float_]],
    ) -> npt.NDArray[np.float_]:
        """
        Evaluate multiple robots.

        Fitness is the distance traveled on the xy plane.

        :param solutions: Solutions to evaluate.
        :returns: Fitnesses of the solutions.
        """
        # Create robots from the brain parameters.
        robots = []
        for params in solutions:
            amplitudes = params[:len(self._active_hinges)]
            phases = params[len(self._active_hinges):len(self._active_hinges) * 2] * 2 * math.pi
            touch_weights = (params[len(self._active_hinges) * 2:len(
                self._active_hinges) * 3] * config.FREQUENCY - config.FREQUENCY)
            neighbour_touch_weights = (params[len(self._active_hinges) * 3:len(
                self._active_hinges) * 4] * config.FREQUENCY - config.FREQUENCY)
            sensor_phase_offset = params[len(self._active_hinges) * 4:] * 2 * math.pi
            brain = SineBrain(active_hinges=self._active_hinges,
                              amplitudes=amplitudes,
                              phases=phases,
                              touch_weights=touch_weights,
                              neighbour_touch_weights=neighbour_touch_weights,
                              sensor_phase_offset=sensor_phase_offset)
            robots.append(ModularRobot(body=self._body, brain=brain))

        # Create the scenes.
        scenes = []
        for robot in robots:
            scene = ModularRobotScene(terrain=self._terrain)
            scene.add_robot(robot)
            scenes.append(scene)

        # Simulate all scenes.
        scene_states = simulate_scenes(
            simulator=self._simulator,
            batch_parameters=make_standard_batch_parameters(),
            scenes=scenes,
        )

        # Calculate the xy displacements.
        xy_displacements = [
            fitness_functions.forward_displacement(
                states[0].get_modular_robot_simulation_state(robot),
                states[-1].get_modular_robot_simulation_state(robot),
            )
            for robot, states in zip(robots, scene_states)
        ]

        return np.array(xy_displacements)
