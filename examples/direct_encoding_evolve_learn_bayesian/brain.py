import math

import config
from revolve2.modular_robot import ModularRobotControlInterface
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain import BrainInstance, Brain
from revolve2.modular_robot.sensor_state import ModularRobotSensorState


class SineBrainInstance(BrainInstance):
    """ANN brain instance."""

    active_hinges: list[ActiveHinge]
    t: list[float]
    amplitudes: list[float]
    phases: list[float]
    touch_weights: list[float]
    sensor_phase_offset: list[float]

    def __init__(
        self,
        active_hinges: list[ActiveHinge],
        amplitudes: list[float],
        phases: list[float],
        touch_weights: list[float],
        sensor_phase_offset: list[float]
    ) -> None:
        """
        Initialize the Object.

        :param active_hinges: The active hinges to control.
        """
        self.active_hinges = active_hinges
        self.amplitudes = amplitudes
        self.phases = phases
        self.touch_weights = touch_weights
        self.sensor_phase_offset = sensor_phase_offset
        self.t = [0.0] * len(active_hinges)

    def control(
        self,
        dt: float,
        sensor_state: ModularRobotSensorState,
        control_interface: ModularRobotControlInterface,
    ) -> None:
        """
        Control the modular robot.

        :param dt: Elapsed seconds since last call to this function.
        :param sensor_state: Interface for reading the current sensor state.
        :param control_interface: Interface for controlling the robot.
        """

        # Here you can implement your controller.
        # The current controller does nothing except for always settings the joint positions to 0.5.
        i = -1
        for active_hinge, amplitude, phase, touch_weight, sensor_phase_offset \
                in zip(self.active_hinges, self.amplitudes, self.phases, self.touch_weights, self.sensor_phase_offset):
            i += 1
            touch_sensor = int(control_interface.get_touch_sensor(active_hinge) > 0)
            target = amplitude * math.sin(self.t[i] + phase * config.FREQUENCY)
            control_interface.set_active_hinge_target(active_hinge, target)
            self.t[i] += dt * config.FREQUENCY + dt*touch_sensor*touch_weight*math.sin(self.t[i] + sensor_phase_offset)


class SineBrain(Brain):
    """The ANN brain."""

    active_hinges: list[ActiveHinge]
    amplitudes: list[float]
    phases: list[float]
    touch_weights: list[float]
    sensor_phase_offset: list[float]

    def __init__(
        self,
        active_hinges: list[ActiveHinge],
        amplitudes: list[float],
        phases: list[float],
        touch_weights: list[float],
        sensor_phase_offset: list[float]
    ) -> None:
        """
        Initialize the Object.

        :param active_hinges: The active hinges to control.
        """
        self.active_hinges = active_hinges
        self.amplitudes = amplitudes
        self.phases = phases
        self.touch_weights = touch_weights
        self.sensor_phase_offset = sensor_phase_offset

    def make_instance(self) -> BrainInstance:
        """
        Create an instance of this brain.

        :returns: The created instance.
        """
        return SineBrainInstance(
            active_hinges=self.active_hinges,
            amplitudes=self.amplitudes,
            phases=self.phases,
            touch_weights=self.touch_weights,
            sensor_phase_offset=self.sensor_phase_offset
        )
