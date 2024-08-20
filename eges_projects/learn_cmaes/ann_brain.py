import math

import numpy as np

from revolve2.modular_robot import ModularRobotControlInterface
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain import BrainInstance, Brain
from revolve2.modular_robot.sensor_state import ModularRobotSensorState

from ann import NeuralNetwork


class ANNBrain(Brain):
    active_hinges: list[ActiveHinge]
    model_parameters: list

    def __init__(self, active_hinges, model_parameters):
        self.active_hinges = active_hinges
        self.model_parameters = model_parameters

    def make_instance(self) -> BrainInstance:
        return ANNBrainInstance(active_hinges=self.active_hinges, model_parameters=self.model_parameters)


class ANNBrainInstance(BrainInstance):
    active_hinges: list[ActiveHinge]
    t: list[float]
    models: list[NeuralNetwork]

    def __init__(self, active_hinges, model_parameters):
        self.active_hinges = active_hinges
        self.t = [0.0] * len(active_hinges)
        self.models = []

        for model_parameter in model_parameters:
            model = NeuralNetwork()
            model.set_weights(model_parameter)
            self.models.append(model)

    def control(self, dt: float, sensor_state: ModularRobotSensorState,
                control_interface: ModularRobotControlInterface) -> None:
        i = 0
        for active_hinge, model in zip(self.active_hinges, self.models):

            touch_sensor = control_interface.get_touch_sensor(active_hinge)
            sine_sensor = 0.1 * math.sin(self.t[i])
            value = model.forward(np.array([[touch_sensor, sine_sensor]]))

            control_interface.set_active_hinge_target(active_hinge, value)

            self.t[i] += dt * 0.5
            i += 1
