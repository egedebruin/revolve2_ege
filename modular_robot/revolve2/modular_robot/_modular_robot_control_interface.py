from abc import ABC, abstractmethod

from .body.base import ActiveHinge


class ModularRobotControlInterface(ABC):
    """Interface for controlling modular robots."""

    @abstractmethod
    def set_active_hinge_target(self, active_hinge: ActiveHinge, target: float) -> None:
        """
        Set the position target for an active hinge.

        :param active_hinge: The active hinge to set the target for.
        :param target: The target to set.
        """

    @abstractmethod
    def get_touch_sensor(self, active_hinge: ActiveHinge) -> float:
        pass

    @abstractmethod
    def get_actuator_force(self) -> float:
        pass
