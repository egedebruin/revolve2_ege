"""Genotype class."""

from __future__ import annotations

from database_components.base import Base
from database_components.brain_genotype import BrainGenotype

from revolve2.experimentation.database import HasId
from revolve2.modular_robot import ModularRobot

import sqlalchemy.orm as orm


class LearnGenotype(Base, HasId, BrainGenotype):
    """A genotype that is an array of parameters."""

    __tablename__ = "learn_genotype"
    execution_time: orm.Mapped[float] = orm.mapped_column(default=0.0)

    def develop(self, body) -> ModularRobot:
        """
        Develop the genotype into a modular robot.

        :returns: The created robot.
        """
        brain = self.develop_brain(body)
        return ModularRobot(body=body, brain=brain)