"""Genotype class."""

from __future__ import annotations

import sqlalchemy.orm as orm

import uuid

import numpy as np
from base import Base
import config
from body_genotype_direct import BodyGenotypeDirect
from brain_genotype import BrainGenotype

from revolve2.experimentation.database import HasId
from revolve2.modular_robot import ModularRobot


class Genotype(Base, HasId, BodyGenotypeDirect, BrainGenotype):
    """SQLAlchemy model for a genotype for a modular robot body and brain."""

    __tablename__ = "genotype"
    parent_1_genotype_id: orm.Mapped[int] = orm.mapped_column(default=-1)
    parent_2_genotype_id: orm.Mapped[int] = orm.mapped_column(default=-1)
    mutation_parameter: orm.Mapped[float] = orm.mapped_column(nullable=True, default=None)

    @classmethod
    def initialize(
        cls,
        rng: np.random.Generator,
    ) -> Genotype:
        """
        Create a random genotype.

        :param innov_db_body: Multineat innovation database for the body. See Multineat library.
        :param innov_db_brain: Multineat innovation database for the brain. See Multineat library.
        :param rng: Random number generator.
        :returns: The created genotype.
        """
        brain = cls.initialize_brain()
        body = cls.initialize_body(rng=rng, brain=brain)

        return Genotype(body=body.body, brain=brain.brain)

    def mutate(
        self,
        rng: np.random.Generator,
    ) -> Genotype:
        """
        Mutate this genotype.

        This genotype will not be changed; a mutated copy will be returned.

        :param innov_db_body: Multineat innovation database for the body. See Multineat library.
        :param innov_db_brain: Multineat innovation database for the brain. See Multineat library.
        :param rng: Random number generator.
        :returns: A mutated copy of the provided genotype.
        """
        brain = BrainGenotype(brain=self.brain.copy())
        body, mutation_parameter = self.mutate_body(rng, brain)

        genotype = Genotype(body=body.body, brain=brain.brain)
        genotype.mutation_parameter = mutation_parameter
        return genotype

    @staticmethod
    def crossover(
        parent1: Genotype,
        parent2: Genotype,
        rng: np.random.Generator,
    ) -> (Genotype, Genotype):
        """
        Perform crossover between two genotypes.

        :param parent1: The first genotype.
        :param parent2: The second genotype.
        :param rng: Random number generator.
        :returns: A newly created genotype.
        """
        child1, child2 = BodyGenotypeDirect.crossover_body(parent1, parent2, rng)

        if config.CONTROLLERS != -1:
            return Genotype(body=child1.body, brain=parent1.brain), Genotype(body=child2.body, brain=parent2.brain)

        all_brains = {**parent1.brain, **parent2.brain}

        child_1_brain = {key: all_brains[key] for key in child1.get_brain_uuids() if key in all_brains}
        child_2_brain = {key: all_brains[key] for key in child2.get_brain_uuids() if key in all_brains}

        if len(child_1_brain.keys()) == 0:
            new_uuid = uuid.uuid4()
            child_1_brain = {new_uuid: np.array([])}

        if len(child_2_brain.keys()) == 0:
            new_uuid = uuid.uuid4()
            child_2_brain = {new_uuid: np.array([])}

        return Genotype(body=child1.body, brain=child_1_brain), Genotype(body=child2.body, brain=child_2_brain)

    def develop(self) -> ModularRobot:
        """
        Develop the genotype into a modular robot.

        :returns: The created robot.
        """
        body = self.develop_body()
        brain = self.develop_brain(body)
        return ModularRobot(body=body, brain=brain)
