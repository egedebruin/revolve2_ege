"""Genotype class."""

from __future__ import annotations

import multineat
import numpy as np
import sqlalchemy.orm as orm

from revolve2.ci_group.genotypes.cppnwin.modular_robot.v1 import BodyGenotypeOrmV1
from .base import Base

from database_components.brain_genotype import BrainGenotype
from revolve2.experimentation.database import HasId


class Genotype(Base, HasId, BodyGenotypeOrmV1, BrainGenotype):
    """SQLAlchemy model for a genotype for a modular robot body and brain."""

    __tablename__ = "genotype"
    parent_1_genotype_id: orm.Mapped[int] = orm.mapped_column(default=-1)
    parent_2_genotype_id: orm.Mapped[int] = orm.mapped_column(default=-1)

    @classmethod
    def initialize(
        cls,
        innov_db_body: multineat.InnovationDatabase,
        rng: np.random.Generator,
    ) -> Genotype:
        """
        Create a random genotype.

        :param innov_db_body: Multineat innovation database for the body. See Multineat library.
        :param innov_db_brain: Multineat innovation database for the brain. See Multineat library.
        :param rng: Random number generator.
        :returns: The created genotype.
        """
        body = cls.random_body(innov_db_body, rng)

        return Genotype(body=body.body, brain={})

    def mutate(
        self,
        innov_db_body: multineat.InnovationDatabase,
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
        body = self.mutate_body(innov_db_body, rng)
        brain = self.mutate_brain(rng)

        genotype = Genotype(body=body.body, brain=brain.brain)
        return genotype

    @classmethod
    def crossover(
            cls,
            parent1: Genotype,
            parent2: Genotype,
            rng: np.random.Generator,
    ) -> Genotype:
        """
        Perform crossover between two genotypes.

        :param parent1: The first genotype.
        :param parent2: The second genotype.
        :param rng: Random number generator.
        :returns: A newly created genotype.
        """
        body = cls.crossover_body(parent1, parent2, rng)
        brain = cls.crossover_brain(parent1, parent2)

        return Genotype(body=body.body, brain=brain.brain)
