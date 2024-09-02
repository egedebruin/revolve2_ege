"""Individual class."""

from dataclasses import dataclass

from sqlalchemy import orm

from .base import Base
from .genotype import Genotype

from revolve2.experimentation.optimization.ea import Individual as GenericIndividual


@dataclass
class Individual(
    Base, GenericIndividual[Genotype], population_table="population", kw_only=True
):
    """An individual in a population."""

    __tablename__ = "individual"
    original_generation: orm.Mapped[int] = orm.mapped_column(nullable=False)
    objective_value: orm.Mapped[float] = orm.mapped_column(nullable=False)
    reproduction_fitness: orm.Mapped[float] = orm.mapped_column(nullable=True)
    survivor_fitness: orm.Mapped[float] = orm.mapped_column(nullable=True)
