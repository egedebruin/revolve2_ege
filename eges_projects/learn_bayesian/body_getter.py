import config

from revolve2.experimentation.database import open_database_sqlite, OpenMethod

from database_components.learn_genotype import LearnGenotype
from database_components.learn_individual import LearnIndividual
from database_components.learn_population import LearnPopulation
from database_components.learn_generation import LearnGeneration
from database_components.genotype import Genotype
from database_components.individual import Individual
from database_components.population import Population
from database_components.generation import Generation

from sqlalchemy import select
from sqlalchemy.orm import Session


def get_best_genotype(file_name):
    dbengine = open_database_sqlite(
        file_name, open_method=OpenMethod.OPEN_IF_EXISTS
    )
    with Session(dbengine) as ses:
        row = ses.execute(
            select(LearnGenotype)
            .join_from(LearnGenotype, LearnIndividual, LearnGenotype.id == LearnIndividual.genotype_id)
            .join_from(LearnIndividual, LearnPopulation, LearnIndividual.population_id == LearnPopulation.id)
            .join_from(LearnPopulation, LearnGeneration, LearnPopulation.id == LearnGeneration.learn_population_id)
            .join_from(LearnGeneration, Genotype, Genotype.id == LearnGeneration.genotype_id)
            .join_from(Genotype, Individual, Individual.genotype_id == Genotype.id)
            .join_from(Individual, Population, Population.id == Individual.population_id)
            .join_from(Population, Generation, Generation.population_id == Population.id)
            .where(Generation.generation_index == 166)
        ).all()
        assert row is not None

    genotypes = []
    for genotype in row:
        genotypes.append(genotype[0])

    return genotypes
