import config

from revolve2.experimentation.database import open_database_sqlite, OpenMethod

from database_components.learn_genotype import LearnGenotype
from database_components.learn_individual import LearnIndividual

from sqlalchemy import select
from sqlalchemy.orm import Session


def get_best_genotype(file_name):
    dbengine = open_database_sqlite(
        file_name, open_method=OpenMethod.OPEN_IF_EXISTS
    )
    with Session(dbengine) as ses:
        row = ses.execute(
            select(LearnGenotype, LearnIndividual.fitness)
            .join_from(LearnGenotype, LearnIndividual, LearnGenotype.id == LearnIndividual.genotype_id)
            .order_by(LearnIndividual.fitness.desc())
            .limit(1)
        ).one()
        assert row is not None
    genotype = row[0]

    return genotype
