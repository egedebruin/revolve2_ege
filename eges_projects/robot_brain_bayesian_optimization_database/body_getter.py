import os

import config

from revolve2.experimentation.database import open_database_sqlite, OpenMethod

from learn_genotype import LearnGenotype
from learn_individual import LearnIndividual

from sqlalchemy import select
from sqlalchemy.orm import Session


def get_best_genotype():
    dbengine = open_database_sqlite(
        config.DATABASE_FILE_OLD, open_method=OpenMethod.OPEN_IF_EXISTS
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
