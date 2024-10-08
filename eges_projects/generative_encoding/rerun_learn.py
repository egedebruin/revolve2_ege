"""Rerun the best robot between all experiments."""

import config
from evaluator import Evaluator
from sqlalchemy import select
from sqlalchemy.orm import Session

from database_components.learn_genotype import LearnGenotype
from database_components.learn_individual import LearnIndividual
from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.logging import setup_logging


def main() -> None:
    """Perform the rerun."""
    setup_logging()

    # Load the best individual from the database.
    dbengine = open_database_sqlite(
        config.DATABASE_FILE, open_method=OpenMethod.OPEN_IF_EXISTS
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
        fitness = row[1]

    modular_robot = genotype.develop()

    print(f"Best fitness: {fitness}")
    print(len(genotype.brain))

    # Create the evaluator.
    evaluator = Evaluator(headless=False, num_simulators=1)

    # Show the robot.
    evaluator.evaluate(modular_robot)


if __name__ == "__main__":
    main()
