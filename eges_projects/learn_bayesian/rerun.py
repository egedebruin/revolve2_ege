"""Rerun the best robot between all experiments."""

import config
from evaluator import Evaluator
from database_components.genotype import Genotype
from database_components.individual import Individual
from sqlalchemy import select
from sqlalchemy.orm import Session

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
            select(Genotype, Individual.fitness)
            .join_from(Genotype, Individual, Genotype.id == Individual.genotype_id)
            .order_by(Individual.fitness.desc())
            .limit(1)
        ).one()
        assert row is not None

        genotype = row[0]

    robot = genotype.develop()
    print(genotype.brain)

    # Create the evaluator.
    evaluator = Evaluator(headless=False, num_simulators=1)

    # Show the robot.
    evaluator.evaluate(robot)


if __name__ == "__main__":
    main()
