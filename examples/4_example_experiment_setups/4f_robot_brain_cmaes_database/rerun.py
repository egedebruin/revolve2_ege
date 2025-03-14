"""Rerun the best robot between all experiments."""

import logging

import config
from database_components import Genotype, Individual
from evaluator import Evaluator
from sqlalchemy import select
from sqlalchemy.orm import Session

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.logging import setup_logging
from revolve2.modular_robot.body import RightAngles
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.body.v1 import BodyV1, ActiveHingeV1, BrickV1
from revolve2.modular_robot.brain.cpg import (
    active_hinges_to_cpg_network_structure_neighbor,
)

def make_body() -> BodyV1:
    """
    Create a body for the robot.

    :returns: The created body.
    """
    # A modular robot body follows a 'tree' structure.
    # The 'Body' class automatically creates a center 'core'.
    # From here, other modular can be attached.
    # Modules can be attached in a rotated fashion.
    # This can be any angle, although the original design takes into account only multiples of 90 degrees.
    # You should explore the "standards" module as it contains lots of preimplemented elements you can use!
    body = BodyV1()
    body.core_v1.front = ActiveHingeV1(rotation=RightAngles.DEG_90)
    body.core_v1.back = ActiveHingeV1(rotation=RightAngles.DEG_90)
    body.core_v1.left = ActiveHingeV1(rotation=RightAngles.DEG_90)
    body.core_v1.right = ActiveHingeV1(rotation=RightAngles.DEG_90)

    body.core_v1.front.attachment = BrickV1(rotation=RightAngles.DEG_270)
    body.core_v1.back.attachment = BrickV1(rotation=RightAngles.DEG_270)
    body.core_v1.left.attachment = BrickV1(rotation=RightAngles.DEG_270)
    body.core_v1.right.attachment = BrickV1(rotation=RightAngles.DEG_270)

    body.core_v1.front.attachment.front = ActiveHingeV1(rotation=RightAngles.DEG_0)
    body.core_v1.back.attachment.front = ActiveHingeV1(rotation=RightAngles.DEG_0)
    body.core_v1.left.attachment.front = ActiveHingeV1(rotation=RightAngles.DEG_0)
    body.core_v1.right.attachment.front = ActiveHingeV1(rotation=RightAngles.DEG_0)

    body.core_v1.front.attachment.front.attachment = BrickV1(rotation=RightAngles.DEG_0)
    body.core_v1.back.attachment.front.attachment = BrickV1(rotation=RightAngles.DEG_0)
    body.core_v1.left.attachment.front.attachment = BrickV1(rotation=RightAngles.DEG_0)
    body.core_v1.right.attachment.front.attachment = BrickV1(rotation=RightAngles.DEG_0)
    return body


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
        fitness = row[1]

    parameters = genotype.parameters

    logging.info(f"Best fitness: {fitness}")
    logging.info(f"Best parameters: {parameters}")

    # Prepare the body and brain structure
    maked_body = make_body()
    active_hinges = maked_body.find_modules_of_type(ActiveHinge)

    # Create the evaluator.
    evaluator = Evaluator(
        headless=False,
        num_simulators=1,
        active_hinges=active_hinges,
        body=maked_body,
    )

    # Show the robot.
    evaluator.evaluate([parameters])


if __name__ == "__main__":
    main()
