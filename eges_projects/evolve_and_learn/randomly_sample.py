import concurrent.futures
import json
import os
import random
from argparse import ArgumentParser

from sqlalchemy import Column, Integer, Float, select
from sqlalchemy.orm import Session, declarative_base

from database_components.generation import Generation
from database_components.genotype import Genotype
from database_components.individual import Individual
from database_components.learn_genotype import LearnGenotype
from database_components.population import Population
from evaluator import Evaluator

from genotypes.body_genotype_direct import CoreGenotype, BodyDeveloper
from revolve2.experimentation.database import open_database_sqlite, OpenMethod

Base = declarative_base()

class RandomSample(Base):
    __tablename__ = 'random_sample'
    id = Column(Integer, primary_key=True, autoincrement=True)
    genotype_id = Column(Integer)
    repetition = Column(Integer)
    objective_value = Column(Float)

def main(inherit_samples, environment, repetition):
    evaluator = Evaluator(headless=True, num_simulators=1)
    database_name = f"learn-30_controllers-adaptable_survivorselect-newest_parentselect-tournament_inheritsamples-{inherit_samples}_environment-{environment}_{repetition}."
    files = [file for file in os.listdir('results/new_big') if file.startswith(database_name)]

    for file_name in files:
        # Load the best individual from the database.
        dbengine = open_database_sqlite(
            'results/new_big/' + file_name, open_method=OpenMethod.OPEN_IF_EXISTS
        )
        dbengine_write = open_database_sqlite(
            'results/random/' + file_name, open_method=OpenMethod.NOT_EXISTS_AND_CREATE
        )
        Base.metadata.create_all(dbengine_write)

        with Session(dbengine) as ses:
            genotypes = ses.execute(
                select(Genotype._serialized_body, Genotype.id)
                .join_from(Generation, Population, Generation.population_id == Population.id)
                .join_from(Population, Individual, Population.id == Individual.population_id)
                .join_from(Individual, Genotype, Individual.genotype_id == Genotype.id)
                .where(Generation.generation_index <= 600)
            ).fetchall()

        with concurrent.futures.ProcessPoolExecutor(
                max_workers=100
        ) as executor:
            futures = []
            for serialized_body, genotype_id in genotypes:
                futures.append(executor.submit(sample, evaluator, serialized_body, genotype_id))

        for future in futures:
            samples, genotype_id = future.result()
            for (i, objective_value) in samples:
                new_data = RandomSample(genotype_id=genotype_id, repetition=i, objective_value=objective_value)
                with Session(dbengine_write) as session:
                    session.add(new_data)
                    session.commit()

def sample(evaluator, serialized_body, genotype_id):
    result = []

    body = CoreGenotype(0.0).deserialize(json.loads(serialized_body))
    body.reverse_phase_function(body.reverse_phase)
    body_developer = BodyDeveloper(body)
    body_developer.develop()
    brain_uuids = body.check_for_brains()

    for i in range(30):
        next_point = {}
        for key in brain_uuids:
            next_point['amplitude_' + str(key)] = random.random()
            next_point['phase_' + str(key)] = random.random()
            next_point['offset_' + str(key)] = random.random()

        new_learn_genotype = LearnGenotype(brain={})
        new_learn_genotype.next_point_to_brain(next_point, brain_uuids)

        modular_robot = new_learn_genotype.develop(body_developer.body)
        objective_value = evaluator.evaluate(modular_robot)
        result.append((i, objective_value))
    return result, genotype_id

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--environment", required=True)
    parser.add_argument("--inheritsamples", required=True)
    parser.add_argument("--repetition", required=True)
    args = parser.parse_args()
    main(args.inheritsamples, args.environment, args.repetition)