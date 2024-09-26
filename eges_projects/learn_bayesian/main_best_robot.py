"""Main script for the example."""

import logging
import concurrent.futures
import os
from argparse import ArgumentParser

import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from sklearn.gaussian_process.kernels import Matern

import config
from database_components.base import Base
from evaluator import Evaluator
from database_components.experiment import Experiment
from database_components.generation import Generation
from database_components.population import Population
from sqlalchemy.orm import Session
import body_getter
from database_components.genotype import Genotype
from database_components.individual import Individual

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.rng import seed_from_time, make_rng


def run_experiment(i, genotype, old_file):
    logging.info("----------------")
    logging.info("Start experiment")

    # Open the database, only if it does not already exists.
    dbengine = open_database_sqlite(
        "results/after_learn_2309/" + old_file + "_" + str(i+1) + ".sqlite", open_method=OpenMethod.NOT_EXISTS_AND_CREATE
    )
    # Create the structure of the database.
    Base.metadata.create_all(dbengine)

    # Create an rng seed.
    rng_seed = seed_from_time() % 2**32  # Cma seed must be smaller than 2**32.
    rng = make_rng(rng_seed)

    # Create and save the experiment instance.
    experiment = Experiment(rng_seed=rng_seed)
    logging.info("Saving experiment configuration.")
    with Session(dbengine) as session:
        session.add(experiment)
        session.commit()

    # Intialize the evaluator that will be used to evaluate robots.
    evaluator = Evaluator(
        headless=True,
        num_simulators=config.NUM_SIMULATORS,
        environment='noisy'
    )

    pbounds = {}

    for uuid in genotype.brain.keys():
        pbounds['amplitude_' + str(uuid)] = [0, 1]
        pbounds['phase_' + str(uuid)] = [0, 1]

    optimizer = BayesianOptimization(
        f=None,
        pbounds=pbounds,
        allow_duplicate_points=True,
        random_state=int(rng.integers(low=0, high=2 ** 32))
    )
    optimizer.set_gp_params(alpha=config.ALPHA, kernel=Matern(nu=config.NU, length_scale=config.LENGTH_SCALE, length_scale_bounds=(config.LENGTH_SCALE - 0.01, config.LENGTH_SCALE + 0.01)))
    utility = UtilityFunction(kind="ucb", kappa=config.KAPPA)

    # Run cma for the defined number of generations.
    logging.info("Start optimization process.")

    best_value = -100
    best_point = {}

    for i in range(config.NUM_GENERATIONS + config.NUM_RANDOM_SAMPLES):
        logging.info(f"Generation {i + 1} / {config.NUM_GENERATIONS + config.NUM_RANDOM_SAMPLES}.")

        if i < config.NUM_RANDOM_SAMPLES:
            next_point = {}
            for key in genotype.brain.keys():
                next_point['amplitude_' + str(key)] = genotype.brain[key][0]
                next_point['phase_' + str(key)] = genotype.brain[key][1]
            next_point = dict(sorted(next_point.items()))
        else:
            bo_point = optimizer.suggest(utility)
            bo_utility = utility.utility([list(bo_point.values())], optimizer._gp, 0)
            next_point = {}
            next_best = 0
            for _ in range(10000):
                possible_point = {}
                for key in best_point.keys():
                    possible_point[key] = best_point[key] + np.random.normal(0, config.NEIGHBOUR_SCALE)
                possible_point = dict(sorted(possible_point.items()))

                utility_value = utility.utility([list(possible_point.values())], optimizer._gp, 0)
                if utility_value > next_best:
                    next_best = utility_value
                    next_point = possible_point
            if bo_utility >= next_best:
                next_point = bo_point

        new_learn_genotype = Genotype(brain={}, body=genotype.body)
        for brain_uuid in genotype.brain.keys():
            new_learn_genotype.brain[brain_uuid] = np.array(
                [
                    next_point['amplitude_' + str(brain_uuid)],
                    next_point['phase_' + str(brain_uuid)],
                ]
            )
        robot = new_learn_genotype.develop()

        fitness = evaluator.evaluate(robot)

        if fitness > best_value:
            best_value = fitness
            best_point = next_point

        optimizer.register(params=next_point, target=fitness)
        print(f"Fitness: {fitness}")

        population = Population(
            individuals=[
                Individual(genotype=new_learn_genotype, fitness=fitness)
            ]
        )

        # Make it all into a generation and save it to the database.
        generation = Generation(
            experiment=experiment,
            generation_index=i,
            population=population,
        )

        with Session(dbengine, expire_on_commit=False) as session:
            session.add(generation)
            session.commit()
    return True


def read_args():
    # Read args
    parser = ArgumentParser()
    parser.add_argument("--learn_environment", required=True)
    args = parser.parse_args()

    return "learn-", args.learn_environment


def run_experiments():
    file_name = 'learn-1'
    folder = "results/2309"
    files = [file for file in os.listdir(folder) if file.startswith(file_name)]
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=config.NUM_PARALLEL_PROCESSES
    ) as executor:
        futures = []
        for file in files:
            genotypes = body_getter.get_best_genotype(folder + "/" + file)
            for i, genotype in enumerate(genotypes):
                futures.append(executor.submit(run_experiment, i, genotype, file))
    for future in futures:
        future.result()


def main() -> None:
    """Run the program."""
    # Set up logging.
    run_experiments()


if __name__ == "__main__":
    main()
