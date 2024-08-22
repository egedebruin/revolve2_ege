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


def latin_hypercube(n, k, rng: np.random.Generator):
    """
    Generate Latin Hypercube samples.

    Parameters:
    n (int): Number of samples.
    k (int): Number of dimensions.

    Returns:
    numpy.ndarray: Array of Latin Hypercube samples of shape (n, k).
    """
    # Generate random permutations for each dimension
    perms = [rng.permutation(n) for _ in range(k)]

    # Generate the samples
    samples = np.empty((n, k))

    for i in range(k):
        # Generate the intervals
        interval = np.linspace(0, 1, n+1)

        # Assign values from each interval to the samples
        for j in range(n):
            samples[perms[i][j], i] = rng.uniform(interval[j], interval[j+1])

    return samples


def run_experiment(i, old_file, environment):
    logging.info("----------------")
    logging.info("Start experiment")

    # Open the database, only if it does not already exists.
    dbengine = open_database_sqlite(
        'results/after_learn3/' + old_file.replace(".sqlite", "") + "_" + environment + "_" + str(i+1) + ".sqlite", open_method=OpenMethod.NOT_EXISTS_AND_CREATE
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
        environment=environment
    )

    genotype = body_getter.get_best_genotype('results/to_learn/' + old_file)
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

    lhs = latin_hypercube(config.NUM_RANDOM_SAMPLES, 2 * len(genotype.brain.keys()), rng)

    # Run cma for the defined number of generations.
    logging.info("Start optimization process.")

    best_value = 0
    best_point = {}

    for i in range(config.NUM_GENERATIONS + config.NUM_RANDOM_SAMPLES):
        logging.info(f"Generation {i + 1} / {config.NUM_GENERATIONS + config.NUM_RANDOM_SAMPLES}.")

        if i < config.NUM_RANDOM_SAMPLES:
            j = 0
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
    parser.add_argument("--learn", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--learn_environment", required=True)
    args = parser.parse_args()

    return "learn-" + str(args.learn) + "_evosearch-1_controllers-adaptable_select-tournament_environment-" + args.environment, args.learn_environment


def run_experiments():
    file_name, learn_environment = read_args()
    files = [file for file in os.listdir("results/to_learn") if file.startswith(file_name)]
    for file in files:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=config.NUM_PARALLEL_PROCESSES
        ) as executor:
            futures = [
                executor.submit(run_experiment, i, file, learn_environment)
                for i in range(config.NUM_PARALLEL_PROCESSES)
            ]
            for future in futures:
                future.result()


def main() -> None:
    """Run the program."""
    # Set up logging.
    run_experiments()


if __name__ == "__main__":
    main()
