"""Main script for the example."""
import concurrent.futures
import logging
import time
from argparse import ArgumentParser

import numpy as np
from bayes_opt import BayesianOptimization, UtilityFunction
from sklearn.gaussian_process.kernels import Matern
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

import config
import selection
from database_components.base import Base
from database_components.experiment import Experiment
from database_components.generation import Generation
from database_components.genotype import Genotype
from database_components.individual import Individual
from database_components.learn_generation import LearnGeneration
from database_components.learn_genotype import LearnGenotype
from database_components.learn_individual import LearnIndividual
from database_components.learn_population import LearnPopulation
from database_components.population import Population
from evaluator import Evaluator
from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.rng import make_rng, seed_from_time


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


def run_experiment(dbengine: Engine) -> None:
    """
    Run an experiment.

    :param dbengine: An openened database with matching initialize database structure.
    """
    logging.info("----------------")
    logging.info("Start experiment")

    # Set up the random number generator.
    rng_seed = seed_from_time() % 2**32
    rng = make_rng(rng_seed)

    # Create and save the experiment instance.
    experiment = Experiment(rng_seed=rng_seed)
    logging.info("Saving experiment configuration.")
    with Session(dbengine) as session:
        session.add(experiment)
        session.commit()

    # Initialize the evaluator that will be used to evaluate robots.
    evaluator = Evaluator(headless=True, num_simulators=config.NUM_SIMULATORS)

    # Create an initial population.
    logging.info("Generating initial population.")

    initial_genotypes = [
        Genotype.initialize(
            rng=rng,
        )
        for _ in range(config.POPULATION_SIZE)
    ]

    # Evaluate the initial population.
    logging.info("Evaluating initial population.")

    initial_objective_values, initial_genotypes = learn_population(genotypes=initial_genotypes, evaluator=evaluator, dbengine=dbengine, rng=rng)

    # Create a population of individuals, combining genotype with fitness.
    individuals = []
    for genotype, objective_value in zip(initial_genotypes, initial_objective_values):
        individual = Individual(genotype=genotype, objective_value=objective_value, original_generation=0)
        individuals.append(individual)
    population = Population(
        individuals=individuals
    )
    selection.calculate_reproduction_fitness(population)
    generation = Generation(
        experiment=experiment, generation_index=0, population=population
    )
    logging.info("Saving generation.")
    with Session(dbengine, expire_on_commit=False) as session:
        session.add(generation)
        session.commit()

    # Start the actual optimization process.
    logging.info("Start optimization process.")
    while generation.generation_index < config.NUM_GENERATIONS:
        logging.info(
            f"Real generation {generation.generation_index + 1} / {config.NUM_GENERATIONS}."
        )

        offspring_genotypes = selection.generate_offspring(rng, population)

        # Evaluate the offspring.
        offspring_objective_values, offspring_genotypes = learn_population(genotypes=offspring_genotypes, evaluator=evaluator, dbengine=dbengine, rng=rng)

        # Make an intermediate offspring population.
        offspring_individuals = [
            Individual(genotype=genotype, objective_value=objective_value, original_generation=generation.generation_index + 1) for
            genotype, objective_value in zip(offspring_genotypes, offspring_objective_values)]
        offspring_population = Population(
                individuals=offspring_individuals
            )
        # Create the next population by selecting survivors.
        selection.calculate_survival_fitness(population, offspring_population)
        population = selection.select_survivors(rng, population, offspring_population)
        selection.calculate_reproduction_fitness(population)

        # Make it all into a generation and save it to the database.
        generation = Generation(
            experiment=experiment,
            generation_index=generation.generation_index + 1,
            population=population,
        )
        logging.info("Saving real generation.")
        with Session(dbengine, expire_on_commit=False) as session:
            session.add(generation)
            session.commit()


def learn_population(genotypes, evaluator, dbengine, rng):
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=config.NUM_PARALLEL_PROCESSES
    ) as executor:
        futures = [
            executor.submit(learn_genotype, genotype, evaluator, rng)
            for genotype in genotypes
        ]
    result_objective_values = []
    genotypes = []
    for future in futures:

        objective_value, learn_generations = future.result()
        result_objective_values.append(objective_value)
        genotypes.append(learn_generations[0].genotype)

        for learn_generation in learn_generations:
            with Session(dbengine, expire_on_commit=False) as session:
                session.add(learn_generation)
                session.commit()
    return result_objective_values, genotypes


def learn_genotype(genotype, evaluator, rng):
    brain_uuids = genotype.body.check_for_brains()

    if len(brain_uuids) == 0:
        empty_learn_genotype = LearnGenotype(brain=genotype.brain, body=genotype.body)
        population = LearnPopulation(
            individuals=[
                LearnIndividual(genotype=empty_learn_genotype, objective_value=0)
            ]
        )
        return 0, [LearnGeneration(
            genotype=genotype,
            generation_index=0,
            learn_population=population,
        )]

    pbounds = {}
    for key in brain_uuids:
        pbounds['amplitude_' + str(key)] = [0, 1]
        pbounds['phase_' + str(key)] = [0, 1]
        # pbounds['touch_weight_' + str(key)] = [0, 1]
        # pbounds['sensor_phase_offset_' + str(key)] = [0, 1]

    optimizer = BayesianOptimization(
        f=None,
        pbounds=pbounds,
        allow_duplicate_points=True,
        random_state=int(rng.integers(low=0, high=2**32))
    )
    optimizer.set_gp_params(alpha=config.ALPHA, kernel=Matern(nu=config.NU, length_scale=config.LENGTH_SCALE, length_scale_bounds=(config.LENGTH_SCALE - 0.01, config.LENGTH_SCALE + 0.01)))
    utility = UtilityFunction(kind="ucb", kappa=config.KAPPA)

    best_objective_value = None
    best_learn_genotype = None
    learn_generations = []
    # lhs = latin_hypercube(config.NUM_RANDOM_SAMPLES, 4 * len(brain_uuids), rng)
    lhs = latin_hypercube(config.NUM_RANDOM_SAMPLES, 2 * len(brain_uuids), rng)
    best_point = {}
    for i in range(config.LEARN_NUM_GENERATIONS + config.NUM_RANDOM_SAMPLES):
        logging.info(f"Learn generation {i + 1} / {config.LEARN_NUM_GENERATIONS + config.NUM_RANDOM_SAMPLES}.")
        if i < config.NUM_RANDOM_SAMPLES:
            if config.EVOLUTIONARY_SEARCH:
                next_point = {}
                for key in brain_uuids:
                    next_point['amplitude_' + str(key)] = genotype.brain[key][0]
                    next_point['phase_' + str(key)] = genotype.brain[key][1]
                    # next_point['touch_weight_' + str(key)] = genotype.brain[key][2]
                    # next_point['sensor_phase_offset_' + str(key)] = genotype.brain[key][3]
            else:
                j = 0
                next_point = {}
                for key in brain_uuids:
                    next_point['amplitude_' + str(key)] = lhs[i][j]
                    next_point['phase_' + str(key)] = lhs[i][j + 1]
                    # next_point['touch_weight_' + str(key)] = lhs[i][j + 2]
                    # next_point['sensor_phase_offset_' + str(key)] = lhs[i][j + 3]
                    j += 1
                next_point = dict(sorted(next_point.items()))
        else:
            next_point = optimizer.suggest(utility)
            next_point = dict(sorted(next_point.items()))
            next_best = utility.utility([list(next_point.values())], optimizer._gp, 0)
            for _ in range(10000):
                possible_point = {}
                for key in best_point.keys():
                    possible_point[key] = best_point[key] + np.random.normal(0, config.NEIGHBOUR_SCALE)
                possible_point = dict(sorted(possible_point.items()))

                utility_value = utility.utility([list(possible_point.values())], optimizer._gp, 0)
                if utility_value > next_best:
                    next_best = utility_value
                    next_point = possible_point

        new_learn_genotype = LearnGenotype(brain={}, body=genotype.body)
        for brain_uuid in brain_uuids:
            new_learn_genotype.brain[brain_uuid] = np.array(
                [
                    next_point['amplitude_' + str(brain_uuid)],
                    next_point['phase_' + str(brain_uuid)],
                    # next_point['touch_weight_' + str(brain_uuid)],
                    # next_point['sensor_phase_offset_' + str(brain_uuid)]
                ]
            )
        robot = new_learn_genotype.develop()

        # Evaluate them.
        start_time = time.time()
        objective_value = evaluator.evaluate(robot)
        end_time = time.time()
        new_learn_genotype.execution_time = end_time - start_time

        if best_objective_value is None or objective_value >= best_objective_value:
            best_objective_value = objective_value
            best_learn_genotype = new_learn_genotype
            best_point = next_point

        optimizer.register(params=next_point, target=objective_value)

        # From the samples and fitnesses, create a population that we can save.
        population = LearnPopulation(
            individuals=[
                LearnIndividual(genotype=new_learn_genotype, objective_value=objective_value)
            ]
        )
        # Make it all into a generation and save it to the database.
        learn_generation = LearnGeneration(
            genotype=genotype,
            generation_index=i,
            learn_population=population,
        )
        learn_generations.append(learn_generation)

    if config.OVERWRITE_BRAIN_GENOTYPE:
        for key, value in best_learn_genotype.brain.items():
            genotype.brain[key] = value
        genotype.brain = {k: v for k, v in sorted(genotype.brain.items())}

    return best_objective_value, learn_generations


def read_args():
    # Read args
    parser = ArgumentParser()
    parser.add_argument("--learn", required=True)
    parser.add_argument("--controllers", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--repetition", required=True)
    parser.add_argument("--evosearch", required=True)
    parser.add_argument("--survivorselect", required=True)
    parser.add_argument("--parentselect", required=True)
    args = parser.parse_args()
    if args.evosearch == '1':
        config.NUM_RANDOM_SAMPLES = 1
        config.LEARN_NUM_GENERATIONS = int(args.learn) - 1
    else:
        config.NUM_RANDOM_SAMPLES = min(int(int(args.learn) / 10), 1)
        config.LEARN_NUM_GENERATIONS = int(int(args.learn) - int(args.learn) / 10)
    config.NUM_GENERATIONS = int((config.FUNCTION_EVALUATIONS / (int(args.learn) * config.OFFSPRING_SIZE)))
    config.CONTROLLERS = int(args.controllers)
    config.ENVIRONMENT = args.environment
    config.EVOLUTIONARY_SEARCH = args.evosearch == '1'
    config.SURVIVOR_SELECT_STRATEGY = args.survivorselect
    config.PARENT_SELECT_STRATEGY = args.parentselect
    controllers_string = 'adaptable' if config.CONTROLLERS == -1 else config.CONTROLLERS
    config.DATABASE_FILE = ("learn-" + str(args.learn) + "_evosearch-" + args.evosearch + "_controllers-" +
                            str(controllers_string) + "_survivorselect-" + args.survivorselect + "_parentselect-" +
                            args.parentselect + "_environment-" + args.environment + "_" + str(args.repetition) + ".sqlite")


def main() -> None:
    """Run the program."""
    if config.READ_ARGS:
        read_args()

    # Set up logging.
    setup_logging(file_name="log.txt")

    # Open the database, only if it does not already exists.
    dbengine = open_database_sqlite(
        config.DATABASE_FILE, open_method=OpenMethod.NOT_EXISTS_AND_CREATE
    )
    # Create the structure of the database.
    Base.metadata.create_all(dbengine)

    # Run the experiment several times.
    for _ in range(config.NUM_REPETITIONS):
        run_experiment(dbengine)


if __name__ == "__main__":
    main()
