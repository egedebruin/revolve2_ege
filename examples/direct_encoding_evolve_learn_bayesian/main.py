"""Main script for the example."""
import concurrent.futures
import logging

from bayes_opt import BayesianOptimization, UtilityFunction
from sklearn.gaussian_process.kernels import Matern

import config
import numpy as np
import numpy.typing as npt
from base import Base
from evaluator import Evaluator
from experiment import Experiment
from generation import Generation
from genotype import Genotype
from individual import Individual
from population import Population
from learn_genotype import LearnGenotype
from learn_individual import LearnIndividual
from learn_population import LearnPopulation
from learn_generation import LearnGeneration
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy import select

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.optimization.ea import population_management, selection
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


def select_parents(
        rng: np.random.Generator,
        population: Population,
        offspring_size: int,
) -> npt.NDArray[np.float_]:
    """
    Select pairs of parents using a tournament.

    :param rng: Random number generator.
    :param population: The population to select from.
    :param offspring_size: The number of parent pairs to select.
    :returns: Pairs of indices of selected parents. offspring_size x 2 ints.
    """
    return np.array(
        [
            selection.multiple_unique(
                2,
                [individual.genotype for individual in population.individuals],
                [individual.fitness for individual in population.individuals],
                lambda _, fitnesses: selection.tournament(rng, fitnesses, k=4),
            )
            for _ in range(int(offspring_size))
        ],
    )


def select_parent(
        rng: np.random.Generator,
        population: Population,
        offspring_size: int,
) -> npt.NDArray[np.float_]:
    """
    Select pairs of parents using a tournament.

    :param rng: Random number generator.
    :param population: The population to select from.
    :param offspring_size: The number of parent pairs to select.
    :returns: Pairs of indices of selected parents. offspring_size x 2 ints.
    """
    return np.array(
        [
            selection.multiple_unique(
                1,
                [individual.genotype for individual in population.individuals],
                [individual.fitness for individual in population.individuals],
                lambda _, fitnesses: selection.tournament(rng, fitnesses, k=4),
            )
            for _ in range(int(offspring_size))
        ],
    )


def select_survivors(
    rng: np.random.Generator,
    original_population: Population,
    offspring_population: Population,
) -> Population:
    """
    Select survivors using a tournament.

    :param rng: Random number generator.
    :param original_population: The population the parents come from.
    :param offspring_population: The offspring.
    :returns: A newly created population.
    """
    original_survivors, offspring_survivors = population_management.steady_state(
        [i.genotype for i in original_population.individuals],
        [i.fitness for i in original_population.individuals],
        [i.genotype for i in offspring_population.individuals],
        [i.fitness for i in offspring_population.individuals],
        lambda n, genotypes, fitnesses: selection.multiple_unique(
            n,
            genotypes,
            fitnesses,
            lambda _, fitnesses: selection.tournament(rng, fitnesses, k=4),
        ),
    )

    return Population(
        individuals=[
            Individual(
                genotype=original_population.individuals[i].genotype,
                fitness=original_population.individuals[i].fitness,
            )
            for i in original_survivors
        ]
        + [
            Individual(
                genotype=offspring_population.individuals[i].genotype,
                fitness=offspring_population.individuals[i].fitness,
            )
            for i in offspring_survivors
        ]
    )


def find_best_robot(
        current_best: Individual | None, population: list[Individual]
) -> Individual:
    """
    Return the best robot between the population and the current best individual.

    :param current_best: The current best individual.
    :param population: The population.
    :returns: The best individual.
    """
    return max(
        population + [] if current_best is None else [current_best],
        key=lambda x: x.fitness,
    )


def initial_genotypes_from_previous_database(rng) -> list[Genotype]:
    dbengine = open_database_sqlite(
        config.PREVIOUS_DATABASE_FILE, open_method=OpenMethod.OPEN_IF_EXISTS
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

    result = [genotype]
    for _ in range(config.POPULATION_SIZE - 1):
        child_genotype = genotype.mutate(rng)
        result.append(child_genotype)
    return result


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

    if config.INITIAL_POPULATION_FROM_DATABASE:
        initial_genotypes = initial_genotypes_from_previous_database(rng)
    else:
        initial_genotypes = [
            Genotype.initialize(
                rng=rng,
            )
            for _ in range(config.POPULATION_SIZE)
        ]

    # Evaluate the initial population.
    logging.info("Evaluating initial population.")

    initial_fitnesses = learn_population(genotypes=initial_genotypes, evaluator=evaluator, dbengine=dbengine, rng=rng)

    # Create a population of individuals, combining genotype with fitness.
    individuals = []
    for genotype, fitness in zip(initial_genotypes, initial_fitnesses):
        individual = Individual(genotype=genotype, fitness=fitness)
        individuals.append(individual)
    population = Population(
        individuals=individuals
    )
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

        if config.CROSSOVER:
            parents = select_parents(rng, population, config.OFFSPRING_SIZE / 2)
            # Create offspring. Two offspring per pair of parents, with a copy of one of its parents' distribution
            offspring_genotypes = []
            for parent1_i, parent2_i in parents:
                child_genotype_1, child_genotype_2 = Genotype.crossover(
                    population.individuals[parent1_i].genotype,
                    population.individuals[parent2_i].genotype,
                    rng,
                )
                offspring_genotypes.append(child_genotype_1.mutate(rng))
                offspring_genotypes.append(child_genotype_2.mutate(rng))
        else:
            parents = select_parent(rng, population, config.OFFSPRING_SIZE)
            offspring_genotypes = []
            for [parent_i] in parents:
                child_genotype = population.individuals[parent_i].genotype.mutate(rng)
                offspring_genotypes.append(child_genotype)

        # Evaluate the offspring.
        offspring_fitnesses = learn_population(genotypes=offspring_genotypes, evaluator=evaluator, dbengine=dbengine, rng=rng)

        # Make an intermediate offspring population.
        offspring_individuals = [Individual(genotype=genotype, fitness=fitness) for genotype, fitness in zip(offspring_genotypes, offspring_fitnesses)]
        # Create the next population by selecting survivors.
        population = select_survivors(
            rng,
            population,
            Population(
                individuals=offspring_individuals
            ),
        )
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
        result_fitnesses = []
        for future in futures:

            fitness, generations = future.result()
            result_fitnesses.append(fitness)

            for generation in generations:
                with Session(dbengine, expire_on_commit=False) as session:
                    session.add(generation)
                    session.commit()
    return result_fitnesses


def learn_genotype(genotype, evaluator, rng):
    pbounds = {}

    for key in genotype.brain.keys():
        pbounds['amplitude_' + str(key)] = [0, 1]
        pbounds['phase_' + str(key)] = [0, 1]
        pbounds['touch_weight_' + str(key)] = [0, 1]
        pbounds['neighbour_touch_weight_' + str(key)] = [0, 1]
        pbounds['sensor_phase_offset_' + str(key)] = [0, 1]

    optimizer = BayesianOptimization(
        f=None,
        pbounds=pbounds,
        allow_duplicate_points=True,
    )
    optimizer.set_gp_params(alpha=config.ALPHA, kernel=Matern(nu=config.NU, length_scale=config.LENGTH_SCALE, length_scale_bounds=(config.LENGTH_SCALE - 0.01, config.LENGTH_SCALE + 0.01)))
    utility = UtilityFunction(kind="ucb", kappa=config.KAPPA)

    best_fitness = 0
    best_learn_genotype = None
    generations = []
    lhs = latin_hypercube(config.NUM_RANDOM_SAMPLES, 5 * len(genotype.brain.keys()), rng)
    best_point = {}
    for i in range(config.LEARN_NUM_GENERATIONS + config.NUM_RANDOM_SAMPLES):
        logging.info(f"Learn generation {i + 1} / {config.LEARN_NUM_GENERATIONS + config.NUM_RANDOM_SAMPLES}.")
        if i < config.NUM_RANDOM_SAMPLES:
            j = 0
            next_point = {}
            for key in genotype.brain.keys():
                next_point['amplitude_' + str(key)] = lhs[i][j]
                next_point['phase_' + str(key)] = lhs[i][j + 1]
                next_point['touch_weight_' + str(key)] = lhs[i][j + 2]
                next_point['neighbour_touch_weight_' + str(key)] = lhs[i][j + 3]
                next_point['sensor_phase_offset_' + str(key)] = lhs[i][j + 4]
                j += 5
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
        for brain_uuid in genotype.brain.keys():
            new_learn_genotype.brain[brain_uuid] = np.array(
                [next_point['amplitude_' + str(brain_uuid)],
                 next_point['phase_' + str(brain_uuid)],
                 next_point['touch_weight_' + str(brain_uuid)],
                 next_point['neighbour_touch_weight_' + str(brain_uuid)],
                 next_point['sensor_phase_offset_' + str(brain_uuid)]]
            )
        robot = new_learn_genotype.develop()

        # Evaluate them.
        fitness = evaluator.evaluate(robot)

        if fitness >= best_fitness:
            best_fitness = fitness
            best_learn_genotype = new_learn_genotype
            best_point = next_point

        optimizer.register(params=next_point, target=fitness)

        # From the samples and fitnesses, create a population that we can save.
        population = LearnPopulation(
            individuals=[
                LearnIndividual(genotype=new_learn_genotype, fitness=fitness)
            ]
        )
        # Make it all into a generation and save it to the database.
        generation = LearnGeneration(
            genotype=genotype,
            generation_index=i,
            learn_population=population,
        )
        generations.append(generation)
    genotype.brain = best_learn_genotype.brain
    return best_fitness, generations


def main() -> None:
    """Run the program."""
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
