"""Plot fitness over generations for all experiments, averaged."""
import json

import config
import matplotlib.pyplot as plt
import pandas

from body_genotype_direct import CoreGenotype
from experiment import Experiment
from generation import Generation
from genotype import Genotype
from individual import Individual
from population import Population
from sqlalchemy import select

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.logging import setup_logging


def calculate_value(serialized_body):
    return len(CoreGenotype(0.0).deserialize(json.loads(serialized_body)).check_for_brains())


def main() -> None:
    """Run the program."""
    setup_logging()

    dbengine = open_database_sqlite(
        config.DATABASE_FILE, open_method=OpenMethod.OPEN_IF_EXISTS
    )

    df = pandas.read_sql(
        select(
            Experiment.id.label("experiment_id"),
            Generation.generation_index,
            Individual.fitness,
            Genotype._serialized_body
        )
        .join_from(Experiment, Generation, Experiment.id == Generation.experiment_id)
        .join_from(Generation, Population, Generation.population_id == Population.id)
        .join_from(Population, Individual, Population.id == Individual.population_id)
        .join_from(Individual, Genotype, Individual.genotype_id == Genotype.id),
        dbengine,
    )

    df['controllers'] = df['serialized_body'].apply(lambda x: calculate_value(x))

    agg_per_experiment_per_generation = (
        df.groupby(["experiment_id", "generation_index"])
        .agg({"controllers": ["max", "mean"]})
        .reset_index()
    )
    agg_per_experiment_per_generation.columns = [
        "experiment_id",
        "generation_index",
        "max_controllers",
        "mean_controllers",
    ]

    agg_per_generation = (
        agg_per_experiment_per_generation.groupby("generation_index")
        .agg({"max_controllers": ["mean", "std"], "mean_controllers": ["mean", "std"]})
        .reset_index()
    )
    agg_per_generation.columns = [
        "generation_index",
        "max_controllers_mean",
        "max_controllers_std",
        "mean_controllers_mean",
        "mean_controllers_std",
    ]

    plt.figure()

    # Plot max
    plt.plot(
        agg_per_generation["generation_index"],
        agg_per_generation["max_controllers_mean"],
        label="Max controllers",
        color="b",
    )
    plt.fill_between(
        agg_per_generation["generation_index"],
        agg_per_generation["max_controllers_mean"] - agg_per_generation["max_controllers_std"],
        agg_per_generation["max_controllers_mean"] + agg_per_generation["max_controllers_std"],
        color="b",
        alpha=0.2,
    )

    # Plot mean
    plt.plot(
        agg_per_generation["generation_index"],
        agg_per_generation["mean_controllers_mean"],
        label="Mean fitness",
        color="r",
    )
    plt.fill_between(
        agg_per_generation["generation_index"],
        agg_per_generation["mean_controllers_mean"]
        - agg_per_generation["mean_controllers_std"],
        agg_per_generation["mean_controllers_mean"]
        + agg_per_generation["mean_controllers_std"],
        color="r",
        alpha=0.2,
    )

    plt.xlabel("Generation index")
    plt.ylabel("Controllers")
    plt.title("Mean and max controllers across repetitions with std as shade")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
