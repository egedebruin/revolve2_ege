"""Plot fitness over generations for all experiments, averaged."""
import matplotlib.pyplot as plt
import numpy as np
import pandas
import os
import re

from matplotlib.cm import get_cmap
from numpy.core.defchararray import startswith

from database_components.experiment import Experiment
from database_components.generation import Generation
from database_components.genotype import Genotype
from database_components.individual import Individual
from database_components.population import Population
from sqlalchemy import select

from revolve2.experimentation.database import OpenMethod, open_database_sqlite


def get_df(folder_name):
    files = [file for file in os.listdir("../evolve_and_learn/results/1501/" + folder_name)]
    if len(files) == 0:
        return None
    dfs = []
    i = 1
    for file_name in files:
        dbengine = open_database_sqlite("../evolve_and_learn/results/1501/" + folder_name + "/" + file_name, open_method=OpenMethod.OPEN_IF_EXISTS)

        df_mini = pandas.read_sql(
            select(
                Experiment.id.label("experiment_id"),
                Generation.generation_index,
                Individual.fitness,
                Genotype._serialized_brain
            )
            .join_from(Experiment, Generation, Experiment.id == Generation.experiment_id)
            .join_from(Generation, Population, Generation.population_id == Population.id)
            .join_from(Population, Individual, Population.id == Individual.population_id)
            .join_from(Individual, Genotype, Individual.genotype_id == Genotype.id),
            dbengine,
        )
        if df_mini.empty or max(df_mini['generation_index']) < 2:
            continue
        df_mini = df_mini[df_mini['experiment_id'] == 1]
        df_mini['experiment_id'] = i
        dfs.append(df_mini)
        i += 1
    return pandas.concat(dfs)


def plot_database():
    folder_to_label = {
        "after_learn_random": "Random robots",
        "after_learn_best": "Best robots, random first sample",
        "after_learn_best_kickstart": "Best robots, evolutionary first sample",
    }
    x_value = 30
    values = []
    cmap = get_cmap('viridis')
    for folder_name in ["learn"]:
        full_df = get_df(folder_name)
        full_df['controllers'] = full_df['serialized_brain'].apply(lambda x: len(x.split(";")))
        grouped_df = full_df.groupby('controllers')

        for name, df in grouped_df:
            # Calculate the max fitness per generation index within each experiment
            df['max_fitness'] = df.groupby(['experiment_id', 'generation_index'])['fitness'].transform('max').groupby(
                df['experiment_id']).cummax()

            # Calculate the max and min of the max_fitness for each experiment_id
            max_fitness = df.groupby('experiment_id')['max_fitness'].transform('max')

            # Scale the max_fitness values
            df['scaled_max_fitness'] = (df['max_fitness'] - 0) / (max_fitness - 0)

            agg_per_generation = (
                df.groupby(["generation_index"])
                .agg({"scaled_max_fitness": ["mean"]})
                .reset_index()
            )

            agg_per_generation.columns = [
                "generation_index",
                "mean_scaled_fitness",
            ]

            plt.plot(agg_per_generation['generation_index'], agg_per_generation['mean_scaled_fitness'], label=name, color=cmap((float(name) - 2) / 11))
            values.append(np.interp(x_value, agg_per_generation['generation_index'], agg_per_generation['mean_scaled_fitness']))

    # plt.xlim(-10, 510)
    # plt.ylim(-0.1, 1.1)
    plt.xlabel("Samples", fontsize=12)
    plt.ylabel("Relative performance", fontsize=12)
    plt.legend()
    plt.show()


def main():
    plot_database()


if __name__ == "__main__":
    main()