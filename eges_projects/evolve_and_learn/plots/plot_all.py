"""Plot fitness over generations for all experiments, averaged."""

import matplotlib.pyplot as plt
import pandas
import os

import pandas as pd
from matplotlib.ticker import ScalarFormatter

from database_components.experiment import Experiment
from database_components.generation import Generation
from database_components.individual import Individual
from database_components.population import Population
from sqlalchemy import select

from revolve2.experimentation.database import OpenMethod, open_database_sqlite

def get_df(learn, controllers, environment, survivor_select, folder, inherit_samples):
    database_name = f"learn-{learn}_controllers-{controllers}_survivorselect-{survivor_select}_parentselect-tournament_inheritsamples-{inherit_samples}_environment-{environment}"
    print(database_name)
    files = [file for file in os.listdir(folder) if file.startswith(database_name)]
    if len(files) == 0:
        return None
    dfs = []
    i = 1
    for file_name in files:
        try:
            dbengine = open_database_sqlite(folder + "/" + file_name, open_method=OpenMethod.OPEN_IF_EXISTS)

            df_mini = pandas.read_sql(
                select(
                    Generation.generation_index,
                    Individual.objective_value.label("fitness")
                )
                .join_from(Generation, Population, Generation.population_id == Population.id)
                .join_from(Population, Individual, Population.id == Individual.population_id)
                .where(Generation.generation_index <= 500)
                ,
                dbengine,
            )
            df_mini['experiment_id'] = i
            dfs.append(df_mini)
            i += 1
        except:
            print(file_name)

    return pandas.concat(dfs)


def plot_database(ax_thingy, x_axis, learn, environment, controllers, survivor_select, folder, inherit_samples):
    max_or_mean = 'max'
    df = get_df(learn, controllers, environment, survivor_select, folder, inherit_samples)

    if df is None:
        return

    agg_per_experiment_per_generation = (
        df.groupby(["experiment_id", x_axis])
        .agg({"fitness": ["max", "mean"]})
        .reset_index()
    )
    agg_per_experiment_per_generation.columns = [
        "experiment_id",
        x_axis,
        "max_fitness",
        "mean_fitness",
    ]

    agg_per_experiment_per_generation['max_fitness'] = agg_per_experiment_per_generation.groupby(['experiment_id', x_axis])['max_fitness'].transform('max').groupby(
        agg_per_experiment_per_generation['experiment_id']).cummax()

    agg_per_generation = (
        agg_per_experiment_per_generation.groupby(x_axis)
        .agg({"max_fitness": ["mean", "std"], "mean_fitness": ["mean", "std"]})
        .reset_index()
    )
    agg_per_generation.columns = [
        x_axis,
        "max_fitness_mean",
        "max_fitness_std",
        "mean_fitness_mean",
        "mean_fitness_std",
    ]

    to_color = {
        '-1': 'red',
        '0': 'blue',
        '5': 'green',
        '-2': 'black',
        '3': 'yellow',
        '4': 'purple',
        '50': 'grey',
        'flat': 'red',
        'noisy': 'blue',
        'steps': 'green',
        'hills': 'black',
    }

    to_label = {
        '-1': 'No inheritance',
        '0': 'Inherit samples',
        '5': 'Redo samples',
        '-2': 'Inherit prior',
        'flat': 'flat',
        'noisy': 'noisy',
        'steps': 'steps',
        'hills': 'hills',
    }

    ax_thingy.plot(
        agg_per_generation[x_axis],
        agg_per_generation[max_or_mean + "_fitness_mean"],
        linewidth=2,
        color=to_color[inherit_samples],
        label=to_label[inherit_samples],
    )
    ax_thingy.fill_between(
        agg_per_generation[x_axis],
        agg_per_generation[max_or_mean + "_fitness_mean"]
        - agg_per_generation[max_or_mean + "_fitness_std"],
        agg_per_generation[max_or_mean + "_fitness_mean"]
        + agg_per_generation[max_or_mean + "_fitness_std"],
        color=to_color[inherit_samples],
        alpha=0.1,
    )

    ax_thingy.set_title(environment)

    # ax_thingy.set_ylim(0, 15)


def main() -> None:
    fig, ax = plt.subplots(nrows=4, sharex=True)
    folder = "./results/new_big/cpg"
    for inherit_samples in ['-1', '5', '0']:
        for i, environment in enumerate(['flat', 'noisy', 'hills', 'steps']):
            plot_database(ax[i], 'generation_index', '30', environment, 'adaptable', 'newest', folder,
                          inherit_samples)

    ax[0].legend(loc='lower right', fontsize=10)
    plt.show()


if __name__ == "__main__":
    main()
