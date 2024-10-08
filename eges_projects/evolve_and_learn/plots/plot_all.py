"""Plot fitness over generations for all experiments, averaged."""

import matplotlib.pyplot as plt
import pandas
import os
from database_components.experiment import Experiment
from database_components.generation import Generation
from database_components.individual import Individual
from database_components.population import Population
from sqlalchemy import select

from revolve2.experimentation.database import OpenMethod, open_database_sqlite

folder = "./results/0209"

def get_df(learn, evosearch, controllers, environment, survivor_select):
    database_name = f"learn-{learn}_evosearch-{evosearch}_controllers-{controllers}_select-{survivor_select}_environment-{environment}"
    print(database_name)
    files = [file for file in os.listdir(folder) if file.startswith(database_name)]
    if len(files) == 0:
        return None
    dfs = []
    i = 1
    for file_name in files:
        dbengine = open_database_sqlite(folder + "/" + file_name, open_method=OpenMethod.OPEN_IF_EXISTS)

        df_mini = pandas.read_sql(
            select(
                Experiment.id.label("experiment_id"),
                Generation.generation_index,
                Individual.fitness
            )
            .join_from(Experiment, Generation, Experiment.id == Generation.experiment_id)
            .join_from(Generation, Population, Generation.population_id == Population.id)
            .join_from(Population, Individual, Population.id == Individual.population_id),
            dbengine,
        )
        df_mini = df_mini[df_mini['experiment_id'] == 1]
        df_mini['experiment_id'] = i

        df_mini['morphologies'] = df_mini['generation_index'] + 1
        df_mini['function_evaluations'] = df_mini['generation_index'] * int(learn) * 50 + int(learn) * 50
        dfs.append(df_mini)
        i += 1
    return pandas.concat(dfs)


def plot_database(ax_thingy, x_axis, learn, environment, controllers, evosearch, survivor_select):
    max_or_mean = 'max'
    df = get_df(learn, evosearch, controllers, environment, survivor_select)

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

    learn_to_color = {
        '1': 'purple',
        '30': 'blue',
        '50': 'red',
        '100': 'green',
    }

    learn_to_label = {
        '1': 'Learn budget: 1',
        '30': 'Learn budget: 30',
        '50': 'Learn budget: 50',
        '100': 'Learn budget: 100',
    }

    ax_thingy.plot(
        agg_per_generation[x_axis],
        agg_per_generation[max_or_mean + "_fitness_mean"],
        linewidth=2,
        color=learn_to_color[learn],
        label=learn_to_label[learn]
    )
    ax_thingy.fill_between(
        agg_per_generation[x_axis],
        agg_per_generation[max_or_mean + "_fitness_mean"]
        - agg_per_generation[max_or_mean + "_fitness_std"],
        agg_per_generation[max_or_mean + "_fitness_mean"]
        + agg_per_generation[max_or_mean + "_fitness_std"],
        alpha=0.1,
        color=learn_to_color[learn],
    )

    #ax_thingy.set_ylim(0, 25)
    if x_axis == "function_evaluations":
        ax_thingy.set_xlim(0, 400000)
    else:
        ax_thingy.set_xlim(0, 135)


def main() -> None:
    fig, ax = plt.subplots(nrows=3, ncols=2, sharex='col', sharey='row')
    for i, x_axis in enumerate(["morphologies", "function_evaluations"]):
        for (learn, evosearch) in [('1', '1'), ('50', '1'), ('100', '1')]:
            for j, environment in enumerate(['flat', 'steps']):
                plot_database(ax[j][i], x_axis, learn, environment, 'adaptable', evosearch, "tournament")
                if j == 3:
                    ax[j][i].set_xlabel(x_axis.replace("_", " ").title())

    ax[0][1].legend(loc='upper left', fontsize=10)

    fig.text(0.04, 0.5, 'Fitness', va='center', rotation='vertical', fontsize=16)
    fig.text(0.08, 0.25, 'Environment: Noisy', va='center', rotation='vertical', fontsize=14)
    fig.text(0.08, 0.5, 'Environment: Steps', va='center', rotation='vertical', fontsize=14)
    fig.text(0.08, 0.75, 'Environment: Flat', va='center', rotation='vertical', fontsize=14)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


if __name__ == "__main__":
    main()
