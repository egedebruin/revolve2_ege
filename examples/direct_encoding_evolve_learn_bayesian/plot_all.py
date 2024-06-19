"""Plot fitness over generations for all experiments, averaged."""

import config
import matplotlib.pyplot as plt
import pandas
import os
from experiment import Experiment
from generation import Generation
from individual import Individual
from population import Population
from sqlalchemy import select

from revolve2.experimentation.database import OpenMethod, open_database_sqlite


def get_df(learn, evosearch, controllers, environment, survivor_select):
    database_name = f"learn-{learn}_evosearch-{evosearch}_controllers-{controllers}_select-{survivor_select}_environment-{environment}"
    print(database_name)
    files = [file for file in os.listdir("results/new") if file.startswith(database_name)]
    if len(files) == 0:
        return None
    dfs = []
    i = 1
    for file_name in files:
        dbengine = open_database_sqlite("results/new/" + file_name, open_method=OpenMethod.OPEN_IF_EXISTS)

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

        df_mini['morphologies'] = df_mini['generation_index'] * 10 + 50
        df_mini['function_evaluations'] = df_mini['generation_index'] * int(learn) * 10 + int(learn) * 50
        dfs.append(df_mini)
        i += 1
    return pandas.concat(dfs)


def plot_database(ax_thingy, x_axis, learn, environment, controllers, evosearch, survivor_select):
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
        '10': 'green',
        '11': 'purple',
        '300': 'blue',
        '500': 'red',
    }

    learn_to_label = {
        '11': 'Learn budget: 1, EvoSearch',
        '10': 'Learn budget: 1',
        '300': 'Learn budget: 30',
        '500': 'Learn budget: 50',
    }

    ax_thingy.plot(
        agg_per_generation[x_axis],
        agg_per_generation["max_fitness_mean"],
        linewidth=2,
        color=learn_to_color[learn+evosearch],
        label=learn_to_label[learn+evosearch]
    )
    ax_thingy.fill_between(
        agg_per_generation[x_axis],
        agg_per_generation["max_fitness_mean"]
        - agg_per_generation["max_fitness_std"],
        agg_per_generation["max_fitness_mean"]
        + agg_per_generation["max_fitness_std"],
        alpha=0.1,
        color=learn_to_color[learn+evosearch],
    )
    if x_axis == "function_evaluations":
        #ax_thingy.set_xlim(0, 150000)
        if controllers == '1':
            ax_thingy.legend(loc='upper left', fontsize=10)
        if controllers == '8':
            ax_thingy.set_xlabel("Function evaluations", fontsize=12)
    else:
        #ax_thingy.set_xlim(0, 5000)
        if controllers == '8':
            ax_thingy.set_xlabel("Morphologies evaluated", fontsize=12)
    # if environment == 'hills' or environment == 'noisy':
    #     ax_thingy.axis(ymin=0, ymax=10)
    # elif environment == 'flat':
    #     ax_thingy.axis(ymin=0, ymax=20)


def main() -> None:
    environment = 'noisy'
    fig, ax = plt.subplots(nrows=4, ncols=2, sharex='col', sharey='row')
    for i, x_axis in enumerate(["morphologies", "function_evaluations"]):
        for learn in ['1', '50']:
            for j, survivor_select in enumerate(["tournament", "newest"]):
                plot_database(ax[j][i], x_axis, learn, environment, 'adaptable', '0', survivor_select)
                if learn == '1':
                    plot_database(ax[j][i], x_axis, learn, environment, 'adaptable', '1', survivor_select)

    fig.text(0.04, 0.5, 'Fitness', va='center', rotation='vertical', fontsize=16)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


if __name__ == "__main__":
    main()
