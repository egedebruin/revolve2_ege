import pandas
import json
import os
from revolve2.experimentation.database import open_database_sqlite, OpenMethod
from sqlalchemy import select

from body_genotype_direct import CoreGenotype
from experiment import Experiment
from generation import Generation
from genotype import Genotype
from individual import Individual
from population import Population

from scipy.stats import wilcoxon

import matplotlib.pyplot as plt


def calculate_number_of_controllers(serialized_body):
    return len(CoreGenotype(0.0).deserialize(json.loads(serialized_body)).check_for_brains())


def calculate_number_of_modules(serialized_body):
    return CoreGenotype(0.0).deserialize(json.loads(serialized_body)).get_amount_nodes()


def calculate_number_of_hinges(serialized_body):
    return CoreGenotype(0.0).deserialize(json.loads(serialized_body)).get_amount_hinges()


def get_dfs_normal(database_name, generation_index):
    dfs = []
    for i in range(1, 6):
        dbengine = open_database_sqlite("results/" + database_name + "_" + str(i) + ".sqlite",
                                        open_method=OpenMethod.OPEN_IF_EXISTS)

        df_mini = pandas.read_sql(
            select(
                Experiment.id.label("experiment_id"),
                Generation.generation_index,
                Individual.fitness,
                Genotype._serialized_body
            )
            .join_from(Experiment, Generation, Experiment.id == Generation.experiment_id)
            .join_from(Generation, Population, Generation.population_id == Population.id)
            .join_from(Population, Individual, Population.id == Individual.population_id)
            .join_from(Individual, Genotype, Individual.genotype_id == Genotype.id)
            .where(Generation.generation_index == generation_index),
            dbengine,
        )
        df_mini = df_mini[df_mini['experiment_id'] == 1]

        df_mini['experiment_id'] = i
        dfs.append(df_mini)
    return pandas.concat(dfs)


def get_dfs_short_learn(database_name, generation_index):
    files = [file for file in os.listdir("results") if file.startswith(database_name)]
    dfs = []
    i = 0
    for file_name in files:
        dbengine = open_database_sqlite("results/" + file_name, open_method=OpenMethod.OPEN_IF_EXISTS)
        df_mini = pandas.read_sql(
            select(
                Experiment.id.label("experiment_id"),
                Generation.generation_index,
                Individual.fitness,
                Genotype._serialized_body
            )
            .join_from(Experiment, Generation, Experiment.id == Generation.experiment_id)
            .join_from(Generation, Population, Generation.population_id == Population.id)
            .join_from(Population, Individual, Population.id == Individual.population_id)
            .join_from(Individual, Genotype, Individual.genotype_id == Genotype.id)
            .where((Generation.generation_index == 9995) | (Generation.generation_index == generation_index)),
            dbengine,
        )

        df_final_generation = df_mini[df_mini['generation_index'] == 9995]
        df_mini = df_mini[df_mini['experiment_id'].isin(df_final_generation['experiment_id'].unique())]
        df_mini = df_mini[df_mini['generation_index'] == generation_index]

        df_mini['experiment_id'] += i
        i += len(df_final_generation['experiment_id'].unique())
        dfs.append(df_mini)
    return pandas.concat(dfs)


def get_df(database_name: str):
    if "learn-1" in database_name:
        df = get_dfs_short_learn(database_name, 9995)
    else:
        if "learn-50" in database_name:
            if database_name == 'learn-50_controllers-1_hills':
                df = get_dfs_normal(database_name, 195)
            else:
                df = get_dfs_normal(database_name, 195)
        else:
            df = get_dfs_normal(database_name, 329)

    df['controllers'] = df['serialized_body'].apply(lambda x: calculate_number_of_controllers(x))
    df['modules'] = df['serialized_body'].apply(lambda x: calculate_number_of_modules(x))
    df['hinges'] = df['serialized_body'].apply(lambda x: calculate_number_of_hinges(x))
    df = df.drop(columns=['serialized_body'])

    agg_per_experiment_per_generation = (
        df.groupby(["experiment_id", "generation_index"])
        .agg({"fitness": ["mean", "max"], "controllers": ["mean"], "modules": ["mean"], "hinges": ["mean"]})
        .reset_index()
    )
    agg_per_experiment_per_generation.columns = [
        "experiment_id",
        "generation_index",
        "mean_fitness",
        "max_fitness",
        "mean_controllers",
        "mean_modules",
        "mean_hinges",
    ]

    return agg_per_experiment_per_generation


def make_boxplot(df, category, ax_thingy):
    value_columns = ['mean_fitness']
    categorical_columns = [category]

    grouped = df.groupby(categorical_columns, observed=True)

    for value_column in value_columns:
        box_data = []  # List to hold data for boxplot
        labels = []  # List to hold labels for x-axis
        # Iterate over groups
        for name, group in grouped:
            box_data.append(group[value_column].values)
            if category == "learn":
                labels.append("Learn budget: " + '_'.join(str(x) for x in name))
            else:
                labels.append("Controllers: " + '_'.join(str(x) for x in name))
        # Create boxplot
        ax_thingy.boxplot(box_data, labels=labels)
        # Rotate x-axis labels for better readability


def main() -> None:
    dfs = []
    for learn in ['1', '30', '50']:
        for controllers in ['1', '4', '8']:
            for environment in ['flat', 'hills']:
                current_result = get_df('learn-' + learn + "_controllers-" + controllers + "_" + environment)
                current_result['learn'] = learn
                current_result['controllers'] = controllers
                current_result['environment'] = environment
                dfs.append(current_result)
    df = pandas.concat(dfs)
    df['learn'] = df['learn'].astype('category')
    df['controllers'] = df['controllers'].astype('category')
    df['environment'] = df['environment'].astype('category')
    df = df.drop(columns=['generation_index'])
    df_flat = df[df['environment'] == 'flat']
    # df_hills = df[df['environment'] == 'hills']
    #
    # df_1_learn = df_hills[df_hills['learn'] == '1']
    # df_30_learn = df_hills[df_hills['learn'] == '30']
    # df_50_learn = df_hills[df_hills['learn'] == '50']
    #
    # df_1_controller = df_hills[df_hills['controllers'] == '1']
    # df_4_controller = df_hills[df_hills['controllers'] == '4']
    # df_8_controller = df_hills[df_hills['controllers'] == '8']

    # statistic = wilcoxon(list(df_1_learn['mean_fitness']), list(df_30_learn['mean_fitness']))
    # print("Learn: 1 vs 30")
    # print(statistic)
    # print()
    # statistic = wilcoxon(list(df_1_learn['mean_fitness']), list(df_50_learn['mean_fitness']))
    # print("Learn: 1 vs 50")
    # print(statistic)
    # print()
    # statistic = wilcoxon(list(df_30_learn['mean_fitness']), list(df_50_learn['mean_fitness']))
    # print("Learn: 30 vs 50")
    # print(statistic)
    # print()
    #
    # statistic = wilcoxon(list(df_1_controller['mean_fitness']), list(df_4_controller['mean_fitness']))
    # print("Controllers: 1 vs 4")
    # print(statistic)
    # print()
    # statistic = wilcoxon(list(df_1_controller['mean_fitness']), list(df_8_controller['mean_fitness']))
    # print("Controllers: 1 vs 8")
    # print(statistic)
    # print()
    # statistic = wilcoxon(list(df_4_controller['mean_fitness']), list(df_8_controller['mean_fitness']))
    # print("Controllers: 4 vs 8")
    # print(statistic)
    # print()

    fig, ax = plt.subplots(2, 1, sharex='col', sharey='row')
    make_boxplot(df_flat, 'learn', ax[0])
    # make_boxplot(df_hills, 'learn', ax[1])
    fig.text(0.04, 0.5, 'Mean fitness', va='center', rotation='vertical', fontsize=16)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax[0].text(0.05, 0.9, "Flat environment", transform=ax[0].transAxes,
                  fontsize=10,
                  bbox=props)
    # ax[1].text(0.05, 0.9, "Hills environment", transform=ax[1].transAxes,
    #               fontsize=10,
    #               bbox=props)

    plt.subplots_adjust(wspace=0, hspace=0.05)
    plt.show()


if __name__ == "__main__":
    main()
