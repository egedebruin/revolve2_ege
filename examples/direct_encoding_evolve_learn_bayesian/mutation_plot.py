import os

from experiment import Experiment
from generation import Generation
from individual import Individual
from genotype import Genotype
from population import Population

import pandas
import matplotlib.pyplot as plt

from revolve2.experimentation.database import open_database_sqlite, OpenMethod
from sqlalchemy import select


def categorize_mutation(mutation_parameter):
    if mutation_parameter < 0.25:
        return 'Add'
    elif 0.25 <= mutation_parameter < 0.5:
        return 'Remove'
    elif 0.5 <= mutation_parameter < 0.75:
        return 'Switch'
    else:
        return 'Reverse'


def get_df(learn, evosearch, controllers, environment, survivor_select):
    database_name = f"learn-{learn}_evosearch-{evosearch}_controllers-{controllers}_select-{survivor_select}_environment-{environment}"
    print(database_name)
    files = ['test.sqlite']
    if len(files) == 0:
        return None
    dfs = []
    i = 1
    for file_name in files:
        dbengine = open_database_sqlite(file_name, open_method=OpenMethod.OPEN_IF_EXISTS)

        df_mini = pandas.read_sql(
            select(
                Experiment.id.label("experiment_id"),
                Generation.generation_index,
                Genotype.id.label("genotype_id"),
                Genotype.parent_1_genotype_id,
                Genotype.parent_2_genotype_id,
                Genotype.mutation_parameter,
                Individual.fitness
            )
            .join_from(Experiment, Generation, Experiment.id == Generation.experiment_id)
            .join_from(Generation, Population, Generation.population_id == Population.id)
            .join_from(Population, Individual, Population.id == Individual.population_id)
            .join_from(Individual, Genotype, Individual.genotype_id == Genotype.id),
            dbengine,
        )
        df_mini = df_mini[df_mini['experiment_id'] == 1]
        df_mini['experiment_id'] = i

        df_mini['morphologies'] = df_mini['generation_index'] * 10 + 50
        df_mini['function_evaluations'] = df_mini['generation_index'] * int(learn) * 10 + int(learn) * 50
        mapping_dict = dict(zip(df_mini['genotype_id'], df_mini['fitness']))
        df_mini['parents_fitness'] = (df_mini['parent_1_genotype_id'].map(mapping_dict) + df_mini['parent_2_genotype_id'].map(mapping_dict)) / 2
        dfs.append(df_mini)
        i += 1
    return pandas.concat(dfs)


def do_thingy(df):
    for mutation, group in df.groupby('mutation'):
        x = []
        y = []
        for generation, group_2 in group.groupby('generation_index'):
            x.append(generation)
            y.append((group_2['fitness'] - group_2['parents_fitness']).max())
        plt.plot(x, y, label=mutation)
    plt.legend()
    plt.show()


def lets_go_mutation_plot_exclamation_mark(df, ax_thingy):
    mutation_counts = df.groupby(['function_evaluations', 'mutation']).size().reset_index(name='count')
    total_counts = mutation_counts.groupby('function_evaluations')['count'].sum().reset_index()
    mutation_counts = pandas.merge(mutation_counts, total_counts, on='function_evaluations', suffixes=('', '_total'))
    mutation_counts['proportion'] = mutation_counts['count'] / mutation_counts['count_total']
    mutation_proportions_pivot = mutation_counts.pivot_table(index='function_evaluations', columns='mutation',
                                                             values='proportion', fill_value=0)
    mutation_proportions_pivot.reset_index(inplace=True)

    ax_thingy.plot(mutation_proportions_pivot['function_evaluations'], mutation_proportions_pivot['Add'])
    ax_thingy.plot(mutation_proportions_pivot['function_evaluations'], mutation_proportions_pivot['Remove'])
    ax_thingy.plot(mutation_proportions_pivot['function_evaluations'], mutation_proportions_pivot['Switch'])
    ax_thingy.plot(mutation_proportions_pivot['function_evaluations'], mutation_proportions_pivot['Reverse'])
    ax_thingy.axhline(y=0.25, color='black', linestyle='--')
    ax_thingy.axis(ymin=0, ymax=1)


def plot_mutation(df, ax_thingy):
    df = df.copy()
    df = df[df['mutation_parameter'] > 0]
    df['mutation'] = df['mutation_parameter'].apply(categorize_mutation)
    df.drop(columns=['mutation_parameter'], inplace=True)

    lets_go_mutation_plot_exclamation_mark(df, ax_thingy)


def main() -> None:
    environment = 'noisy'
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex='col')
    for controllers in ['adaptable']:
        i = 0
        for learn, evosearch in [('10', '0')]:
            df = get_df(learn, evosearch, controllers, environment, 'tournament')
            if df is None:
                continue
            plot_mutation(df, ax[i])
            i += 1
    ax[0].legend(['Add', 'Remove', 'Switch', 'Reverse'], loc='upper left')
    plt.show()


if __name__ == "__main__":
    main()
