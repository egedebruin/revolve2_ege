import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uuid

from database_components.experiment import Experiment
from database_components.generation import Generation
from database_components.genotype import Genotype
from database_components.individual import Individual
from database_components.learn_individual import LearnIndividual
from database_components.learn_genotype import LearnGenotype

import pandas

from database_components.population import Population
from revolve2.experimentation.database import open_database_sqlite, OpenMethod
from sqlalchemy import select
import seaborn as sns


def deserialize_brain(serialized_brain):
    result = []
    for value in serialized_brain.split(';'):
        if value == '':
            continue
        new_uuid, values = value.split(':')
        string_list = values.split(',')
        float_list = [float(value) for value in string_list]
        result.append(np.array(float_list))
    return result


def number_of_brains(serialized_brain):
    return len(serialized_brain.split(';'))


def get_df(learn, controllers, environment, survivor_select, inherit_samples, folder):
    database_name = f"learn-{learn}_controllers-{controllers}_survivorselect-{survivor_select}_parentselect-tournament_inheritsamples-{inherit_samples}_environment-{environment}"
    print(database_name)
    files = [file for file in os.listdir(folder) if file.startswith(database_name)]
    if len(files) == 0:
        return None
    dfs = []
    i = 1
    for file_name in files:
        # if i > 3:
        #     break
        dbengine = open_database_sqlite(folder + "/" + file_name, open_method=OpenMethod.OPEN_IF_EXISTS)
        df_mini = pandas.read_sql(
            select(
                Genotype.id.label("genotype_id"),
                LearnIndividual.objective_value,
                LearnGenotype._serialized_brain
            )
            .join_from(Genotype, LearnIndividual, Genotype.id == LearnIndividual.morphology_genotype_id)
            .join_from(LearnIndividual, LearnGenotype, LearnIndividual.genotype_id == LearnGenotype.id),
            dbengine
        )
        df_mini = df_mini.loc[df_mini.groupby('genotype_id')['objective_value'].idxmax()]
        df_mini['experiment_id'] = i

        threshold = df_mini['objective_value'].quantile(0.9)
        df_mini = df_mini[df_mini['objective_value'] >= threshold].copy()

        dfs.append(df_mini)
        i += 1
    return pandas.concat(dfs)


def to_correct_df(df):
    df['brain'] = df['serialized_brain'].apply(lambda x: deserialize_brain(x))
    flatten_brain = [item for sublist in df['brain'] for item in sublist]
    expanded_df = pd.DataFrame(flatten_brain, columns=['0', '1', '2']).reset_index(drop=True)
    return expanded_df

def main() -> None:
    fig, ax = plt.subplots(nrows=4, ncols=2, sharex=True, sharey=True, figsize=(8, 10))  # Increase figure size
    folder = "./results/new_big/cpg"
    environments = ['flat', 'noisy', 'hills', 'steps']
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Custom colors for better contrast

    all_dfs = []
    for j, environment in enumerate(environments):
        what_to_plot = []
        for inherit_samples in ['-1', '0', '5']:
            df = get_df('30', 'adaptable', environment, 'newest', inherit_samples, folder)

            df = to_correct_df(df)
            result_df = df.melt()
            result_df['min_distance'] = result_df[['value']].apply(lambda x: min(x.iloc[0], 1 - x.iloc[0]), axis=1)

            bins = np.arange(0, 0.52, 0.02)
            labels = [f"{bins[i]:.2f}-{bins[i + 1]:.2f}" for i in range(len(bins) - 1)]

            result_df['bin'] = pd.cut(result_df['min_distance'], bins=bins, right=False, labels=labels)
            real_result_df = pd.get_dummies(result_df['bin']).sum().to_frame().T
            real_result_df['environment'] = environment
            real_result_df['inherit_samples'] = inherit_samples
            all_dfs.append(real_result_df)
            continue
            what_to_plot.append(list(result_df['min_distance']))

        # sns.violinplot(data=what_to_plot, ax=ax[j][0], bw_adjust=0.2, palette=colors, linewidth=1.2, inner="quartile")
        # sns.violinplot(data=what_to_plot_best, ax=ax[j][1], bw_adjust=0.2, palette=colors, linewidth=1.2, inner="quartile")
        #
        # ax[j][0].set_xticks([0, 1, 2])
        # ax[j][1].set_xticks([0, 1, 2])
        # ax[j][0].set_xticklabels(['No inheritance', 'Inherit samples', 'Reevaluate'], fontsize=10)
        # ax[j][1].set_xticklabels(['No inheritance', 'Inherit samples', 'Reevaluate'], fontsize=10)
        # ax[j][0].set_title(f"{environment.capitalize()}", fontsize=12, fontweight="bold")
        # ax[j][0].grid(True, linestyle="--", alpha=0.6)
        # ax[j][1].grid(True, linestyle="--", alpha=0.6)

    # fig.supylabel("Min Distance", fontsize=14, fontweight="bold")
    # fig.supxlabel("Inherit Samples", fontsize=14, fontweight="bold")
    # fig.tight_layout(pad=2)
    # plt.show()
    pd.concat(all_dfs).to_csv('results/cpg-parameters-in-bins-best-robots.csv')



if __name__ == "__main__":
    main()
