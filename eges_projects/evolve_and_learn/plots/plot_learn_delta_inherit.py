from database_components.generation import Generation
from database_components.genotype import Genotype
from database_components.individual import Individual
from database_components.learn_individual import LearnIndividual
from database_components.population import Population
from randomly_sample import RandomSample
from revolve2.experimentation.database import open_database_sqlite, OpenMethod
from sqlalchemy import select
import pandas as pd
import matplotlib.pyplot as plt

def get_df(environment, inherit_samples, repetition):
    database_name = f"learn-30_controllers-adaptable_survivorselect-newest_parentselect-tournament_inheritsamples-{inherit_samples}_environment-{environment}_{repetition}.sqlite"
    dbengine = open_database_sqlite('results/new_big/' + database_name, open_method=OpenMethod.OPEN_IF_EXISTS)

    df = pd.read_sql(
        select(
            Generation.generation_index,
            Genotype.id.label("genotype_id"),
            LearnIndividual.generation_index.label('learn_generation_index'),
            LearnIndividual.objective_value
        )
        .join_from(Generation, Population, Generation.population_id == Population.id)
        .join_from(Population, Individual, Population.id == Individual.population_id)
        .join_from(Individual, Genotype, Individual.genotype_id == Genotype.id)
        .join_from(Genotype, LearnIndividual, Genotype.id == LearnIndividual.morphology_genotype_id)
        .where(Generation.generation_index <= 501),
        dbengine,
    )

    return df

def plot_database(ax, environment, inherit_samples):
    """Plot the learning delta across generations for different inheritance settings."""

    max_or_mean = "mean"  # Can be changed to "max" if needed
    dfs = []

    for repetition in range(1, 21):
        print(repetition)
        if environment == 'noisy' and inherit_samples == '5' and repetition == 4:
            continue  # Skipping a known exception

        df_mini = get_df(environment, inherit_samples, repetition)
        df_mini_early = df_mini.loc[df_mini['learn_generation_index'] < 5]
        df_mini_late = df_mini.loc[df_mini['learn_generation_index'] >= 5]

        df_mini_early = df_mini_early.loc[df_mini_early.groupby('genotype_id')['objective_value'].idxmax()]
        df_mini_late = df_mini_late.loc[df_mini_late.groupby('genotype_id')['objective_value'].idxmax()]

        df_merged = df_mini_early.merge(df_mini_late, on='genotype_id', suffixes=('_1', '_2'))

        # Subtract column B values and keep relevant columns
        df_result = df_merged[['genotype_id', 'generation_index_1']].rename(columns={'generation_index_1': 'generation_index'})
        df_result['learn_delta'] = df_merged['objective_value_2'] - df_merged['objective_value_1']

        df_result['experiment_id'] = repetition
        dfs.append(df_result)

    df = pd.concat(dfs)

    # Aggregation per experiment per generation
    agg_per_experiment_per_generation = (
        df.groupby(["experiment_id", 'generation_index'])
        .agg({"learn_delta": ["mean"]})
        .reset_index()
    )
    agg_per_experiment_per_generation.columns = [
        "experiment_id", "generation_index", "mean_learn_delta"
    ]

    # Final aggregation over all experiments
    agg_per_generation = (
        agg_per_experiment_per_generation.groupby("generation_index")
        .agg({"mean_learn_delta": ["mean", "std"]})
        .reset_index()
    )
    agg_per_generation.columns = [
        "generation_index", "mean_learn_delta_mean", "mean_learn_delta_std"
    ]

    # Colors and labels for different inheritance settings
    color_map = {'-1': '#66c2a5', '0': '#fc8d62', '5': '#8da0cb'}
    label_map = {
        '-1': 'No Inheritance',
        '0': 'Inherit Samples',
        '5': 'Redo Samples',
    }

    ax.plot(
        agg_per_generation["generation_index"],
        agg_per_generation[max_or_mean + "_learn_delta_mean"],
        linewidth=2,
        color=color_map[inherit_samples],
        label=label_map[inherit_samples],
    )
    ax.fill_between(
        agg_per_generation["generation_index"],
        agg_per_generation[max_or_mean + "_learn_delta_mean"]
        - agg_per_generation[max_or_mean + "_learn_delta_std"],
        agg_per_generation[max_or_mean + "_learn_delta_mean"]
        + agg_per_generation[max_or_mean + "_learn_delta_std"],
        color=color_map[inherit_samples],
        alpha=0.2,  # Softer transparency for better readability
    )

    # Improving aesthetics
    if environment == 'steps':
        ax.set_xlabel("Generation Index", fontsize=12)
    ax.set_ylabel("Mean Learning Delta", fontsize=12)
    if environment == 'flat':
        ax.set_title("Learning Progress Over Generations", fontsize=14, fontweight="bold")
        ax.legend(title="Inheritance Type", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)  # Light grid for readability


def main():
    fig, ax = plt.subplots(nrows=4, sharex=True)
    for inherit_samples in ['5']:
        for i, environment in enumerate(['flat', 'noisy', 'hills', 'steps']):
            print(environment)
            plot_database(ax[i], environment, inherit_samples)
    plt.show()