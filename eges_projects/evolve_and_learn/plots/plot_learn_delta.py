from database_components.generation import Generation
from database_components.genotype import Genotype
from database_components.individual import Individual
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
            Individual.objective_value.label("fitness"),
            Genotype.id.label("genotype_id")
        )
        .join_from(Generation, Population, Generation.population_id == Population.id)
        .join_from(Population, Individual, Population.id == Individual.population_id)
        .join_from(Individual, Genotype, Genotype.id == Individual.genotype_id)
        .where(Generation.generation_index <= 501),
        dbengine,
    )

    return df

def get_best_samples(environment, inherit_samples, repetition):
    database_name = f"learn-30_controllers-adaptable_survivorselect-newest_parentselect-tournament_inheritsamples-{inherit_samples}_environment-{environment}_{repetition}.sqlite"
    dbengine = open_database_sqlite(
        'results/random/' + database_name, open_method=OpenMethod.OPEN_IF_EXISTS
    )
    samples = pd.read_sql(
        select(
            RandomSample
        ),
        dbengine,
    )
    return samples.groupby('genotype_id')['objective_value'].max()

def plot_database(ax, environment, inherit_samples):
    labels = {
        '-1': 'No inheritance', '0': 'Inherit', '5': 'Reevaluate',
        'flat': 'Flat', 'noisy': 'Rugged', 'hills': 'Hills', 'steps': 'Steps'
    }
    """Plot the learning delta across generations for different inheritance settings."""

    max_or_mean = "mean"  # Can be changed to "max" if needed
    dfs = []

    for repetition in range(1, 21):
        if environment == 'noisy' and inherit_samples == '5' and repetition == 4:
            continue  # Skipping a known exception

        df_mini = get_df(environment, inherit_samples, repetition)
        best_random_samples = get_best_samples(environment, inherit_samples, repetition)

        df_mini = df_mini.merge(best_random_samples, on='genotype_id')
        df_mini['learn_delta'] = df_mini['fitness'] - df_mini['objective_value']
        df_mini['experiment_id'] = repetition
        dfs.append(df_mini)

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
        '5': 'Reevaluate',
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
        ax.legend(title="Inheritance Type", fontsize=10)
        ax.set_xlabel("Generation index", fontsize=14, fontweight="bold")
    ax.set_title(labels[environment], fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)  # Light grid for readability


def main():
    fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(8, 10))
    for inherit_samples in ['-1', '0', '5']:
        print(inherit_samples)
        for i, environment in enumerate(['flat', 'noisy', 'hills', 'steps']):
            print(environment)
            plot_database(ax[i], environment, inherit_samples)
    fig.text(0.08, 0.5, "Mean learning delta", va='center', rotation='vertical', fontsize=14, fontweight="bold")
    fig.text(0.32, 0.92, "Learning delta over generations", fontsize=16, fontweight="bold")
    plt.show()