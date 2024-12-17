import concurrent.futures

import matplotlib.pyplot as plt
import pandas
import os
import json

import pandas as pd
from matplotlib.ticker import ScalarFormatter

from database_components.experiment import Experiment
from database_components.genotype import Genotype
from revolve2.experimentation.database import OpenMethod, open_database_sqlite

from database_components.generation import Generation
from database_components.individual import Individual
from database_components.population import Population
from genotypes.body_genotype_direct import CoreGenotype
from sqlalchemy import select

def tree_edit_distance(tree1, tree2):
    # Base cases: one of the trees is empty
    if tree1 is None:
        return size(tree2)
    if tree2 is None:
        return size(tree1)

    # If the root labels are the same, no relabeling is needed
    cost = 0 if tree1.type == tree2.type else 1

    tree1_children = list(tree1.children.values())
    tree2_children = list(tree2.children.values())

    # Now we need to calculate the edit distance between the children of the trees
    m, n = len(tree1_children), len(tree2_children)

    # Create a DP table for the subtrees of tree1 and tree2
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize DP table for insertion and deletion of subtrees
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + size(tree1_children[i - 1])
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + size(tree2_children[j - 1])

    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = min(
                dp[i - 1][j] + size(tree1_children[i - 1]),  # Deleting a subtree from tree1
                dp[i][j - 1] + size(tree2_children[j - 1]),  # Inserting a subtree from tree2
                dp[i - 1][j - 1] + tree_edit_distance(tree1_children[i - 1], tree2_children[j - 1])
                # Recursively calculate TED for subtrees
            )

    # The result is the root cost (if relabeling is necessary) plus the edit distance between the children
    return cost + dp[m][n]


def size(tree):
    # Calculate the size of the tree (number of nodes)
    if tree is None:
        return 0
    return 1 + sum(size(c) for c in list(tree.children.values()))

def get_df(learn, controllers, environment, survivor_select, folder, popsize, inherit_samples):
    database_name = f"learn-{learn}_controllers-{controllers}_survivorselect-{survivor_select}_parentselect-tournament_inheritsamples-{inherit_samples}_environment-{environment}"
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
                Individual.objective_value.label("fitness"),
                Generation.generation_index,
                Genotype
            )
            .join_from(Experiment, Generation, Experiment.id == Generation.experiment_id)
            .join_from(Generation, Population, Generation.population_id == Population.id)
            .join_from(Population, Individual, Population.id == Individual.population_id)
            .join_from(Individual, Genotype, Individual.genotype_id == Genotype.id),
            dbengine,
        )
        df_mini = df_mini[df_mini['experiment_id'] == 1]
        df_mini['experiment_id'] = i
        dfs.append(df_mini)
        i += 1
    return pandas.concat(dfs)

def plot_database(learn, environment, controllers, survivor_select, folder, popsize, inherit_samples):
    df = get_df(learn, controllers, environment, survivor_select, folder, popsize, inherit_samples)

    result = {
        'generation': [],
        'experiment_id': [],
        'tree_edit_distance': [],
        'fitness': [],
        'parent_fitness': [],
        'inherit_samples': [],
        'survivor_selection': [],
    }
    experiments = df['experiment_id'].nunique()
    for experiment_id in range(1, experiments + 1):
        print(experiment_id)
        df_experiment = df.loc[df['experiment_id'] == experiment_id]
        for row in df_experiment.itertuples():
            filtered_rows = df_experiment.loc[df_experiment['id'] == row.parent_1_genotype_id]
            if not filtered_rows.empty:
                parent = filtered_rows.iloc[0]
                ted = tree_edit_distance(CoreGenotype(0.0).deserialize(json.loads(parent['serialized_body'])),
                                    CoreGenotype(0.0).deserialize(json.loads(row.serialized_body)))
                result['generation'].append(row.generation_index)
                result['experiment_id'].append(experiment_id)
                result['tree_edit_distance'].append(ted)
                result['fitness'].append(row.fitness)
                result['parent_fitness'].append(parent['fitness'])
                result['inherit_samples'].append(inherit_samples)
                result['survivor_selection'].append(survivor_select)

    return pd.DataFrame(result)

def main() -> None:
    folder, popsize = ("./results/2511", 20)
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=3
    ) as executor:
        futures = []
        for i, survivor_select in enumerate(['newest', 'best']):
            for inherit_samples in ['-1', '0', '5']:
                futures.append(executor.submit(plot_database, '50', 'noisy', 'adaptable', survivor_select, folder, popsize, inherit_samples))
    dfs = [future.result() for future in futures]

    pd.concat(dfs).to_csv("results/2511/ted-fitness", index=False)