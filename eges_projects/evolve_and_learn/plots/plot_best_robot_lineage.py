import concurrent.futures

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
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

def get_df(environment, folder, inherit_samples, repetition):
    database_name = f"learn-30_controllers-adaptable_survivorselect-newest_parentselect-tournament_inheritsamples-{inherit_samples}_environment-{environment}_cpg_{repetition}."
    files = [file for file in os.listdir(folder) if file.startswith(database_name)]
    if len(files) == 0:
        return None
    dfs = []
    i = 1
    for file_name in files:
        dbengine = open_database_sqlite(folder + "/" + file_name, open_method=OpenMethod.OPEN_IF_EXISTS)

        df_mini = pandas.read_sql(
            select(
                Individual.objective_value.label("fitness"),
                Genotype.id,
                Genotype.parent_1_genotype_id
            )
            .join_from(Individual, Genotype, Individual.genotype_id == Genotype.id),
            dbengine,
        )
        dfs.append(df_mini)
        i += 1
    return pandas.concat(dfs)

def get_genotype(environment, folder, inherit_samples, repetition, genotype_id):
    database_name = f"learn-30_controllers-adaptable_survivorselect-newest_parentselect-tournament_inheritsamples-{inherit_samples}_environment-{environment}_cpg_{repetition}."
    files = [file for file in os.listdir(folder) if file.startswith(database_name)]
    if len(files) == 0:
        return None
    dfs = []
    i = 1
    for file_name in files:
        dbengine = open_database_sqlite(folder + "/" + file_name, open_method=OpenMethod.OPEN_IF_EXISTS)

        df_mini = pandas.read_sql(
            select(
                Genotype
            )
            .where(Genotype.id == genotype_id),
            dbengine,
        )
        dfs.append(df_mini)
        i += 1
    return pandas.concat(dfs)

def main() -> None:
    fig, ax = plt.subplots(nrows=4, ncols=2, sharex=True)
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

    folder = "./results/new_big/cpg"
    for inherit_samples in ['-1', '0', '5']:
        for env_number, environment in enumerate(['flat', 'noisy', 'hills', 'steps']):
            result = []
            for i in range(1, 21):
                df = get_df(environment, folder, inherit_samples, i)
                if df is None:
                    continue
                best_robot = df.loc[df['fitness'].idxmax()]
                best_genotype = get_genotype(environment, folder, inherit_samples, i, best_robot.id)

                lineage = []
                max_i = 50
                while best_robot['parent_1_genotype_id'] != -1 and max_i > 0:
                    max_i -= 1
                    parent_genotype = get_genotype(environment, folder, inherit_samples, i, best_robot.parent_1_genotype_id)
                    ted = tree_edit_distance(CoreGenotype(0.0).deserialize(json.loads(best_genotype['serialized_body'][0])),
                                       CoreGenotype(0.0).deserialize(json.loads(parent_genotype['serialized_body'][0])))
                    lineage.append((best_robot['fitness'], ted))

                    best_robot = df.loc[df['id'] == best_robot['parent_1_genotype_id']].iloc[0]
                    best_genotype = parent_genotype

                lineage.append((best_robot['fitness'], 0))

                fitnesses, teds = zip(*lineage)
                x_values = range(len(lineage))

                result.append([fitnesses, teds])
            plotable_fitnesses = []
            plotable_teds = []
            for fitnesses, teds in result:
                i = 0
                for fitness in fitnesses:
                    if i >= len(plotable_fitnesses):
                        plotable_fitnesses.append([])
                    plotable_fitnesses[i].append(fitness)
                    i += 1
                i = 0
                for ted in teds:
                    if i >= len(plotable_teds):
                        plotable_teds.append([])
                    plotable_teds[i].append(ted)
                    i += 1

            mean_values, h, x = compute_mean(plotable_fitnesses)
            mean_values_ted, h_ted, x_ted = compute_mean(plotable_teds)

            # Plot the mean
            ax[env_number][0].plot(x, mean_values, label="Mean", color=to_color[inherit_samples])
            ax[env_number][0].fill_between(x, mean_values - h, mean_values + h, color="blue", alpha=0.2, label="95% CI")
            ax[env_number][1].plot(x_ted, mean_values_ted, label="Mean", color=to_color[inherit_samples])
            ax[env_number][1].fill_between(x_ted, mean_values_ted - h_ted, mean_values_ted + h_ted, color="blue", alpha=0.2, label="95% CI")

    # Labels and title
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Mean and 95% Confidence Interval")
    plt.legend()
    plt.show()

def compute_mean(data):
    mean_values = []
    conf_intervals = []

    for row in data:
        mean = np.mean(row)  # Row-wise mean
        std_err = stats.sem(row) if len(row) > 1 else 0  # SEM, avoid div by zero
        h = std_err * stats.t.ppf(0.975, len(row) - 1) if len(row) > 1 else 0  # 95% CI
        mean_values.append(mean)
        conf_intervals.append(h)

    # Define x-axis (row index)
    x = np.arange(len(mean_values))

    # Convert lists to NumPy arrays
    mean_values = np.array(mean_values)
    conf_intervals = np.array(conf_intervals)

    return mean_values, conf_intervals, x