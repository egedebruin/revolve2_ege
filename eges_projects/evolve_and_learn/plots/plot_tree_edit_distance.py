import json
import os

import pandas
import matplotlib.pyplot as plt

from database_components.genotype import Genotype
from database_components.individual import Individual
from database_components.population import Population
from database_components.generation import Generation
from genotypes.body_genotype_direct import CoreGenotype
from sqlalchemy import select

from revolve2.experimentation.database import OpenMethod, open_database_sqlite


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

def get_all_genotypes(learn, survivor_select):
    folder = "results/0909"
    database_name = f"learn-{learn}_evosearch-1_controllers-adaptable_survivorselect-{survivor_select}_parentselect-tournament_environment-noisy"
    print(database_name)
    files = [file for file in os.listdir(folder) if file.startswith(database_name)]
    if len(files) == 0:
        return None
    dfs = []
    experiment_id = 1
    for file_name in files:
        dbengine = open_database_sqlite(folder + "/" + file_name, open_method=OpenMethod.OPEN_IF_EXISTS)

        df_mini = pandas.read_sql(
            select(
                Individual.fitness,
                Generation.generation_index,
                Genotype
            )
            .join_from(Generation, Population, Generation.population_id == Population.id)
            .join_from(Population, Individual, Population.id == Individual.population_id)
            .join_from(Individual, Genotype, Individual.genotype_id == Genotype.id),
            dbengine,
        )
        df_mini['experiment_id'] = experiment_id
        dfs.append(df_mini)
        experiment_id += 1
    return pandas.concat(dfs)

def main():
    # with concurrent.futures.ProcessPoolExecutor(
    #         max_workers=6
    # ) as executor:
    #     futures = []
    #     for learn in ['1', '30']:
    #         for environment in ['flat', 'noisy', 'steps']:
    #             futures.append(executor.submit(create_file_content, learn, environment))
    # dfs = []
    # for future in futures:
    #     dfs.append(pd.DataFrame(future.result()))
    # pd.concat(dfs).to_csv("results/tree-edit-distance-2908.csv", index=False)
    for learn in ['1', '30']:
        for survivor_select in ['best', 'newest']:
            create_file_content(learn, survivor_select)



def create_file_content(learn, survivor_select):
    result = {
        'learn': [],
        'survivor_select': [],
        'generation': [],
        'experiment_id': [],
        'average_tree_edit_distance': [],
        'average_size': [],
        'max_tree_edit_distance': [],
        'max_size': [],
    }
    df = get_all_genotypes(learn, survivor_select)
    max_generations = df['generation_index'].nunique()
    experiments = df['experiment_id'].nunique()
    for experiment_id in range(1, experiments + 1):
        for i in range(max_generations):
            df_experiment = df.loc[df['experiment_id'] == experiment_id]
            df_generation = df_experiment.loc[df_experiment['generation_index'] == i]
            bodies = list(df_generation['serialized_body'])

            tree_edit_distances = []
            sizes = []
            for j in range(len(bodies)):
                sizes.append(size(CoreGenotype(0.0).deserialize(json.loads(bodies[j]))))
                for k in range(j + 1, len(bodies)):
                    tree_edit_distances.append(tree_edit_distance(CoreGenotype(0.0).deserialize(json.loads(bodies[j])), CoreGenotype(0.0).deserialize(json.loads(bodies[k]))))
            result['learn'].append(learn)
            result['survivor_select'].append(survivor_select)
            result['generation'].append(i)
            result['experiment_id'].append(experiment_id)
            result['average_tree_edit_distance'].append(sum(tree_edit_distances) / len(tree_edit_distances))
            result['average_size'].append(sum(sizes) / len(sizes))
            result['max_tree_edit_distance'].append(max(tree_edit_distances))
            result['max_size'].append(max(sizes))

    return result


def plot_thingy():
    fig, ax = plt.subplots(nrows=3, ncols=2, sharex='col')
    x_axis = 'function_evaluations'
    df = pandas.read_csv('results/tree-edit-distance-2908.csv', sep=",")

    learn_to_color = {
        1: 'red',
        30: 'blue'
    }

    for learn in [1, 30]:
        for j, environment in enumerate(['flat', 'steps', 'noisy']):
            for i, thingy in enumerate(['average_tree_edit_distance', 'average_size']):
                # ax[j][i].set_xlim(0, 6500)
                current_df = df.loc[df['learn'] == learn]
                current_df = current_df.loc[current_df['environment'] == environment]
                current_df['morphologies'] = current_df['generation'] * 50 + 50
                current_df['function_evaluations'] = current_df['generation'] * int(learn) * 50 + int(learn) * 50

                agg_per_experiment_per_generation = (
                    current_df.groupby(["experiment_id", x_axis])
                    .agg({thingy: ["max", "mean"]})
                    .reset_index()
                )
                agg_per_experiment_per_generation.columns = [
                    "experiment_id",
                    x_axis,
                    "max_" + thingy,
                    "mean_" + thingy,
                ]
                agg_per_generation = (
                    agg_per_experiment_per_generation.groupby(x_axis)
                    .agg({"max_" + thingy: ["mean", "std"], "mean_" + thingy: ["mean", "std"]})
                    .reset_index()
                )
                agg_per_generation.columns = [
                    x_axis,
                    "max_" + thingy + "_mean",
                    "max_" + thingy + "_std",
                    "mean_" + thingy + "_mean",
                    "mean_" + thingy + "_std",
                ]

                ax[j][i].plot(
                    agg_per_generation[x_axis],
                    agg_per_generation["mean_" + thingy + "_mean"],
                    linewidth=2,
                    color=learn_to_color[learn],
                )
                ax[j][i].fill_between(
                    agg_per_generation[x_axis],
                    agg_per_generation["mean_" + thingy + "_mean"]
                    - agg_per_generation["mean_" + thingy + "_std"],
                    agg_per_generation["mean_" + thingy + "_mean"]
                    + agg_per_generation["mean_" + thingy + "_std"],
                    alpha=0.1,
                    color=learn_to_color[learn]
            )
    plt.show()

if __name__ == '__main__':
    main()
