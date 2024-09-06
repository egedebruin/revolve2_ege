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


def fast_non_dominated_sort(population):
    """
    Perform non-dominated sorting on the population.
    """
    S = [[] for _ in range(len(population))]
    front = [[]]
    n = [0] * len(population)
    rank = [0] * len(population)

    for p in range(len(population)):
        for q in range(len(population)):
            if dominates(population[p], population[q]):
                S[p].append(q)
            elif dominates(population[q], population[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            front[0].append(p)

    i = 0
    while front[i]:
        next_front = []
        for p in front[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        front.append(next_front)

    del front[-1]  # Remove the last empty front
    return front, rank


def dominates(individual1, individual2):
    """
    Returns True if individual1 dominates individual2, False otherwise.
    """
    return (individual1[0] <= individual2[0] and individual1[1] <= individual2[1]) and (individual1 != individual2)


def calculate_crowding_distance(front, population):
    """
    Calculate the crowding distance for each individual in a front.
    """
    distance = [0] * len(front)
    if len(front) > 0:
        for m in range(2):  # Assuming two objectives
            sorted_front = sorted(front, key=lambda x: population[x][m])
            distance[0] = distance[-1] = float('inf')
            min_value = population[sorted_front[0]][m]
            max_value = population[sorted_front[-1]][m]
            if max_value != min_value:
                for i in range(1, len(front) - 1):
                    distance[i] += (population[sorted_front[i + 1]][m] - population[sorted_front[i - 1]][m]) / (
                                max_value - min_value)
    return distance


def nsga_ii_selection(population, num_to_select=10):
    """
    Select individuals based on NSGA-II algorithm.

    :param population: A list of individuals where each individual is a list/tuple of two objective values.
    :param num_to_select: The number of individuals to select.
    :return: A list of selected individuals.
    """
    fronts, _ = fast_non_dominated_sort(population)

    selected_individuals = []
    for front in fronts:
        if len(selected_individuals) + len(front) <= num_to_select:
            selected_individuals.extend(front)
        else:
            crowding_distances = calculate_crowding_distance(front, population)
            sorted_front = sorted(zip(front, crowding_distances), key=lambda x: x[1], reverse=True)
            selected_individuals.extend(
                [individual for individual, _ in sorted_front[:num_to_select - len(selected_individuals)]])
            break

    return selected_individuals

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

def get_all_genotypes(learn, environment):
    folder = "results/3008"
    database_name = f"learn-{learn}_evosearch-1_controllers-adaptable_select-tournament_environment-{environment}"
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
        for environment in ['flat', 'noisy', 'steps']:
            create_file_content(learn, environment)



def create_file_content(learn, environment):
    result = {
        'learn': [],
        'environment': [],
        'generation': [],
        'experiment_id': [],
        'average_tree_edit_distance': [],
        'average_size': [],
        'max_tree_edit_distance': [],
        'max_size': [],
    }
    df = get_all_genotypes(learn, environment)
    max_generations = df['generation_index'].nunique()
    experiments = df['experiment_id'].nunique()
    for experiment_id in range(1, experiments + 1):
        for i in range(max_generations):
            df_experiment = df.loc[df['experiment_id'] == experiment_id]
            df_generation = df_experiment.loc[df_experiment['generation_index'] == i]
            bodies = list(df_generation['serialized_body'])
            fitnesses = list(df_generation['fitness'])

            sizes = []
            population = []
            for j in range(len(bodies)):
                print(j)
                tree_edit_distances = []
                sizes.append(size(CoreGenotype(0.0).deserialize(json.loads(bodies[j]))))
                for k in range(len(bodies)):
                    tree_edit_distances.append(tree_edit_distance(CoreGenotype(0.0).deserialize(json.loads(bodies[j])), CoreGenotype(0.0).deserialize(json.loads(bodies[k]))))
                sorted_indexes = sorted(range(len(tree_edit_distances)), key=lambda i: tree_edit_distances[i])[1:6]
                local_competition_fitness = 0
                for k in sorted_indexes:
                    if fitnesses[j] > fitnesses[k]:
                        local_competition_fitness += 1
                population.append((sum(tree_edit_distances)/len(tree_edit_distances), local_competition_fitness))
            selected_individuals = nsga_ii_selection(population)
            print(selected_individuals)

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
