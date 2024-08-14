import pandas
import os
import matplotlib.pyplot as plt
from revolve2.experimentation.database import open_database_sqlite, OpenMethod
from sqlalchemy import select

from experiment import Experiment
from generation import Generation
from genotype import Genotype
from individual import Individual
from population import Population
from learn_generation import LearnGeneration
from learn_population import LearnPopulation
from learn_individual import LearnIndividual


def get_df(learn, evosearch, controllers, environment, survivor_select):
    database_name = f"learn-{learn}_evosearch-{evosearch}_controllers-{controllers}_select-{survivor_select}_environment-{environment}"
    print(database_name)
    files = [file for file in os.listdir("../results/1208") if file.startswith(database_name)]
    if len(files) == 0:
        return None
    dfs = []
    i = 1
    for file_name in files:
        dbengine = open_database_sqlite("results/1208/" + file_name, open_method=OpenMethod.OPEN_IF_EXISTS)

        df_mini = pandas.read_sql(
            select(
                Experiment.id.label("experiment_id"),
                Generation.generation_index,
                Genotype.id.label("genotype_id"),
                LearnGeneration.generation_index.label('learn_generation_index'),
                LearnIndividual.fitness
            )
            .join_from(Experiment, Generation, Experiment.id == Generation.experiment_id)
            .join_from(Generation, Population, Generation.population_id == Population.id)
            .join_from(Population, Individual, Population.id == Individual.population_id)
            .join_from(Individual, Genotype, Individual.genotype_id == Genotype.id)
            .join_from(Genotype, LearnGeneration, Genotype.id == LearnGeneration.genotype_id)
            .join_from(LearnGeneration, LearnPopulation, LearnGeneration.learn_population_id == LearnPopulation.id)
            .join_from(LearnPopulation, LearnIndividual, LearnPopulation.id == LearnIndividual.population_id),
            dbengine,
        )
        df_mini['experiment_id'] = i
        df_mini['function_evaluations'] = df_mini['generation_index'] * int(learn) * 50 + int(learn) * 50
        dfs.append(df_mini)
        i += 1
    return pandas.concat(dfs)


def main() -> None:
    fig, ax = plt.subplots(nrows=4, ncols=2, sharex='col', sharey='row')
    for i, environment in enumerate(['flat']):
        for j, (learn, evosearch) in enumerate([('30', '1')]):
            df = get_df(learn, evosearch, 'adaptable', environment, 'tournament')
            grouped = df.groupby(['experiment_id', 'genotype_id'])

            snaggywaggy = {
                'difference': [],
                'difference2': [],
                'function_evaluations': [],
            }
            for name, group in grouped:
                first = group.loc[group['learn_generation_index'] < 1]
                first_three = group.loc[group['learn_generation_index'] < 5]
                last = group.loc[group['learn_generation_index'] >= int(learn) - 1]
                last_three = group.loc[group['learn_generation_index'] >= int(learn) - 5]
                snaggywaggy['difference'].append(last_three.max()['fitness'] - first_three.max()['fitness'])
                snaggywaggy['difference2'].append(last.mean()['fitness'] - first.mean()['fitness'])
                snaggywaggy['function_evaluations'].append(first_three.mean()['function_evaluations'])

            thisisit = pandas.DataFrame(snaggywaggy)
            agg = (
                thisisit.groupby(["function_evaluations"])
                .agg({"difference": ["mean"], "difference2": ["mean"]})
                .reset_index()
            )
            agg.columns = [
                "function_evaluations",
                "mean_difference",
                "mean_difference2",
            ]
            ax[i][j].plot(agg['function_evaluations'], agg['mean_difference'])
            # ax[i][j].plot(agg['function_evaluations'], agg['mean_difference2'])
    plt.show()


if __name__ == "__main__":
    main()
