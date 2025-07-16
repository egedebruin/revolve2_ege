import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import select
from randomly_sample import RandomSample
from revolve2.experimentation.database import OpenMethod, open_database_sqlite

plt.style.use('seaborn-v0_8-whitegrid')

def process_files(folder, pattern, color, label):
    dfs = []
    i = 0
    for file in os.listdir(folder):
        if pattern not in file:
            continue
        dbengine = open_database_sqlite(os.path.join(folder, file), open_method=OpenMethod.OPEN_IF_EXISTS)

        df = pd.read_sql(select(RandomSample), dbengine)

        if len(df) < 20000:
            continue

        df['cum_objective_value'] = df.groupby(['genotype_id', 'learn']) \
            .apply(lambda g: g['objective_value'].expanding().max(), include_groups=False) \
            .reset_index(level=[0, 1], drop=True)
        scaling_factor = df.groupby('genotype_id')['cum_objective_value'].transform('max')
        df['scaled_objective_value'] = df['cum_objective_value'] / scaling_factor

        pivot_df = df.pivot(index=['genotype_id', 'repetition'], columns='learn', values='scaled_objective_value')
        result_df = (pivot_df[1] - pivot_df[0]).reset_index()
        result_df.columns = ['genotype_id', 'repetition', 'scaled_objective_value']
        result_df['i'] = i
        dfs.append(result_df)
        i += 1

    if dfs:
        end_df = pd.concat(dfs)
        to_plot = []
        for name, group in end_df.groupby(['genotype_id', 'i']):
            group = group.reset_index()
            if pd.isna(group['scaled_objective_value'][0]):
                continue
            to_plot.append(np.array(group['scaled_objective_value']))

        x = range(1, 501)
        y = np.mean(to_plot, axis=0)
        q25 = np.percentile(to_plot, 25, axis=0)
        q75 = np.percentile(to_plot, 75, axis=0)

        plt.plot(x, y, label=label, color=color, linewidth=2.5)
        plt.fill_between(x, q25, q75, color=color, alpha=0.15, linewidth=0)


def main():
    plt.figure(figsize=(8, 5))

    folder = "results/random_long"

    process_files(folder, "inheritsamples--1", '#66c2a5', 'No inheritance')
    process_files(folder, "inheritsamples-0", '#fc8d62', 'Inherit samples')
    process_files(folder, "inheritsamples-5", '#8da0cb', 'Reevaluate')

    plt.title('Learning improvement over random', fontsize=16, fontweight="bold")
    plt.xlabel('Learning / Random iteration', fontsize=14, fontweight="bold")
    ylabel = plt.ylabel('Scaled difference', fontsize=14, labelpad=0, fontweight="bold")
    ylabel.set_position((-0.18, 0.5))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.subplots_adjust(left=0.08)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
