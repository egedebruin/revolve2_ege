import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

def main():
    to_color = {
        '-1': '#66c2a5',
        '0': '#fc8d62',
        '5': '#8da0cb',
    }
    to_label = {
        '-1': 'No inheritance',
        '0': 'Inherit samples',
        '5': 'Reevaluate',
    }
    to_title = {
        'flat': 'Flat',
        'noisy': 'Rugged',
        'hills': 'Hills',
        'steps': 'Steps',
    }

    df = pd.read_csv('./results/ted-fitness-parent.csv')
    df['fitness_improvement'] = df['fitness'] - df['parent_fitness']

    df = df.loc[df['fitness'] > 0]
    df = df.loc[df['parent_fitness'] > 0]
    df = df.loc[df['tree_edit_distance'] > 0]

    # df = df.loc[df['fitness_improvement'] > 0]
    df = df.loc[df['generation'] < 20]
    # df = df.loc[df['tree_edit_distance'] < 8]

    fig = plt.figure(figsize=(10, 12))
    gs = gridspec.GridSpec(4, 2, width_ratios=[1, 2], height_ratios=[1, 1, 1, 1])  # 4 environments, 2 columns

    for j, environment in enumerate(['flat', 'noisy', 'hills', 'steps']):
        sub_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[j, 0], hspace=0.05)  # Stack 3 small plots

        for i, inherit_samples in enumerate([-1, 0, 5]):
            ax = plt.subplot(sub_gs[i])
            df_mini = df.loc[(df['inherit_samples'] == inherit_samples) & (df['environment'] == environment)]

            if not df_mini.empty:
                ax.scatter(df_mini['tree_edit_distance'], df_mini['fitness_improvement'],
                           color=to_color[str(inherit_samples)], s=10)

            if j == 3:
                ax.set_xlabel('Tree edit distance')
            ax.set_yticks([])  # Remove y-axis ticks to save space

        # Right column: Single trend line plot
        ax_right = plt.subplot(gs[j, 1])
        for inherit_samples in [-1, 0, 5]:
            df_mini = df.loc[(df['inherit_samples'] == inherit_samples) & (df['environment'] == environment)]
            if len(df_mini) > 1:
                m, b = np.polyfit(df_mini['tree_edit_distance'], df_mini['fitness_improvement'], 1)
                ax_right.plot(df_mini['tree_edit_distance'], m * df_mini['tree_edit_distance'] + b,
                              color=to_color[str(inherit_samples)], label=to_label[str(inherit_samples)], linewidth=2)

        if j == 0:
            ax_right.legend()
        if j == 3:
            ax_right.set_xlabel('Tree edit distance')
        ax_right.set_title(to_title[environment], fontsize=16, pad=5)
        ax_right.set_ylabel('Fitness improvement')

    plt.tight_layout()
    plt.show()


def main_the_second():
    df = pd.read_csv('./results/ted-fitness-parent_cpg.csv')
    for i in range(30):
        df_ted = df.loc[df['tree_edit_distance'] == i]
        print(i, len(df_ted.index)/len(df.index))

def main_the_third():
    to_label = {
        '-1': 'No inheritance',
        '0': 'Inherit samples',
        '5': 'Reevaluate',
    }

    df = pd.read_csv('./results/ted-fitness-parent.csv')
    df['fitness_improvement'] = df['fitness'] - df['parent_fitness']

    df = df.loc[(df['fitness'] > 0) & (df['parent_fitness'] > 0) & (df['tree_edit_distance'] > 0)]
    df = df.loc[df['generation'] < 20]

    # Convert `inherit_samples` column
    df['inherit_samples'] = df['inherit_samples'].astype(str).map(to_label)

    fig, ax = plt.subplots(nrows=4, ncols=4, sharex=True, figsize=(12, 12))

    for i, environment in enumerate(['flat', 'noisy', 'hills', 'steps']):
        for j, ted in enumerate([1, 3, 5, 7]):
            df_mini = df.loc[(df['environment'] == environment) & (df['tree_edit_distance'] == ted)]
            palette = {"No inheritance": "#66c2a5", "Inherit samples": "#fc8d62", "Reevaluate": "#8da0cb"}

            # Set legend only for the top-right plot
            show_legend = (i == 0 and j == 3)

            sns.boxplot(x="inherit_samples", y="fitness_improvement", data=df_mini, ax=ax[i, j],
                        hue="inherit_samples", palette=palette, width=0.6, legend=show_legend)

            # Only add the y-axis label once (first column)
            if j == 0:
                ax[i, j].set_ylabel("Fitness Improvement")
            else:
                ax[i, j].set_ylabel("")

            # Set x-axis label as `ted` value
            ax[i, j].set_xlabel("Tree edit distance: " + str(ted))

            # Remove default x-tick labels
            ax[i, j].set_xticklabels([])

    plt.tight_layout()
    plt.show()

def main_the_force():
    # Set theme for aesthetics
    sns.set_theme(style="whitegrid", font_scale=1.3)

    to_label = {
        '-1': 'No inheritance',
        '0': 'Inherit samples',
        '5': 'Reevaluate',
    }

    to_color = {
        'No inheritance': '#66c2a5',
        'Inherit samples': '#fc8d62',
        'Reevaluate': '#8da0cb',
    }

    to_title = {
        'flat': 'Flat',
        'noisy': 'Rugged',
        'hills': 'Hills',
        'steps': 'Steps',
    }

    df = pd.read_csv('./results/ted-fitness-parent.csv')
    df['fitness_improvement'] = df['fitness'] - df['parent_fitness']
    df['inherit_samples'] = df['inherit_samples'].astype(str).map(to_label)

    df2 = df.loc[df['fitness_improvement'] > 0].copy()

    # Convert all tree edit distances to strings and group TED > 5 into "6+"
    df['tree_edit_distance_grouped'] = df['tree_edit_distance'].apply(lambda x: int(x) if x <= 4 else "5+")
    df2['tree_edit_distance_grouped'] = df2['tree_edit_distance'].apply(lambda x: int(x) if x <= 4 else "5+")

    # Define categorical order explicitly (as strings)
    ted_order = [0, 1, 2, 3, 4, '5+']
    hue_order = list(to_label.values())

    fig, ax = plt.subplots(nrows=4, ncols=2, sharex=True, figsize=(8, 7))

    # Add column titles
    ax[0][0].annotate('All samples', xy=(0.5, 1.3), xycoords='axes fraction', fontsize=16, fontweight='bold',
                      ha='center')
    ax[0][1].annotate('Only improving', xy=(0.5, 1.3), xycoords='axes fraction', fontsize=16, fontweight='bold',
                      ha='center')

    for i, environment in enumerate(['flat', 'noisy', 'hills', 'steps']):
        for j, dataframe in enumerate([df, df2]):
            df_mini = dataframe.loc[dataframe['environment'] == environment]

            # Count occurrences for each combination of `tree_edit_distance_grouped` and `inherit_samples`
            df_counts = df_mini.groupby(['tree_edit_distance_grouped', 'inherit_samples']).size().reset_index(
                name='count')

            sns.barplot(
                data=df_counts,
                x='tree_edit_distance_grouped',
                y='count',
                hue='inherit_samples',
                hue_order=hue_order,
                order=ted_order,  # Ensure correct x-axis order
                palette=to_color,
                ax=ax[i][j],
                dodge=True,
                alpha=0.9,
                edgecolor="black",
                width=0.6
            )

            # Clean, concise titles
            ax[i][j].set_title(to_title[environment], fontsize=14, pad=8)
            ax[i][j].set_xlabel("")
            ax[i][j].set_ylabel("")

            # Improve tick formatting
            ax[i][j].tick_params(axis='y', labelsize=10)

    # Add a single centered x-axis label
    fig.text(0.52, 0.01, "Tree Edit Distance", fontsize=14, fontweight="bold", ha="center")

    # Customize the legend only once, outside the plot
    handles, labels = ax[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", fontsize=10, frameon=True)

    # Remove all individual legends inside subplots
    for i in range(4):
        for j in range(2):
            ax[i][j].legend_.remove()

    fig.text(0.002, 0.5, "Occurrences", va='center', rotation='vertical', fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()