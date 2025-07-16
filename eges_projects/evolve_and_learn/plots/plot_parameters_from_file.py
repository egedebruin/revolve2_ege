import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    sns.set_theme(style="whitegrid", font_scale=1.2)

    inherit_samples_colors = {
        0: "#fc8d62",  # Orange
        -1: "#66c2a5",  # Green
        5: "#8da0cb",  # Blue
    }

    inherit_samples_labels = {
        '0': "Inherit samples",
        '-1': "No inheritance",
        '5': "Reevaluate",
    }

    environment_to_label = {
        'flat': 'Flat',
        'noisy': 'Rugged',
        'hills': 'Hills',
        'steps': 'Steps',
    }

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6), sharex=True,
                             gridspec_kw={'hspace': 0.3, 'wspace': 0.1})

    axes[0][0].annotate('All robots', xy=(0.5, 1.1), xycoords='axes fraction', fontsize=16, fontweight='bold', ha='center')
    axes[0][1].annotate('10% best robots', xy=(0.5, 1.1), xycoords='axes fraction', fontsize=16, fontweight='bold', ha='center')

    for j, end_file in enumerate(['', '-best-robots']):
        for i, start_file in enumerate(['', 'cpg-']):
            full_df = pd.read_csv(f"results/{start_file}parameters-in-bins{end_file}.csv", index_col=0)
            df = full_df.drop('environment', axis=1)

            df_melted = df.melt(id_vars=['inherit_samples'], var_name='bin', value_name='count')

            bin_order = sorted(df.columns[:-1], key=lambda x: float(x.split('-')[0]))
            df_melted['bin'] = pd.Categorical(df_melted['bin'], categories=bin_order, ordered=True)

            total_per_sample = df_melted.groupby('inherit_samples')['count'].transform('sum')
            df_melted['percentage'] = (df_melted['count'] / total_per_sample) * 100

            # Sort bars by inherit_samples first, then by bin
            df_melted = df_melted.sort_values(['inherit_samples', 'bin'])

            # Assign sequential x-positions (first all `inherit_samples=0`, then all `-1`, then `5`)
            unique_bins = len(bin_order)
            df_melted['x_pos'] = df_melted['bin'].cat.codes + df_melted['inherit_samples'].astype(str).astype('category').cat.codes * unique_bins

            ax = axes[i][j]
            sns.barplot(
                data=df_melted,
                x='x_pos',
                y='percentage',
                hue='inherit_samples',
                ax=ax,
                dodge=False,
                palette=inherit_samples_colors,
            )

            ax.set_xlabel("") if i < 3 else ax.set_xlabel("Distance to extreme", fontsize=12)
            ax.set_ylabel("Percentage (%)" if j == 0 else "", fontsize=12)

            # Draw vertical lines to separate groups
            unique_samples_sorted = sorted(df_melted['inherit_samples'].unique())
            for s in unique_samples_sorted[1:]:  # Skip first group (no need to separate before it)
                boundary = df_melted[df_melted['inherit_samples'] == s]['x_pos'].min() - 0.5
                ax.axvline(boundary, color="black", linestyle="--", linewidth=1.5)

            tick_labels = []
            for k in range(15):
                tick_labels.append(str((k%5)/10))

            ax.set_xticks(range(1, 75, 5))
            ax.set_xticklabels(tick_labels, rotation=45)

            sns.despine()

    handles, labels = axes[0][0].get_legend_handles_labels()
    new_labels = [inherit_samples_labels[label] for label in labels]
    fig.legend(handles, new_labels, loc='right', fontsize=12)

    for ax_row in axes:
        for ax in ax_row:
            ax.get_legend().remove()

    fig.text(0.02, 0.72, "Sine wave controller", va='center', rotation='vertical', fontsize=14, fontweight="bold")
    fig.text(0.02, 0.28, "CPG controller", va='center', rotation='vertical', fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
