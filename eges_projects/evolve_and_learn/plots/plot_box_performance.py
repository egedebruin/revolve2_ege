import matplotlib.pyplot as plt
import pandas as pd
import os
from sqlalchemy import select
import seaborn as sns
import scipy.stats as stats
import itertools

from database_components.generation import Generation
from database_components.individual import Individual
from database_components.population import Population
from revolve2.experimentation.database import open_database_sqlite, OpenMethod


def get_df(environment, folder, inherit_samples):
    database_name = f"learn-30_controllers-adaptable_survivorselect-newest_parentselect-tournament_inheritsamples-{inherit_samples}_environment-{environment}"
    files = [file for file in os.listdir(folder) if file.startswith(database_name)]
    if len(files) == 0:
        return None
    dfs = []
    i = 1
    for file_name in files:
        dbengine = open_database_sqlite(folder + "/" + file_name, open_method=OpenMethod.OPEN_IF_EXISTS)

        df_mini = pd.read_sql(
            select(
                Individual.objective_value.label("fitness")
            )
            .join_from(Generation, Population, Generation.population_id == Population.id)
            .join_from(Population, Individual, Population.id == Individual.population_id)
            .where(Generation.generation_index <= 500)
            .order_by(Individual.objective_value.desc())
            .limit(1),
            dbengine,
        )
        dfs.append(df_mini)
        i += 1

    return pd.concat(dfs)

def add_significance(ax, data, inherit_samples_list, labels):
    inherit_groups = []
    for inherit_samples in inherit_samples_list:
        inherit_groups.append(labels[inherit_samples])
    """Perform Mann-Whitney U-tests for all pairs and add significance markers."""
    y_max = data["fitness"].max()  # Get max fitness value

    comparisons = list(itertools.combinations(inherit_groups, 2))  # Pairwise comparisons
    y_offset = (y_max * 0.05) if y_max > 0 else 0.1  # Ensure a reasonable offset
    current_y = y_max * 1.05  # Start above the max fitness

    for group1, group2 in comparisons:
        # Get fitness values for both groups
        fitness1 = data[data["inherit_samples"] == group1]["fitness"]
        fitness2 = data[data["inherit_samples"] == group2]["fitness"]

        # Perform Mann-Whitney U-test
        if len(fitness1) > 0 and len(fitness2) > 0:
            stat, p_value = stats.mannwhitneyu(fitness1, fitness2, alternative='two-sided')
        else:
            continue  # Skip if any group is empty

        print(group1, group2)
        print(p_value)
        print()

        # Determine significance level
        if p_value < 0.001:
            sig = "***"
        elif p_value < 0.01:
            sig = "**"
        elif p_value < 0.05:
            sig = "*"
        else:
            continue  # Skip non-significant results

        # Get x-coordinates for boxplots
        x1, x2 = inherit_groups.index(group1), inherit_groups.index(group2)

        # Draw a significance line
        ax.plot([x1, x1, x2, x2], [current_y, current_y + y_offset, current_y + y_offset, current_y],
                color='black', linewidth=1)

        # Add significance text
        ax.text((x1 + x2) / 2, current_y + y_offset * 1.2, sig, ha='center', fontsize=12, fontweight="bold")

        # Move up for the next annotation
        current_y += y_offset * 4

    # Adjust y-axis limit to ensure visibility
    ax.set_ylim(0, current_y)


def plot_database(ax, environment, folder, labels):
    print(environment)
    """Plot all inherit_samples in the same boxplot for a given environment."""
    inherit_samples_list = ['-1', '0', '5']

    # Load full dataset with all inherit_samples
    all_data = []
    for inherit_samples in inherit_samples_list:
        df = get_df(environment, folder, inherit_samples)
        df["inherit_samples"] = labels[inherit_samples]  # Apply labels

        # Consistent colors for inherit_samples
        palette = {"No inheritance": "#66c2a5", "Inherit": "#fc8d62", "Reevaluate": "#8da0cb"}

        sns.boxplot(x="inherit_samples", y="fitness", data=df, ax=ax, hue="inherit_samples",
                    palette=palette, width=0.6, legend=False)

        ax.set_xlabel("")
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        # Increase font size for inherit_samples labels
        ax.tick_params(axis='x', labelsize=12, width=2)

        # Add environment title on every plot
        ax.set_title(labels[environment], fontsize=12, fontweight="bold", pad=5)
        all_data.append(df)

    df = pd.concat(all_data, ignore_index=True)  # Combine all inherit_samples

    # Add significance markers
    add_significance(ax, df, inherit_samples_list, labels)


def main():
    labels = {
        '-1': 'No inheritance', '0': 'Inherit', '5': 'Reevaluate',
        'flat': 'Flat', 'noisy': 'Rugged', 'hills': 'Hills', 'steps': 'Steps'
    }
    fig, ax = plt.subplots(nrows=4, ncols=2, sharex=True, figsize=(8, 10))
    folders = ["./results/new_big", "./results/new_big/cpg"]
    environments = ['flat', 'noisy', 'hills', 'steps']
    column_titles = ["Sine-wave controller", "CPG controller"]

    for j, folder in enumerate(folders):
        for i, environment in enumerate(environments):
            plot_database(ax[i, j], environment, folder, labels)

    # Remove all individual y-axis labels
    for a in ax.flatten():
        a.set_ylabel("")

    # Add a single shared y-axis label
    fig.text(0.004, 0.5, "Objective value", va='center', rotation='vertical', fontsize=14, fontweight="bold")

    # Add column titles *above* the plots to avoid overlap
    fig.text(0.18, 0.96, column_titles[0], fontsize=16, fontweight="bold")  # Left column
    fig.text(0.68, 0.96, column_titles[1], fontsize=16, fontweight="bold")  # Right column

    plt.tight_layout(rect=(0, 0, 1, 0.95))  # Adjust layout to prevent overlap
    plt.show()
