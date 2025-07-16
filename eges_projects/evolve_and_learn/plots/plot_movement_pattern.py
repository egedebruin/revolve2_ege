import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

def main():
    # Data
    data = """Controller,Environment,Inheritance,Rolling,Swimming,Walking,Worm
    Sine-wave,Flat,No,18,,,2
    Sine-wave,Flat,Redo,17,,,3
    Sine-wave,Flat,Inherit,19,,,1
    Sine-wave,Rugged,No,19,,,1
    Sine-wave,Rugged,Redo,20,,,
    Sine-wave,Rugged,Inherit,18,,1,1
    Sine-wave,Hills,No,15,,3,2
    Sine-wave,Hills,Redo,6,5,6,3
    Sine-wave,Hills,Inherit,6,,11,3
    Sine-wave,Steps,No,,2,16,2
    Sine-wave,Steps,Redo,,,14,6
    Sine-wave,Steps,Inherit,,1,14,5
    CPG,Flat,No,20,,,
    CPG,Flat,Redo,20,,,
    CPG,Flat,Inherit,20,,,
    CPG,Rugged,No,20,,,
    CPG,Rugged,Redo,19,1,,
    CPG,Rugged,Inherit,20,,,
    CPG,Hills,No,3,10,7,
    CPG,Hills,Redo,3,12,5,
    CPG,Hills,Inherit,6,7.5,6.5,
    CPG,Steps,No,,6,14,
    CPG,Steps,Redo,6,13,1,
    CPG,Steps,Inherit,3,7,10,"""

    custom_palette = {
        "Rolling": "#A6CEE3",  # Blue
        "Walking": "#FDBF6F",  # Orange
        "Swimming": "#B2DF8A",  # Green
        "Worm": "#FB9A99",  # Red
        # Add more locomotion categories as needed
    }

    # Read data into DataFrame
    df = pd.read_csv(StringIO(data))
    df['Controller'] = df['Controller'].str.strip()

    # Melt the DataFrame for better visualization
    df_melted = df.melt(id_vars=["Controller", "Environment", "Inheritance"],
                         var_name="Locomotion", value_name="Value").dropna()

    # Convert categorical variables to ordered types for consistent plotting
    df_melted["Controller"] = pd.Categorical(df_melted["Controller"], categories=["Sine-wave", "CPG"])
    df_melted["Environment"] = pd.Categorical(df_melted["Environment"], categories=["Flat", "Rugged", "Hills", "Steps"])
    df_melted["Inheritance"] = pd.Categorical(df_melted["Inheritance"], categories=["No", "Redo", "Inherit"])

    # Normalize values to percentages (relative to the max value in the dataset)
    df_melted["Value"] = df_melted["Value"] / df_melted["Value"].max() * 100
    df_melted_sine = df_melted[df_melted["Controller"] == "Sine-wave"]
    df_melted_cpg = df_melted[df_melted["Controller"] == "CPG"]

    # Set up the plot with two subplots
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

    # Plot Sine-wave
    sns.barplot(
        data=df_melted_sine, x="Environment", y="Value", hue="Locomotion",
        dodge=True, ci=None, palette=custom_palette, ax=ax[0]
    )
    ax[0].set_title("Sine-wave controller", fontsize=14, color="royalblue")
    ax[0].set_xlabel("Environment", fontsize=14)
    ax[0].set_ylabel("Occurrences (%)", fontsize=14)
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)

    # Plot CPG
    sns.barplot(
        data=df_melted_cpg, x="Environment", y="Value", hue="Locomotion",
        dodge=True, ci=None, palette=custom_palette, ax=ax[1], legend=False
    )
    ax[1].set_title("CPG controller", fontsize=14, color="darkorange")
    ax[1].set_xlabel("Environment", fontsize=14)
    ax[1].set_ylabel("")
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)

    # Improve visualization
    for axis in ax:
        axis.grid(axis="y", linestyle="--", alpha=0.7)
    ax[0].legend(title="Locomotion type")

    # Add main title
    plt.tight_layout()
    plt.show()
