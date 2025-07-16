import pandas as pd

def main():
    df = pd.read_csv('./results/random_robots_learn_2.csv')
    df = df.loc[df['robot_id'] < 21]
    df = df.groupby('robot_id').head(30)

    best_flat = df.groupby('robot_id')['flat'].max()
    best_hills = df.groupby('robot_id')['hills'].max()
    print(best_flat.mean())
    print(best_hills.mean())