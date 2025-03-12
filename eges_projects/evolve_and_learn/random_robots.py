import concurrent.futures

import random
from argparse import ArgumentParser

import pandas as pd
from bayes_opt import BayesianOptimization, acquisition
from sklearn.gaussian_process.kernels import Matern

import config
from database_components.genotype import Genotype
from database_components.learn_genotype import LearnGenotype
from evaluator import Evaluator
from revolve2.experimentation.rng import seed_from_time, make_rng


def main():
    parser = ArgumentParser()
    parser.add_argument("--kappa", required=True)
    parser.add_argument("--alpha", required=True)
    parser.add_argument("--length-scale", required=True)
    parser.add_argument("--do-random", required=True)
    args = parser.parse_args()

    print(f"Received Arguments: {args}")

    kappa = float(args.kappa)
    alpha = float(args.alpha)
    length_scale = float(args.length_scale)
    do_random = args.do_random == '1'

    number_of_robots = 1000
    number_of_iterations = 30
    environments = ['flat', 'noisy', 'hills', 'steps']

    rng_seed = 1111972312
    rng = make_rng(rng_seed)

    random_genotypes = [
        Genotype.initialize(
            rng=rng,
        )
        for _ in range(number_of_robots)
    ]

    evaluators = []
    for environment in environments:
        config.ENVIRONMENT = environment
        evaluators.append(Evaluator(headless=True, num_simulators=1))

    with concurrent.futures.ProcessPoolExecutor(
            max_workers=100
    ) as executor:
        futures = []
        robot_id = 1
        for genotype in random_genotypes:
            futures.append(executor.submit(test_robot, genotype, evaluators, number_of_iterations, robot_id, rng, kappa, alpha, length_scale, do_random))
            robot_id += 1

    dfs = []
    for future in futures:
        df, robot_id = future.result()
        df['robot_id'] = robot_id
        dfs.append(df)
    result_df = pd.concat(dfs)
    result_df['rng'] = rng_seed
    result_df.to_csv(f"random_robots_learn_{kappa}_{alpha}_{length_scale}_{do_random}.csv", index=False)

def test_robot(genotype, evaluators, number_of_iterations, robot_id, rng, kappa, alpha, length_scale, do_random):
    optimizers = []
    for _ in evaluators:
        optimizer = BayesianOptimization(
            f=None,
            pbounds=genotype.get_p_bounds(),
            allow_duplicate_points=True,
            random_state=int(rng.integers(low=0, high=2 ** 32)),
            acquisition_function=acquisition.UpperConfidenceBound(kappa=kappa,
                                                                  random_state=rng.integers(low=0, high=2 ** 32))
        )
        optimizer.set_gp_params(alpha=alpha)
        optimizer.set_gp_params(
            kernel=Matern(nu=5/2, length_scale=length_scale, length_scale_bounds="fixed"))
        optimizers.append(optimizer)

    result = []
    developed_body = genotype.develop_body()
    brain_uuids = list(genotype.brain.keys())
    for i in range(number_of_iterations):
        objective_values = []
        for optimizer, evaluator in zip(optimizers, evaluators):
            if i < 5 and do_random:
                next_point = genotype.get_random_next_point(rng)
            else:
                next_point = optimizer.suggest()
            next_point = dict(sorted(next_point.items()))

            new_learn_genotype = LearnGenotype(brain={})
            new_learn_genotype.next_point_to_brain(next_point, brain_uuids)
            modular_robot = new_learn_genotype.develop(developed_body)

            objective_value = evaluator.evaluate(modular_robot)
            objective_values.append(objective_value)
            optimizer.register(params=next_point, target=objective_value)
        result.append(objective_values)
    return pd.DataFrame(result, columns=['flat', 'noisy', 'hills', 'steps']), robot_id

if __name__ == '__main__':
    main()