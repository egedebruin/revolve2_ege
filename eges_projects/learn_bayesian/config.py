"""Configuration parameters for this example."""

from revolve2.ci_group.modular_robots_v1 import gecko_v1
from revolve2.ci_group.modular_robots_v1 import snake_v1

DATABASE_FILE = "results/after_learn2/learn-1_evosearch-1_controllers-adaptable_select-tournament_environment-noisy_2_noisy_1.sqlite"
DATABASE_FILE_OLD = "old_file.sqlite"
ENVIRONMENT = "noisy"
NUM_SIMULATORS = 1
NUM_PARALLEL_PROCESSES = 10
NUM_GENERATIONS = 270
NUM_RANDOM_SAMPLES = 30
FREQUENCY = 4
BODY = gecko_v1()

KAPPA = 3  # Variation for Acquisition function (Low is exploitation, high is exploration)
ALPHA = 0  # Sampling noise
NU = 5/2  # Smoothness parameter for Matern kernel (Low is rigid, high is smooth)
LENGTH_SCALE = 0.2  # Also affects smoothness, but I'm not sure how (low is rigid, high is smooth (is it though??))
NEIGHBOUR_SCALE = 0.001
ENERGY = 100000
