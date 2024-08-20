"""Configuration parameters for this example."""

from revolve2.ci_group.modular_robots_v1 import gecko_v1
from revolve2.ci_group.modular_robots_v1 import four_legged_spider
from revolve2.ci_group.modular_robots_v1 import testing_bot

DATABASE_FILE = "test.sqlite"
NUM_REPETITIONS = 1
NUM_SIMULATORS = 10
INITIAL_STD = 0.3
NUM_GENERATIONS = 1000
BODY = testing_bot()
FREQUENCY = 4
MAX_ATTRACTION_COEFFICIENT = 0.1
ENERGY = 10000