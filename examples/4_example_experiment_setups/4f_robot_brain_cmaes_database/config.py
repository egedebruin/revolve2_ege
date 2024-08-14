"""Configuration parameters for this example."""

from revolve2.ci_group.modular_robots_v1 import gecko_v1

DATABASE_FILE = "database.sqlite"
NUM_REPETITIONS = 1
NUM_SIMULATORS = 10
INITIAL_STD = 0.3
NUM_GENERATIONS = 500
BODY = gecko_v1()
FREQUENCY = 4
