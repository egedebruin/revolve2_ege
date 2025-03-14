"""Configuration parameters for this example."""

from revolve2.standards.modular_robots_v2 import gecko_v2

DATABASE_FILE = "database.sqlite"
NUM_REPETITIONS = 1
NUM_SIMULATORS = 10
INITIAL_STD = 1/3
NUM_GENERATIONS = 1000
BODY = gecko_v2()
