"""Configuration parameters for this example."""
DATABASE_FILE = "learn-100_controllers-1_hills_1.sqlite"
PREVIOUS_DATABASE_FILE = "test2_short_no_updown.sqlite"
NUM_REPETITIONS = 1
NUM_SIMULATORS = 1
NUM_PARALLEL_PROCESSES = 10

FREQUENCY = 4
ENERGY = 10000

NEW_HINGE_NEW_BRAIN = 0.1
INIT_MIN_MODULES = 5
INIT_MAX_MODULES = 10
SWITCH_BRAIN = 0.5
MAX_ADD_MODULES = 5
MAX_DELETE_MODULES = 5
CONTROLLERS = 1

LEARN_NUM_GENERATIONS = 90
NUM_RANDOM_SAMPLES = 10
KAPPA = 3  # Variation for Acquisition function (Low is exploitation, high is exploration)
ALPHA = 0  # Sampling noise
NU = 5/2  # Smoothness parameter for Matern kernel (Low is rigid, high is smooth)
LENGTH_SCALE = 0.2  # Also affects smoothness, but I'm not sure how (low is rigid, high is smooth (is it though??))
NEIGHBOUR_SCALE = 0.001

POPULATION_SIZE = 50
OFFSPRING_SIZE = 10
NUM_GENERATIONS = 495

CROSSOVER = True
INITIAL_POPULATION_FROM_DATABASE = False
