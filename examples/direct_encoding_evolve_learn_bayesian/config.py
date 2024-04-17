"""Configuration parameters for this example."""
DATABASE_FILE = "learn-20_controllers-1_environment-flat_1.sqlite"
ENVIRONMENT = 'flat'
NUM_REPETITIONS = 1
NUM_SIMULATORS = 1
NUM_PARALLEL_PROCESSES = 10

FREQUENCY = 4
ENERGY = 100000

NEW_HINGE_NEW_BRAIN = 0.1
INIT_MIN_MODULES = 10
INIT_MAX_MODULES = 20
SWITCH_BRAIN = 0.2
MAX_ADD_MODULES = 1
MAX_DELETE_MODULES = 1
CONTROLLERS = 1

LEARN_NUM_GENERATIONS = 18
NUM_RANDOM_SAMPLES = 2
KAPPA = 3  # Variation for Acquisition function (Low is exploitation, high is exploration)
ALPHA = 0  # Sampling noise
NU = 5/2  # Smoothness parameter for Matern kernel (Low is rigid, high is smooth)
LENGTH_SCALE = 0.2  # Also affects smoothness, but I'm not sure how (low is rigid, high is smooth (is it though??))
NEIGHBOUR_SCALE = 0.001
MUTATION_STD = 0.1

POPULATION_SIZE = 50
OFFSPRING_SIZE = 10
NUM_GENERATIONS = 495

CROSSOVER = True
INITIAL_POPULATION_FROM_DATABASE = False
OVERWRITE_BRAIN_GENOTYPE = False
EVOLUTIONARY_SEARCH = False
READ_ARGS = True
