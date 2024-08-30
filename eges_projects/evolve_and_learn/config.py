"""Configuration parameters for this example."""
DATABASE_FILE = "test.sqlite"
ENVIRONMENT = 'steps'
NUM_REPETITIONS = 1
NUM_SIMULATORS = 1
NUM_PARALLEL_PROCESSES = 50

FREQUENCY = 4
ENERGY = 100000
MAX_ATTRACTION_COEFFICIENT = 1

NEW_HINGE_NEW_BRAIN = 1
INIT_MIN_MODULES = 5
INIT_MAX_MODULES = 10
SWITCH_BRAIN = 0.5
MAX_ADD_MODULES = 3
MAX_DELETE_MODULES = 3

MAX_NUMBER_OF_MODULES = 20

KAPPA = 3  # Variation for Acquisition function (Low is exploitation, high is exploration)
ALPHA = 0  # Sampling noise
NU = 5/2  # Smoothness parameter for Matern kernel (Low is rigid, high is smooth)
LENGTH_SCALE = 0.2  # Also affects smoothness, but I'm not sure how (low is rigid, high is smooth (is it though??))
NEIGHBOUR_SCALE = 0.001
MUTATION_STD = 0.1

POPULATION_SIZE = 50
OFFSPRING_SIZE = 50
FUNCTION_EVALUATIONS = 400000

CROSSOVER = False
INITIAL_POPULATION_FROM_DATABASE = False
OVERWRITE_BRAIN_GENOTYPE = False
REVERSE_PHASE = True
READ_ARGS = True

EVOLUTIONARY_SEARCH = False
LEARN_NUM_GENERATIONS = 0
NUM_RANDOM_SAMPLES = 1
CONTROLLERS = -1
NUM_GENERATIONS = 400
SELECT_STRATEGY = 'tournament'
