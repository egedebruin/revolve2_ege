"""Configuration parameters for this example."""
DATABASE_FILE = "results/test.sqlite"
ENVIRONMENT = 'noisy'
NUM_REPETITIONS = 1
NUM_SIMULATORS = 1
NUM_PARALLEL_PROCESSES = 20

FREQUENCY = 4
ENERGY = 100000

NEW_HINGE_NEW_BRAIN = 1
INIT_MIN_MODULES = 15
INIT_MAX_MODULES = 20
SWITCH_BRAIN = 0.5
MAX_ADD_MODULES = 2
MAX_DELETE_MODULES = 2

MAX_NUMBER_OF_MODULES = 20

KAPPA = 3  # Variation for Acquisition function (Low is exploitation, high is exploration)
ALPHA = 1  # Sampling noise
INHERITED_ALPHA = 2
NU = 5/2  # Smoothness parameter for Matern kernel (Low is rigid, high is smooth)
LENGTH_SCALE = 0.2  # Also affects smoothness, but I'm not sure how (low is rigid, high is smooth (is it though??))
NEIGHBOUR_SCALE = 0.001
MUTATION_STD = 0.1

POPULATION_SIZE = 200
OFFSPRING_SIZE = 20
FUNCTION_EVALUATIONS = 600000
LOCAL_COMPETITION_NEIGHBOURHOOD_SIZE = 10
PARENT_TOURNAMENT_SIZE = 5

CROSSOVER = False
REVERSE_PHASE = True
READ_ARGS = True

LEARN_NUM_GENERATIONS = 5
NUM_REDO_INHERITED_SAMPLES = 0
BONUS = False
INHERIT_SAMPLES = False
EVOLUTIONARY_SEARCH = True
CONTROLLERS = 8
SURVIVOR_SELECT_STRATEGY = 'newest'
PARENT_SELECT_STRATEGY = 'tournament'
