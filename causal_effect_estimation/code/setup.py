import numpy as np
from autog_estimation_methods import *
from our_estimation_methods import *

# import infrastructure methods
import sys
sys.path.append("../../")
from infrastructure.network_utils import *
from infrastructure.data_generator import *

# for cleaner output
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# the common set up of my tests
TRUE_CAUSAL_EFFECT_N_UNIT = 5000
AVG_DEGREE = 5
MAX_NEIGHBORS = 10
N_UNITS_LIST = [500, 1000, 2000, 3000, 4000, 5000]
N_ESTIMATES = 200 # number of causal effect estimates for each n_unit
# N_SIMULATIONS = 500 # the number of L samples to draw 
N_SIM_MULTIPLIER = 0.1 # how much simulations to draw as a factor of n_units
BURN_IN = 200
GIBBS_SELECT_EVERY = 3
SAVE_OUTPUT_TO_DIR = "../result/raw_output/"

def GET_TRUE_PARAMS(L_edge_type, A_edge_type, Y_edge_type):
    '''
    Returns the true parameters of the Data Generating Process for the L, A, and Y layers.
    
    Args:
        L_edge_type: str, the type of edge in the L layer. Either 'U' for undirected or 'B' for bidirected.
        A_edge_type: str, the type of edge in the A layer. Either 'U' or 'B'.
        Y_edge_type: str, the type of edge in the Y layer. Either 'U' or 'B'.
    
    Returns:
        A tuple of 3 numpy arrays, each representing the true parameters of the L, A, and Y layers respectively.
    '''
    assert L_edge_type in ['U', 'B']
    assert A_edge_type in ['U', 'B']
    assert Y_edge_type in ['U', 'B']
    
    L_TRUE = np.array([-0.3, 0.4]) if L_edge_type == 'U' else np.array([0.3, 3.5, 0.7]) 
    A_TRUE = np.array([5, 4, -2, -1.2]) if A_edge_type == 'U' else np.array([2, 1, 1.3, 6.4, 5.7, 0.2])
    Y_TRUE = np.array([2, 1, 1.5, -5.3, 1, -4]) if Y_edge_type == 'U' else np.array([0, 1, -3, 2.1, 7, -4.3, 1, 2])
    
    # L_TRUE = np.array([-0.3, 0.4]) if L_edge_type == 'U' else np.array([0.3, 3.5, 0.7]) 
    # A_TRUE = np.array([0.5, 0.4, 0.2, -0.2]) if A_edge_type == 'U' else np.array([2, 1, 1.3, -0.4, -0.7, 0.2])
    # Y_TRUE = np.array([0.2, 1, 1.5, -0.3, 1, -0.4]) if Y_edge_type == 'U' else np.array([0, 1, -3, 0.1, 1, -0.3, 1, 2])
    return L_TRUE, A_TRUE, Y_TRUE

def est_w_autog_parallel_helper(params):
    '''
    Helper function for parallelizing the estimation of the average causal 
    effect using the autog method. 
    
    Args:
        params: a tuple of 7 elements (n_units, L_edge_type, A_edge_type, 
        Y_edge_type, L_true, A_true, Y_true)
        
        Notes:
            n_units: int, the number of units in the network.
            L_edge_type: str, the type of edge in the L layer. Either 'U' or 'B'.
            A_edge_type: str, the type of edge in the A layer. Either 'U' or 'B'.
            Y_edge_type: str, the type of edge in the Y layer. Either 'U' or 'B'.
            L_true: numpy array, the true parameters of the L layer.
            A_true: numpy array, the true parameters of the A layer.
            Y_true: numpy array, the true parameters of the Y layer.
        
    Returns:
        The estimated causal effect Y(A=1) - Y(A=0) using the autog method,
        where Y(A=1) represents the average value of Y_i across all units i 
        in the network when A_i is intervened to be 1 for all i. 
    '''
    # unpack parameters
    n_units, L_edge_type, A_edge_type, Y_edge_type, L_true, A_true, Y_true = params
    
    # create a random network and sample data according to it
    _, network_adj_mat = create_random_network(n_units, AVG_DEGREE, MAX_NEIGHBORS)
    L, A, Y = sample_LAY(network_adj_mat, L_edge_type, A_edge_type, Y_edge_type, 
                         L_true, A_true, Y_true, BURN_IN)
    
    # estimate parameters for the L layer using autog
    if L_edge_type == 'B':
        # in our set up, when L_edge_type is bidirected, the data is continuous
        # L_est has 3 parameters in this case
        # the 3rd position of params is the variance of normal distribution, 
        # which should be non-negative.
        L_est = minimize(npll_L_continuous, x0=np.random.uniform(-1, 1, 3),
                 args=(L, network_adj_mat), 
                 bounds=[(-np.inf, np.inf), (-np.inf, np.inf), (1e-10, np.inf)]).x
    else:
        # otherwise, L variables are binary
        # L_est has 2 parameters in this case
        L_est = minimize(npll_L, x0=np.random.uniform(-1, 1, 2), args=(L, network_adj_mat)).x
    
    # estimate parameters for the Y layer using autog
    Y_est = minimize(npll_Y, x0=np.random.uniform(-1, 1, 6), args=(L, A, Y, network_adj_mat)).x

    # compute the average causal effect using estimated parameters
    Y_A1_est = estimate_causal_effects_U_U(network_adj_mat, 1, L_est, Y_est, 
                                           burn_in=BURN_IN, 
                                           n_simulations=int(N_SIM_MULTIPLIER*n_units),
                                           gibbs_select_every=GIBBS_SELECT_EVERY,
                                           L_is_continuous=(L_edge_type == 'B'))
    Y_A0_est = estimate_causal_effects_U_U(network_adj_mat, 0, L_est, Y_est, 
                                           burn_in=BURN_IN, 
                                           n_simulations=int(N_SIM_MULTIPLIER*n_units),
                                           gibbs_select_every=GIBBS_SELECT_EVERY,
                                           L_is_continuous=(L_edge_type == 'B'))
    return Y_A1_est - Y_A0_est