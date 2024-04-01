import numpy as np
import warnings
from autog import *
from our_estimation_methods import *

# for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

# the common set up of my tests
TRUE_CAUSAL_EFFECT_N_UNIT = 1000 
AVG_DEGREE = 5
N_UNITS_LIST = [100, 300, 500, 700, 900]
N_ESTIMATES = 300 # number of causal effect estimates for each n_unit
N_SIMULATIONS = 1000 # the number of L samples to draw 
BURN_IN = 200
GIBBS_SELECT_EVERY = 3

# true parameters of the Data Generating Process
def GET_TRUE_PARAMS(L_edge_type,  A_edge_type, Y_edge_type):
    assert L_edge_type in ['U', 'B'] and A_edge_type in ['U', 'B'] and Y_edge_type in ['U', 'B']
    L_TRUE = np.array([-0.3, 0.4]) if L_edge_type == 'U' else np.array([0.3, 3.1, 2]) 
    A_TRUE = np.array([0.5, 0.4, 0.2, -0.2]) if A_edge_type == 'U' else np.array([2, 1, 1.3, -0.4, -0.7, 0.2])
    Y_TRUE = np.array([0.2, 1, 1.5, -0.3, 1, -0.4]) if Y_edge_type == 'U' else np.array([0, 1, -3, 0.1, 1, -0.3, 1, 2])
    return L_TRUE, A_TRUE, Y_TRUE

def est_w_autog_parallel_helper(n_units, L_edge_type, A_edge_type, Y_edge_type,
                                L_true, A_true, Y_true):
    _, network_adj_mat = create_random_network(n_units, AVG_DEGREE)
    L, A, Y = sample_LAY(network_adj_mat, L_edge_type, A_edge_type, Y_edge_type, 
                         L_true, A_true, Y_true, BURN_IN, L_biedge_const_var=True)
    
    # estimate parameters for the L and Y layers using the autog method
    L_est = minimize(npll_L, x0=np.random.uniform(-1, 1, 2), args=(L, network_adj_mat)).x
    Y_est = minimize(npll_Y, x0=np.random.uniform(-1, 1, 6), args=(L, A, Y, network_adj_mat)).x

    # compute causal effects using estimated parameters
    Y_A1_est = estimate_causal_effects_U_U(network_adj_mat, 1, L_est, Y_est, burn_in=BURN_IN)
    Y_A0_est = estimate_causal_effects_U_U(network_adj_mat, 0, L_est, Y_est, burn_in=BURN_IN)
    return Y_A1_est - Y_A0_est