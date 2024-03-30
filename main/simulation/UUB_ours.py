# UUB_autog means DGP is UUB and estimation is done via our estimation methods

import os
import sys
sys.path.append('..')
from autog import *
from our_estimation_methods import *

# for cleaner output
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


''' set up '''
L_EDGE_TYPE = 'U'
A_EDGE_TYPE = 'U'
Y_EDGE_TYPE = 'B'

TRUE_CAUSAL_EFFECT_N_UNIT = 5000 #10000
AVG_DEGREE = 5
N_UNITS_LIST = [1000, 2000, 3000]#[1000, 3000, 5000, 7000, 9000]
N_ESTIMATES = 3 # number of causal effect estimates for each n_unit
N_SIMULATIONS = 800 # the number of L samples to draw 
BURN_IN = 200

# true parameters of the Data Generating Process
L_TRUE = np.array([-0.3, 0.4])
A_TRUE = np.array([0.5, 0.4, 0.2, -0.2])
Y_TRUE = np.array([0, 1, -3, 0.1, 1, -0.3, 1, 2])


def parallel_helper(n_units):
    network_dict, network_adj_mat = create_random_network(n_units, AVG_DEGREE)
    L, A, Y = sample_LAY(network_adj_mat, L_EDGE_TYPE, A_EDGE_TYPE, Y_EDGE_TYPE, L_TRUE, A_TRUE, Y_TRUE, BURN_IN)

    return estimate_causal_effects_U_B(network_dict, network_adj_mat, L, A, Y, 
                                       N_SIMULATIONS, gibbs_select_every=3, 
                                       burn_in=BURN_IN)

        
def main():
    
    ''' evaluate true network causal effects '''
    # _, network_adj_mat = create_random_network(TRUE_CAUSAL_EFFECT_N_UNIT, AVG_DEGREE)
    # causal_effect_true = true_causal_effects_U_B(network_adj_mat, L_TRUE, Y_TRUE, BURN_IN, N_SIMULATIONS)
    causal_effect_true = 0.437
    print("True causal effect:", causal_effect_true)
    
    ''' using autog to estimate causal effects from data generated from UUB '''
    causal_effect_ests = {}
    with ProcessPoolExecutor() as executor:
        for n_units in N_UNITS_LIST:
            results = executor.map(parallel_helper, [n_units]*N_ESTIMATES)
            causal_effect_ests[f'n units {n_units}'] = list(results)
    
    ''' save results '''
    df = pd.DataFrame.from_dict(causal_effect_ests, orient='index').transpose()
    df['True Effect'] = causal_effect_true
    current_file_name = os.path.basename(__file__).split('.')[0]
    df.to_csv(f"./result/{current_file_name}.csv", index=False)

if __name__ == "__main__":
    main()