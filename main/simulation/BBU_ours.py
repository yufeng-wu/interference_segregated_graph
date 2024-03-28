# BBU_autog means DGP is BBU and estimation is done via our estimation methods

import sys
sys.path.append('..')
from autog import *
from our_estimation_methods import *
import os


''' set up '''
L_EDGE_TYPE = 'B'
A_EDGE_TYPE = 'B'
Y_EDGE_TYPE = 'U'

TRUE_CAUSAL_EFFECT_N_UNIT = 5000
AVG_DEGREE = 5
N_UNITS_LIST = [500, 1000, 2000, 3000]
N_ESTIMATES = 100 # number of causal effect estimates for each n_unit
N_SIMULATIONS = 100 # the number of L samples to draw 
BURN_IN = 200

# true parameters of the Data Generating Process
L_TRUE = np.array([1.6, 0.3, 2])
A_TRUE = np.array([0, 1, 0.3, -0.4, -0.7, 0.2])
Y_TRUE = np.array([0.2, 1, 1.5, -0.3, 1, -0.4])

def parallel_helper(n_units):
    network_dict, network_adj_mat = create_random_network(n_units, AVG_DEGREE)
    L, A, Y = sample_LAY(network_adj_mat, L_EDGE_TYPE, A_EDGE_TYPE, Y_EDGE_TYPE, 
                         L_TRUE, A_TRUE, Y_TRUE, BURN_IN, L_biedge_const_var=True)

    L_est = estimate_biedge_L_params(network_dict, L, A, Y)
    Y_est = minimize(npll_Y, x0=np.random.uniform(-1, 1, 6), 
                     args=(L, A, Y, network_adj_mat)).x
    
    return causal_effects_B_U(network_adj_mat, L_est, Y_est, BURN_IN, N_SIMULATIONS)
  
        
def main():
    
    ''' evaluate true network causal effects '''
    _, network_adj_mat = create_random_network(TRUE_CAUSAL_EFFECT_N_UNIT, 
                                               AVG_DEGREE)
    causal_effect_true = causal_effects_B_U(network_adj_mat, L_TRUE, Y_TRUE, 
                                            BURN_IN, N_SIMULATIONS)
    print("True causal effect:", causal_effect_true)
    
    ''' using autog to estimate causal effects from data generated from BBU '''
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