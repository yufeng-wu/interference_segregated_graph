import os
from concurrent.futures import ProcessPoolExecutor
from setup import *

L_EDGE_TYPE = 'B'
A_EDGE_TYPE = 'U'
Y_EDGE_TYPE = 'U'

L_TRUE, A_TRUE, Y_TRUE = GET_TRUE_PARAMS(L_EDGE_TYPE, A_EDGE_TYPE, Y_EDGE_TYPE)

def parallel_helper(n_units):
    network_dict, network_adj_mat = create_random_network(n_units, AVG_DEGREE, MAX_NEIGHBORS)
    L, A, Y = sample_LAY(network_adj_mat, L_EDGE_TYPE, A_EDGE_TYPE, Y_EDGE_TYPE, 
                         L_TRUE, A_TRUE, Y_TRUE, BURN_IN)

    L_est = estimate_biedge_L_params(network_dict, L, MAX_NEIGHBORS)
    Y_est = minimize(npll_Y, x0=np.random.uniform(-1, 1, 6), 
                     args=(L, A, Y, network_adj_mat)).x
    
    return causal_effects_B_U(network_adj_mat, L_est, Y_est, BURN_IN, 
                              N_SIM_MULTIPLIER*n_units)
  
        
def main():
    ''' evaluate true network causal effects '''
    _, network_adj_mat = create_random_network(TRUE_CAUSAL_EFFECT_N_UNIT, 
                                               AVG_DEGREE, MAX_NEIGHBORS)
    causal_effect_true = causal_effects_B_U(network_adj_mat, L_TRUE, Y_TRUE, BURN_IN, 
                                            N_SIM_MULTIPLIER*TRUE_CAUSAL_EFFECT_N_UNIT)
    print("True causal effect:", causal_effect_true)
    
    ''' using our method to estimate causal effects from data generated from BUU '''
    causal_effect_ests = {}
    with ProcessPoolExecutor() as executor:
        for n_units in N_UNITS_LIST:
            print("[PROGRESS] n units", n_units)
            results = executor.map(parallel_helper, [n_units]*N_ESTIMATES)
            causal_effect_ests[f'n units {n_units}'] = list(results)
    
    ''' save results '''
    df = pd.DataFrame.from_dict(causal_effect_ests, orient='index').transpose()
    df['True Effect'] = causal_effect_true
    current_file_name = os.path.basename(__file__).split('.')[0]
    df.to_csv(f"{SAVE_OUTPUT_TO_DIR}{current_file_name}.csv", index=False)

if __name__ == "__main__":
    main()