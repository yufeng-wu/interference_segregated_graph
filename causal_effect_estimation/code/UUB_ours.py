import os
from concurrent.futures import ProcessPoolExecutor
from setup import *

L_EDGE_TYPE = 'U'
A_EDGE_TYPE = 'U'
Y_EDGE_TYPE = 'B'

L_TRUE, A_TRUE, Y_TRUE = GET_TRUE_PARAMS(L_EDGE_TYPE, A_EDGE_TYPE, Y_EDGE_TYPE)

def parallel_helper(n_units):
    network_dict, network_adj_mat = create_random_network(n_units, AVG_DEGREE, MAX_NEIGHBORS)
    L, A, Y = sample_LAY(network_adj_mat, L_EDGE_TYPE, A_EDGE_TYPE, Y_EDGE_TYPE, L_TRUE, A_TRUE, Y_TRUE, BURN_IN)
    
    return estimate_causal_effects_U_B(network_dict, network_adj_mat, L, A, Y, 
                                       burn_in=BURN_IN,
                                       n_simulations=int(N_SIM_MULTIPLIER*n_units),
                                       gibbs_select_every=GIBBS_SELECT_EVERY)

def main():
    ''' evaluate true network causal effects '''
    _, network_adj_mat = create_random_network(TRUE_CAUSAL_EFFECT_N_UNIT, AVG_DEGREE, MAX_NEIGHBORS)
    causal_effect_true = true_causal_effects_U_B(network_adj_mat, L_TRUE, Y_TRUE, 
                                                 burn_in=BURN_IN, 
                                                 n_simulations=int(N_SIM_MULTIPLIER*TRUE_CAUSAL_EFFECT_N_UNIT),
                                                 gibbs_select_every=GIBBS_SELECT_EVERY)
    
    ''' using autog to estimate causal effects from data generated from UUB '''
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