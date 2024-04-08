import os
from concurrent.futures import ProcessPoolExecutor
from setup import *

L_EDGE_TYPE = 'U'
A_EDGE_TYPE = 'U'
Y_EDGE_TYPE = 'U'

L_TRUE, A_TRUE, Y_TRUE = GET_TRUE_PARAMS(L_EDGE_TYPE, A_EDGE_TYPE, Y_EDGE_TYPE)
        
def main():
    
    ''' evaluate true network causal effects '''
    _, network_adj_mat = create_random_network(TRUE_CAUSAL_EFFECT_N_UNIT, AVG_DEGREE)
    Y_A1_true = estimate_causal_effects_U_U(network_adj_mat, 1, L_TRUE, Y_TRUE, 
                                            burn_in=BURN_IN,
                                            n_simulations=N_SIM_MULTIPLIER*TRUE_CAUSAL_EFFECT_N_UNIT,
                                            gibbs_select_every=GIBBS_SELECT_EVERY)
    Y_A0_true = estimate_causal_effects_U_U(network_adj_mat, 0, L_TRUE, Y_TRUE, 
                                            burn_in=BURN_IN,
                                            n_simulations=N_SIM_MULTIPLIER*TRUE_CAUSAL_EFFECT_N_UNIT,
                                            gibbs_select_every=GIBBS_SELECT_EVERY)
    causal_effect_true = Y_A1_true - Y_A0_true
    
    print("True causal effect:", causal_effect_true)
    
    ''' using autog to estimate causal effects from data generated from UUU '''
    causal_effect_ests = {}
    with ProcessPoolExecutor() as executor:
        for n_units in N_UNITS_LIST:
            print("[PROGRESS] n units", n_units)
            params = [n_units, L_EDGE_TYPE, A_EDGE_TYPE, Y_EDGE_TYPE, L_TRUE, A_TRUE, Y_TRUE]
            results = executor.map(est_w_autog_parallel_helper, [params]*N_ESTIMATES)
            causal_effect_ests[f'n units {n_units}'] = list(results)
    
    ''' save results '''
    df = pd.DataFrame.from_dict(causal_effect_ests, orient='index').transpose()
    df['True Effect'] = causal_effect_true
    current_file_name = os.path.basename(__file__).split('.')[0]
    df.to_csv(f"{SAVE_OUTPUT_TO_DIR}{current_file_name}.csv", index=False)

if __name__ == "__main__":
    main()