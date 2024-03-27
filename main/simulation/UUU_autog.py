# UUU_autog means DGP is UUU and estimation is done via autog

import sys
sys.path.append('..')
from autog import *


''' set up '''
L_EDGE_TYPE = 'U'
A_EDGE_TYPE = 'U'
Y_EDGE_TYPE = 'U'

TRUE_CAUSAL_EFFECT_N_UNIT = 1000
AVG_DEGREE = 5
N_UNITS_LIST = [100, 200, 300, 400, 500]
N_ESTIMATES = 5 # number of causal effect estimates for each n_unit
BURN_IN = 200

# true parameters of the Data Generating Process
L_TRUE = np.array([-0.3, 0.4])
A_TRUE = np.array([0.3, -0.4, -0.7, -0.2])
Y_TRUE = np.array([0.2, 1, 1.5, -0.3, 1, -0.4])


def est_w_autog_parallel_helper(n_units):
    _, network_adj_mat = create_random_network(n_units, AVG_DEGREE)
    L, A, Y = sample_LAY(network_adj_mat, L_EDGE_TYPE, A_EDGE_TYPE, Y_EDGE_TYPE, L_TRUE, A_TRUE, Y_TRUE, BURN_IN)

    # estimate parameters for the L and Y layers using the autog method
    L_est = minimize(npll_L, x0=np.random.uniform(-1, 1, 2), args=(L, network_adj_mat)).x
    Y_est = minimize(npll_Y, x0=np.random.uniform(-1, 1, 6), args=(L, A, Y, network_adj_mat)).x

    # compute causal effects using estimated parameters
    Y_A1_est = estimate_causal_effects_U_U(network_adj_mat, 1, L_est, Y_est, burn_in=BURN_IN)
    Y_A0_est = estimate_causal_effects_U_U(network_adj_mat, 0, L_est, Y_est, burn_in=BURN_IN)
    return Y_A1_est - Y_A0_est
  
        
def main():
    
    ''' evaluate true network causal effects '''
    _, network_adj_mat = create_random_network(TRUE_CAUSAL_EFFECT_N_UNIT, AVG_DEGREE)
    Y_A1_true = estimate_causal_effects_U_U(network_adj_mat, 1, L_TRUE, Y_TRUE, burn_in=BURN_IN)
    Y_A0_true = estimate_causal_effects_U_U(network_adj_mat, 0, L_TRUE, Y_TRUE, burn_in=BURN_IN)
    causal_effect_true = Y_A1_true - Y_A0_true
    
    print("True causal effect:", causal_effect_true)
    
    ''' using autog to estimate causal effects from data generated from UUU '''
    causal_effect_ests = {}
    with ProcessPoolExecutor() as executor:
        for n_units in N_UNITS_LIST:
            results = executor.map(est_w_autog_parallel_helper, [n_units]*N_ESTIMATES)
            causal_effect_ests[f'n units {n_units}'] = list(results)
    
    ''' save results '''
    df = pd.DataFrame.from_dict(causal_effect_ests, orient='index').transpose()
    df['True Effect'] = causal_effect_true
    df.to_csv(f"./UUU_autog.csv", index=False)


if __name__ == "__main__":
    main()