from autog import *
from maximal_independent_set import maximal_n_apart_independent_set
from util import *

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def main():
    # 1) set up
    n_units_true_causal_effect = 10000
    n_simulations = n_draws_from_pL = 100
    gibbs_select_every = 3
    n_bootstraps = 100
    n_units_list = [1000, 3000, 5000, 7000, 9000]
    burn_in = 100
    edge_types = "UBB"
    
    # true parameters of the Data Generating Process
    true_L = np.array([-0.3, 0.4])
    true_A = np.array([0, 1, 0.3, -0.4, -0.7, 0.2])
    true_Y = np.array([0, 1, 0.5, 0.1, 1, -0.3, 0.6, 0.5])
    
    # 2) evaluate true causal effects just once, using true params specified above
    network_dict, network_adj_mat = create_random_network(n_units_true_causal_effect, 1, 6)
    true_causal_effect = true_causal_effects_U_B(network_adj_mat, 
                                                 params_L=true_L, 
                                                 params_Y=true_Y, 
                                                 burn_in=burn_in, 
                                                 n_simulations=n_simulations)
    
    # 3) generate data from UBB models, estiamte with our proposed U_B estimator
    args_dict_U_B = {
        'n_draws_from_pL' : n_draws_from_pL,
        'gibbs_select_every' : gibbs_select_every
    }
    
    results_UBB = consistency_test(estimate_with_wrapper=estimate_causal_effects_U_B_wrapper,
                     n_bootstraps=n_bootstraps,
                     n_units_list=n_units_list,
                     L_edge_type=edge_types[0],
                     A_edge_type=edge_types[1],
                     Y_edge_type=edge_types[2],
                     true_L=true_L,
                     true_A=true_A,
                     true_Y=true_Y,
                     burn_in=burn_in,
                     args_dict=args_dict_U_B)
    
    # TODO: we can put this in another file too
    # generate data from UBB models, estiamte with autog
    # results_autog = consistency_test(estimate_with_wrapper=autog_wrapper,
    #                  n_bootstraps=n_bootstraps,
    #                  n_units_list=n_units_list,
    #                  L_edge_type=edge_types[0],
    #                  A_edge_type=edge_types[1],
    #                  Y_edge_type=edge_types[2],
    #                  true_L=true_L,
    #                  true_A=true_A,
    #                  true_Y=true_Y,
    #                  burn_in=burn_in,
    #                  args_dict={})
    
   
if __name__ == "__main__":
    main()
