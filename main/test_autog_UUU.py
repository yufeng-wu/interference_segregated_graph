from autog import *

def main():
    # set up
    n_units_true_causal_effect = 10000
    n_bootstraps = 100
    n_units_list = [1000, 3000, 5000, 7000, 9000]
    burn_in = 200
    which_test = "UUU"

    # evaluate true network causal effects
    network = random_network_adjacency_matrix(n_units_true_causal_effect, 1, 6)

    # true parameters of the Data Generating Process (DGP)
    true_L = np.array([-0.3, 0.4])
    true_A = np.array([0.3, -0.4, -0.7, -0.2])
    true_Y = np.array([0.2, 1, 1.5, -0.3, 1, -0.4])

    # evaluate true causal effects just once, using true params specified above
    Y_A1 = estimate_causal_effects_U_U(network, A_value=1, params_L=true_L, params_Y=true_Y, burn_in=200, K=50, N=3)
    Y_A0 = estimate_causal_effects_U_U(network, A_value=0, params_L=true_L, params_Y=true_Y, burn_in=200, K=50, N=3)
    true_causal_effect = Y_A1 - Y_A0
    print("True Causal Effects:", true_causal_effect)

    # using the parallelized function for auto-g estimation of causal effects
    est_causal_effects = autog_consistency_test(
        n_units_list=n_units_list, 
        L_edge_type="U",
        A_edge_type="U",
        Y_edge_type="U",
        n_bootstraps=n_bootstraps, 
        true_L=true_L, 
        true_A=true_A, 
        true_Y=true_Y, 
        burn_in=burn_in
    )

    # save the estimated causal effects together with the true causal effect
    df = pd.DataFrame.from_dict(est_causal_effects, orient='index').transpose()
    df['True Effect'] = true_causal_effect
    df.to_csv(f"./autog_{which_test}_results.csv", index=False)
    
    print("Results saved.")

if __name__ == "__main__":
    main()
