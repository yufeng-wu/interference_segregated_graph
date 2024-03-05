from autog import *

def main():
    # set up
    n_units_true_causal_effect = 4000#9000
    n_bootstraps = 100
    n_units_list = [1000, 3000, 5000, 7000, 9000]
    burn_in = 100#200
    which_test = "UBB"
    
    # evaluate true network causal effects 
    network = random_network_adjacency_matrix(n_units_true_causal_effect, 1, 6)

    true_L = np.array([-0.3, 0.4])
    true_A = np.array([0, 1, 0.3, -0.4, -0.7, 0.2])
    # true_Y = np.array([0, 1, 0.5, 0.1, 1, -0.3, 0.6, 0.4])
    true_Y = np.array([0, 0.1, 0.5, 0.1, 1, -0.3, 0.6, -10])

    Y_A1 = estimate_causal_effects_U_B(network, 1, true_L, true_Y, burn_in, K=10)
    Y_A0 = estimate_causal_effects_U_B(network, 0, true_L, true_Y, burn_in, K=10)
    true_causal_effect = Y_A1 - Y_A0
    print("True Causal Effects:", true_causal_effect)

    # Using the parallelized function for auto-g estimation
    # est_causal_effects = bootstrap_autog(
    #     L_edge_type="U",
    #     A_edge_type="B",
    #     Y_edge_type="B",
    #     n_units_list=n_units_list, 
    #     n_bootstraps=n_bootstraps, 
    #     true_L=true_L, 
    #     true_A=true_A, 
    #     true_Y=true_Y,
    #     burn_in=burn_in
    # )

    # df = pd.DataFrame.from_dict(est_causal_effects, orient='index').transpose()
    # df['True Effect'] = true_causal_effect
    # df.to_csv(f"./autog_{which_test}_results.csv", index=False)
    # print(f"Results saved.")

if __name__ == "__main__":
    main()
