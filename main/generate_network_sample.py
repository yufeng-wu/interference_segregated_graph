# Generate a network with at least 5000 elements 5-independent set

from util import create_random_network
import data_generator as dg
from maximal_independent_set import maximal_n_apart_independent_set
import pickle
import pandas as pd

# Global variables
MIN_NB = 1
MAX_NB = 6
BURN_IN = 500
VERBOSE = True

def generate_edge_types(true_model):
    # Build edge_types based on edge types at each layer
    edge_types = {}

    if true_model[0] == 'U': 
        edge_types['L'] = ['U', {'sample_given_boundary':dg.sample_given_boundary_continuous, 'verbose':VERBOSE, 'burn_in':BURN_IN}]
    else:
        edge_types['L'] = ['B', {'U_dist':dg.U_dist_1, 'f':dg.f_1}]

    if true_model[1] == 'U': 
        edge_types['A'] = ['U', {'sample_given_boundary':dg.sample_given_boundary_continuous, 'verbose':VERBOSE, 'burn_in':BURN_IN}]
    else:
        edge_types['A'] = ['B', {'U_dist':dg.U_dist_1, 'f':dg.f_binary}]

    if true_model[2] == 'U': 
        edge_types['Y'] = ['U', {'sample_given_boundary':dg.sample_given_boundary_continuous, 'verbose':VERBOSE, 'burn_in':BURN_IN}]
    else:
        edge_types['Y'] = ['B', {'U_dist':dg.U_dist_1, 'f':dg.f_1}]

    return edge_types


def create_network_and_ind_set(sample_size):
    ind_set = []
    _FIRST_TRIAL_SCALING_FACTOR = 10
    _SUBSEQUENT_TRIALS_SCALING_FACTOR = 2
    num_vertices = sample_size * _FIRST_TRIAL_SCALING_FACTOR

    while len(ind_set) < sample_size:
        network = create_random_network(n=num_vertices, 
                                        min_neighbors=MIN_NB, 
                                        max_neighbors=MAX_NB)
        ind_set = maximal_n_apart_independent_set(graph=network, n=5, verbose=False)
        num_vertices *= _SUBSEQUENT_TRIALS_SCALING_FACTOR

    return network, ind_set


def main():
    # first generate UBU
    # next generate UBB
    # next generate BBB
    # next generate BBU
    true_models = ["UBU", "BBU", "UBB", "BBB"]
    effective_sample_size = 5000

    network, ind_set = create_network_and_ind_set(effective_sample_size)

    # save the network and ind_set somewhere
    with open("network.pkl", "wb") as file:
        pickle.dump(network, file)

    # Save the independent set as a CSV file
    pd.DataFrame(list(ind_set), columns=["subject"]).to_csv(f"5_ind_set.csv", index=False)

    for true_model in true_models:
        print(true_model)
        edge_types = generate_edge_types(true_model)
        
        # Sample a single realization from the specified Graphical Model
        GM_sample = dg.sample_L_A_Y(n_samples=1, network=network, edge_types=edge_types)[0]

        df = pd.DataFrame(GM_sample)
        df.to_csv(f"{true_model}_sample.csv", index=True)

if __name__ == "__main__":
    main()