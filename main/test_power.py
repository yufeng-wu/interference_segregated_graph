from util import create_random_network
import data_generator as dg
from maximal_independent_set import maximal_n_apart_independent_set
from nonparametric_test_undirected_vs_bidirected import prepare_data, test_edge_type
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import os
from datetime import datetime

# Global variables
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
FILENAME_TO_SAVE = f"TEST_POWER_RESULT_{timestamp}.csv"
ITERS_PER_SAMPLE_SIZE = 1
MIN_NB = 1
MAX_NB = 6
BURN_IN = 100
BOOTSTRAP_ITER = 20
VERBOSE = True
ML_MODEL = RandomForestRegressor() 
PARAM_GRID = {
    'n_estimators': [100, 500],  
    'max_depth': [None, 20]#,
    #'min_samples_split': [2, 10]
}


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
        edge_types['A'] = ['B', {'U_dist':dg.U_dist_1, 'f':dg.f_1}]

    if true_model[2] == 'U': 
        edge_types['Y'] = ['U', {'sample_given_boundary':dg.sample_given_boundary_continuous, 'verbose':VERBOSE, 'burn_in':BURN_IN}]
    else:
        edge_types['Y'] = ['B', {'U_dist':dg.U_dist_1, 'f':dg.f_1}]

    return edge_types


def create_network_and_ind_set(sample_size):
    ind_set = []
    _FIRST_TRIAL_SCALING_FACTOR = 20
    _SUBSEQUENT_TRIALS_SCALING_FACTOR = 2
    num_vertices = sample_size * _FIRST_TRIAL_SCALING_FACTOR

    while len(ind_set) < sample_size:
        network = create_random_network(n=num_vertices, 
                                        min_neighbors=MIN_NB, 
                                        max_neighbors=MAX_NB)
        ind_set = maximal_n_apart_independent_set(graph=network, n=5, verbose=False)
        num_vertices *= _SUBSEQUENT_TRIALS_SCALING_FACTOR

    # Sample (without replace) sample_size elements from ind_set
    ind_set = set(random.sample(list(ind_set), sample_size))

    return network, ind_set


def main():
    columns = ['true_model', 'network_size', 'effective_sample_size', 
               'min_neighbors', 'max_neighbors', 'burn_in_(if_applies)', 
               'bootstrap_iters', 'ML_model_name', 'tuning_param_grid', 
               'L_lower', 'L_upper', 'L_result', 
               'A_lower', 'A_upper', 'A_result', 
               'Y_lower', 'Y_upper', 'Y_result']

    # Check if the CSV file exists and write the header if it doesn't
    if not os.path.exists(FILENAME_TO_SAVE):
        pd.DataFrame(columns=columns).to_csv(FILENAME_TO_SAVE, index=False)

    # 8 cases in total. U = undirected, B = Bidirected. 
    # The 1st letter corresponds to the type of edge in the L layer, 
    # the 2nd corresponds to the A layer, and the 3rd corresponds to the Y layer.
    true_models = ["UUU", "BUU", "UBU", "UUB", "UBB", "BUB", "BBU", "BBB"]
    effective_sample_sizes = [3000, 1000, 300]

    for sample_size in effective_sample_sizes:

        for true_model in true_models:
            edge_types = generate_edge_types(true_model)
            
            for _ in tqdm(range(ITERS_PER_SAMPLE_SIZE), desc=true_model+"_"+str(sample_size)):
                network, ind_set = create_network_and_ind_set(sample_size)
                
                # Sample a single realization from the specified Graphical Model
                GM_sample = dg.sample_L_A_Y(n_samples=1, network=network, edge_types=edge_types)[0]

                # Prepare data using 5-independent set for later ML models
                df = prepare_data(GM_sample, ind_set, network)
               
                # Test edge types (-- or <->) for three layers
                L_lower, L_upper, L_result = test_edge_type(layer="L", dataset=df, bootstrap_iter=BOOTSTRAP_ITER, model=ML_MODEL, param_grid=PARAM_GRID, verbose=VERBOSE)
                A_lower, A_upper, A_result = test_edge_type(layer="A", dataset=df, bootstrap_iter=BOOTSTRAP_ITER, model=ML_MODEL, param_grid=PARAM_GRID, verbose=VERBOSE)
                Y_lower, Y_upper, Y_result = test_edge_type(layer="Y", dataset=df, bootstrap_iter=BOOTSTRAP_ITER, model=ML_MODEL, param_grid=PARAM_GRID, verbose=VERBOSE)
                
                # Create a single row with results from all layers
                new_row = pd.DataFrame({
                    'true_model': [true_model],
                    'network_size': [len(network)],
                    'effective_sample_size': [sample_size],
                    'min_neighbors': [MIN_NB],
                    'max_neighbors': [MAX_NB],
                    'burn_in_(if_applies)': [BURN_IN],
                    'bootstrap_iters': [BOOTSTRAP_ITER],
                    'ML_model_name': [ML_MODEL.__class__.__name__],
                    'tuning_param_grid': [str(PARAM_GRID)],
                    'L_lower': [L_lower],
                    'L_upper': [L_upper],
                    'L_result': [L_result],
                    'A_lower': [A_lower],
                    'A_upper': [A_upper],
                    'A_result': [A_result],
                    'Y_lower': [Y_lower],
                    'Y_upper': [Y_upper],
                    'Y_result': [Y_result]
                })  

                new_row.to_csv(FILENAME_TO_SAVE, mode='a', header=False, index=False)

    print(f"[COMPLETE] All results saved to {FILENAME_TO_SAVE}")

if __name__ == "__main__":
    main()