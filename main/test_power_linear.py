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
import pickle

# Global variables
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
FILENAME_TO_SAVE = f"TEST_POWER_LINEAR_RESULT_{timestamp}.csv"

ITERS_PER_SAMPLE_SIZE = 100
TEST_BOOTSTRAP_ITER = 100
VERBOSE = True
ML_MODEL = LinearRegression()

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

    true_models = ["UBU", "BBU", "UBB", "BBB"]
    effective_sample_sizes = [500, 1000, 2000]

    # read in network.pkl
    with open('network.pkl', 'rb') as file:
        network = pickle.load(file)

    # Load the pre-specified 5-independent set
    ind_set_full = pd.read_csv('5_ind_set.csv')['subject'].tolist()
    
    for true_model in true_models:
        GM_sample = pd.read_csv(f"{true_model}_sample.csv")

        for sample_size in effective_sample_sizes:

            for i in range(ITERS_PER_SAMPLE_SIZE):

                # Randomly select sample_size elements from the pre-specified 5-ind set
                ind_set = random.sample(ind_set_full, sample_size)

                # Prepare data using 5-independent set for later ML models
                df = prepare_data(GM_sample, ind_set, network)
                
                # Test edge types (-- or <->) for three layers
                L_lower, L_upper, L_result = test_edge_type(layer="L", dataset=df, bootstrap_iter=TEST_BOOTSTRAP_ITER, model=ML_MODEL, param_grid=PARAM_GRID, verbose=VERBOSE)
                Y_lower, Y_upper, Y_result = test_edge_type(layer="Y", dataset=df, bootstrap_iter=TEST_BOOTSTRAP_ITER, model=ML_MODEL, param_grid=PARAM_GRID, verbose=VERBOSE)
                    
                # Create a single row with results from all layers and append 
                # necessary information 
                new_row = pd.DataFrame([{
                    'true_model': true_model, 
                    'network_size': len(network), 
                    'effective_sample_size': sample_size,
                    'min_neighbors': min(network.degree()), 
                    'max_neighbors': max(network.degree()), 
                    'burn_in_(if_applies)': 'N/A', 
                    'bootstrap_iters': BOOTSTRAP_ITER, 
                    'ML_model_name': ML_MODEL.__class__.__name__, 
                    'tuning_param_grid': str(PARAM_GRID),
                    'L_lower': L_lower, 
                    'L_upper': L_upper, 
                    'L_result': L_result,
                    'Y_lower': Y_lower, 
                    'Y_upper': Y_upper, 
                    'Y_result': Y_result
                }])

                new_row.to_csv(FILENAME_TO_SAVE, mode='a', header=False, index=False)

    print(f"[COMPLETE] All results saved to {FILENAME_TO_SAVE}")

if __name__ == "__main__":
    main()