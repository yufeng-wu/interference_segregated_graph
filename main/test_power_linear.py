from nonparametric_test_undirected_vs_bidirected import prepare_data, test_edge_type
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
import random
import os
from datetime import datetime
import pickle
import xgboost as xgb


# Global variables
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
FOLDER_TO_SAVE = "../result/"
FILENAME_TO_SAVE = FOLDER_TO_SAVE + f"TEST_POWER_LINEAR_RESULT_{timestamp}.csv"

ITERS_PER_SAMPLE_SIZE = 100
TEST_BOOTSTRAP_ITERS = 100
VERBOSE = True

ML_MODEL = LinearRegression()
PARAM_GRID = {}

# Instantiate an XGBoost regressor object
# ML_MODEL = xgb.XGBRegressor(objective ='reg:squarederror')

# Define a parameter grid to search over (example parameters)
# PARAM_GRID = {
#     'max_depth': [3, 10],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'subsample': [0.7, 1]
# }

# ML_MODEL = RandomForestRegressor()
# PARAM_GRID = {
#         'n_estimators': [100],  
#         'max_depth': [None, 20],
#         'min_samples_split': [2, 10]
#     }

# ML_MODEL = DecisionTreeRegressor()
# PARAM_GRID = {
#     'max_depth': [None, 10], 
#     'min_samples_split': [2, 10],
#     'min_samples_leaf': [1, 5]
# }

DATA_SOURCE = "../data/simulation/"

def main():

    columns = ['true_model', 'data_source', 'network_size', 'effective_sample_size',
               'test_bootstrap_iters', 'ML_model_name', 'tuning_param_grid',
               'L_lower', 'L_upper', 'L_result', 'Y_lower', 'Y_upper', 'Y_result']

    # Check if the CSV file exists and write the header if it doesn't
    if not os.path.exists(FILENAME_TO_SAVE):
        pd.DataFrame(columns=columns).to_csv(FILENAME_TO_SAVE, index=False)

    true_models = ["BBU", "UBU", "UBB", "BBB"]
    effective_sample_sizes = [5000]

    # read in network.pkl # fix this using DATA_SOURCE11
    with open(os.path.join(DATA_SOURCE, 'network.pkl'), 'rb') as file:
        network = pickle.load(file)

    # Load the pre-specified 5-independent set using DATA_SOURCE
    ind_set_full = pd.read_csv(os.path.join(DATA_SOURCE, '5_ind_set.csv'))['subject'].tolist()
    
    for true_model in true_models:
        GM_sample = pd.read_csv(os.path.join(DATA_SOURCE, f"{true_model}_sample.csv"))

        for sample_size in effective_sample_sizes:

            for _ in range(ITERS_PER_SAMPLE_SIZE):

                # Randomly select sample_size elements from the pre-specified 5-ind set
                ind_set = random.sample(ind_set_full, sample_size)

                # Prepare data using 5-independent set for later ML models
                df = prepare_data(GM_sample, ind_set, network)
                
                # Test edge types (-- or <->) for three layers
                L_lower, L_upper, L_result = test_edge_type(layer="L", dataset=df, bootstrap_iter=TEST_BOOTSTRAP_ITERS, model=ML_MODEL, param_grid=PARAM_GRID, verbose=VERBOSE)
                Y_lower, Y_upper, Y_result = test_edge_type(layer="Y", dataset=df, bootstrap_iter=TEST_BOOTSTRAP_ITERS, model=ML_MODEL, param_grid=PARAM_GRID, verbose=VERBOSE)
                    
                # Create a single row with results from all layers and append 
                # necessary information 
                new_row = pd.DataFrame([{
                    'true_model': true_model, 
                    'data_source': DATA_SOURCE,
                    'network_size': len(network), 
                    'effective_sample_size': sample_size,
                    'test_bootstrap_iters': TEST_BOOTSTRAP_ITERS, 
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