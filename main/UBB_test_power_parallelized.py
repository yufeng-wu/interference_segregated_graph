from multiprocessing import Pool
import os
import random
import pandas as pd
import pickle
from datetime import datetime
from nonparametric_test_undirected_vs_bidirected import prepare_data, test_edge_type
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
from joblib import parallel_backend

# Filter out the UserWarning related to nested parallelism
warnings.filterwarnings('ignore', category=UserWarning, message='.*Loky-backed parallel loops cannot be called in a multiprocessing.*')

# Global variables
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
FOLDER_TO_SAVE = "../result/"
FILENAME_TO_SAVE = FOLDER_TO_SAVE + f"UBB_FINAL_TEST_RESULT_{timestamp}.csv"

ITERS_PER_SAMPLE_SIZE = 20
TEST_BOOTSTRAP_ITERS = 100
VERBOSE = True

# ML_MODEL = LinearRegression()
# PARAM_GRID = {}

ML_MODEL = RandomForestRegressor()
PARAM_GRID = {
    'n_estimators': [100],  
    'max_depth': [None, 10],
    'min_samples_split': [2, 20],
    'min_samples_leaf': [1, 10]
}

DATA_SOURCE = "../data/simulation/"

def process_iteration(params):
    true_model, sample_size, iteration = params
    network_path = os.path.join(DATA_SOURCE, 'network.pkl')
    ind_set_path = os.path.join(DATA_SOURCE, '5_ind_set.csv')
    sample_data_path = os.path.join(DATA_SOURCE, f"{true_model}_sample.csv")

    with open(network_path, 'rb') as file:
        network = pickle.load(file)

    ind_set_full = pd.read_csv(ind_set_path)['subject'].tolist()
    GM_sample = pd.read_csv(sample_data_path)

    ind_set = random.sample(ind_set_full, sample_size)
    df = prepare_data(GM_sample, ind_set, network)

    L_lower, L_upper, L_result = test_edge_type(layer="L", dataset=df, bootstrap_iter=TEST_BOOTSTRAP_ITERS, model=ML_MODEL, param_grid=PARAM_GRID, verbose=VERBOSE)
    Y_lower, Y_upper, Y_result = test_edge_type(layer="Y", dataset=df, bootstrap_iter=TEST_BOOTSTRAP_ITERS, model=ML_MODEL, param_grid=PARAM_GRID, verbose=VERBOSE)

    return {
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
    }

def main():
    true_models = ["UBB"]
    effective_sample_sizes = [500, 1000, 2000, 3000, 4000, 5000]

    columns = ['true_model', 'data_source', 'network_size', 'effective_sample_size',
               'test_bootstrap_iters', 'ML_model_name', 'tuning_param_grid',
               'L_lower', 'L_upper', 'L_result', 'Y_lower', 'Y_upper', 'Y_result']

    if not os.path.exists(FILENAME_TO_SAVE):
        pd.DataFrame(columns=columns).to_csv(FILENAME_TO_SAVE, index=False)

    for tm in true_models:
        for ess in effective_sample_sizes:
            iteration_params = [(tm, ess, i) for i in range(ITERS_PER_SAMPLE_SIZE)]

            with Pool() as pool:
                iteration_results = pool.map(process_iteration, iteration_params)

            pd.DataFrame(iteration_results).to_csv(FILENAME_TO_SAVE, mode='a', header=False, index=False)

    print(f"[COMPLETE] All results saved to {FILENAME_TO_SAVE}")

if __name__ == "__main__":
    main()
