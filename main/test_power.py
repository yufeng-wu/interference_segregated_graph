from util import create_random_network
import data_generator as dg
from maximal_independent_set import maximal_n_apart_independent_set
from nonparametric_test_undirected_vs_bidirected import prepare_data, test_edge_type
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

effective_sample_sizes = [100, 500, 1000, 5000, 10000]

ITERS_PER_SAMPLE_SIZE = 100
MIN_NB = 1
MAX_NB = 6
BURN_IN = 1000
BOOTSTRAP_ITER = 100
VERBOSE = False
MODEL = RandomForestRegressor() 
PARAM_GRID = {
    'n_estimators': [100, 500],  
    'max_depth': [None, 20],
    'min_samples_split': [2, 10]
}

results_df = pd.DataFrame()

for sample_size in effective_sample_sizes:
    print(f"Running simulations for sample size: {sample_size}")

    # Simulate under null hypothesis
    false_positives = 0
    for i in tqdm(range(ITERS_PER_SAMPLE_SIZE), desc="Null Hypothesis"):

        ind_set = []
        num_vertices = sample_size * 10
        while len(ind_set) < sample_size:
            network = create_random_network(n=num_vertices, 
                                            min_neighbors=MIN_NB, 
                                            max_neighbors=MAX_NB)
            ind_set = maximal_n_apart_independent_set(graph=network, n=5, verbose=False)
            num_vertices *= 2

        # Sample sample_size elements from ind_set
        ind_set = set(random.sample(list(ind_set), sample_size))

        edge_types = {'L' : ['U', {'prob_v_given_boundary':dg.prob_v_given_boundary_continuous, 'verbose':VERBOSE, 'burn_in':BURN_IN}]}
    
        sample_under_null = dg.sample_L_A_Y(n_samples=1, network=network, edge_types=edge_types)[0]
        df = prepare_data(sample_under_null, ind_set, network)
        lower, upper, reject_null = test_edge_type(layer="L", dataset=df, bootstrap_iter=BOOTSTRAP_ITER, model=MODEL, param_grid=PARAM_GRID)
        
        new_row = pd.DataFrame({
            'true_model': ['Undirected'], 
            'network_size': [len(network)],
            'effective_sample_size': [sample_size],
            'min_nb': [MIN_NB],
            'max_nb': [MAX_NB],
            'burn_in': [BURN_IN],
            'bootstrap_iter': [BOOTSTRAP_ITER],
            'model': [MODEL.__class__.__name__],
            'param_grid': [str(PARAM_GRID)],
            'lower': [lower],
            'upper': [upper],
            'reject_null': [reject_null]
        })

        print(new_row)
        results_df = pd.concat([results_df, new_row], ignore_index=True)

    # Simulate under alternative hypothesis
    for i in tqdm(range(ITERS_PER_SAMPLE_SIZE), desc="Alternative Hypothesis"):
        
        ind_set = []
        num_vertices = sample_size * 10
        while len(ind_set) < sample_size:
            network = create_random_network(n=num_vertices, 
                                            min_neighbors=MIN_NB, 
                                            max_neighbors=MAX_NB)
            ind_set = maximal_n_apart_independent_set(graph=network, n=5, verbose=False)
            num_vertices *= 2

        # Sample sample_size elements from ind_set
        ind_set = set(random.sample(list(ind_set), sample_size))

        sample_under_alt = dg.sample_biedge_L_layer_cont(network=network, max_neighbors=MAX_NB)
        df = prepare_data(sample_under_alt, ind_set, network)
        lower, upper, reject_null = test_edge_type(layer="L", dataset=df, bootstrap_iter=BOOTSTRAP_ITER, model=MODEL, param_grid=PARAM_GRID)
        
        new_row = pd.DataFrame({
            'true_model': ['Bidirected'], 
            'network_size': [len(network)],
            'effective_sample_size': [sample_size],
            'min_nb': [MIN_NB],
            'max_nb': [MAX_NB],
            'burn_in': [BURN_IN],
            'bootstrap_iter': [BOOTSTRAP_ITER],
            'model': [MODEL.__class__.__name__],
            'param_grid': [str(PARAM_GRID)],
            'lower': [lower],
            'upper': [upper],
            'reject_null': [reject_null]
        })

        print(new_row)

        results_df = pd.concat([results_df, new_row], ignore_index=True)
    
csv_filename = "test_power_results.csv"
results_df.to_csv(csv_filename, index=False)
print(f"Results saved to {csv_filename}")