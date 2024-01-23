from util import create_random_network, kth_order_neighborhood
import data_generator as dg
from maximal_independent_set import maximal_n_apart_independent_set
from scipy.special import expit
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

def prepare_data(dataset, ind_set, network):
    '''
    Prepare data from dataset using ind_set into the forat below. 
    Each vertex in ind_set should take one row in the output dataframe.

    Params:
        - dataset: a pandas dataframe with three columns ('L', 'A', 'Y') and each
            row is data from one subject in the graph
        - ind_set: independent set of the graph
        - graph
    '''
    
    df = pd.DataFrame()
    
    for node in ind_set:
        row = {'id': int(node)}

        for layer in ['L', 'A', 'Y']:
            row[layer] = dataset.iloc[node][layer]
            for k_order in range(1, 4):
                vals = [dataset.iloc[i][layer] for i in kth_order_neighborhood(network, node, k_order)]
                row[f'{layer}_{k_order}nb_sum'] = sum(vals)
                row[f'{layer}_{k_order}nb_avg'] = 0 if len(vals) == 0 else sum(vals) / len(vals)

        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    
    return df

def diff_test_accuracy(X, y, null_predictors, alt_predictors, model, param_grid, test_size=0.3, random_state=0):
    '''
    Using random forest classifier with grid search for parameter tuning to 
    train a null model using null_predictors and an alternative model using 
    alt_predictors. 
    
    Return the test accuracy of alternative model minus that of the null model.
    '''
    # prepare training and testing set
    y = y.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Identify overlapping row indices
    overlapping_indices = set(X_train.index) & set(X_test.index)

    # Remove overlapping rows from the training set
    X_train = X_train.drop(index=overlapping_indices)
    y_train = y_train.drop(index=overlapping_indices)

    # Convert the type of Y
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)
 
    grid_search.fit(X_train[null_predictors], y_train)
    best_null_model = grid_search.best_estimator_
    y_pred = best_null_model.predict(X_test[null_predictors])
    mse_null = mean_squared_error(y_test, y_pred)
    
    grid_search.fit(X_train[alt_predictors], y_train)
    best_alt_model = grid_search.best_estimator_
    y_pred = best_alt_model.predict(X_test[alt_predictors])
    mse_alt = mean_squared_error(y_test, y_pred)

    print("MSE Null:", mse_null)
    print("MSE Alt:", mse_alt)
    print("MSE Alt - MSE Null =", mse_alt - mse_null)
    
    return mse_alt - mse_null

def nonparametric_test(X, y, null_predictors, alt_predictors, model, param_grid, bootstrap_iter=100, percentile_lower=2.5, percentile_upper=97.5, verbose=False):
    diff_test_mses = []
    combined = pd.concat([X, y], axis=1)
    
    for i in range(bootstrap_iter):
        if verbose and i % 10 == 0:
            print(i)
        # Bootstrap (sample w replacement) a new dataset from the original X, y
        bootstrapped_combined = combined.sample(n=len(combined), replace=True, random_state=i)

        bootstrapped_X = bootstrapped_combined[X.columns]
        bootstrapped_y = bootstrapped_combined[y.columns]
        
        # compute the test statistic using the bootstrapped dataset
        diff_test_mse = diff_test_accuracy(bootstrapped_X, bootstrapped_y, 
                                           null_predictors=null_predictors, 
                                           alt_predictors=alt_predictors,
                                           model=model, 
                                           param_grid=param_grid,
                                           random_state=i)
        diff_test_mses.append(diff_test_mse)
    
    return np.percentile(diff_test_mses, [percentile_lower, percentile_upper])

def test_edge_type(layer, dataset, bootstrap_iter, model, param_grid, verbose):
    if layer == "L":
        null_predictors = ['L_1nb_sum', 'L_1nb_avg']
        alt_predictors = ['L_1nb_sum', 'L_1nb_avg', 
                          'L_2nb_sum', 'L_2nb_avg']
    if layer == "A":
        null_predictors = ['L', 
                           'L_1nb_sum', 'L_1nb_avg', 
                           'L_2nb_sum', 'L_2nb_avg',
                           'A_1nb_sum', 'A_1nb_avg']
        alt_predictors = ['L', 
                          'L_1nb_sum', 'L_1nb_avg', 
                          'L_2nb_sum', 'L_2nb_avg',
                          'L_3nb_sum', 'L_3nb_avg',
                          'A_1nb_sum', 'A_1nb_avg',
                          'A_2nb_sum', 'A_2nb_avg']
    if layer == "Y":
        null_predictors = ['L', 'A',
                           'L_1nb_sum', 'L_1nb_avg', 
                           'L_2nb_sum', 'L_2nb_avg',
                           'A_1nb_sum', 'A_1nb_avg',
                           'A_2nb_sum', 'A_2nb_avg',
                           'Y_1nb_sum', 'Y_1nb_avg']
        alt_predictors = ['L', 'A',
                           'L_1nb_sum', 'L_1nb_avg', 
                           'L_2nb_sum', 'L_2nb_avg',
                           'L_3nb_sum', 'L_3nb_avg',
                           'A_1nb_sum', 'A_1nb_avg',
                           'A_2nb_sum', 'A_2nb_avg',
                           'A_3nb_sum', 'A_3nb_avg',
                           'Y_1nb_sum', 'Y_1nb_avg',
                           'Y_2nb_sum', 'Y_2nb_avg']

    y = pd.DataFrame(dataset[layer])
    X = dataset.drop([layer, 'id'], axis=1)

    # lower is 2.5th percentile of mse alt model - mse null model
    # upper is 97.5th percentile of mse alt model - mse null model
    lower, upper = nonparametric_test(X, y, 
                                      null_predictors=null_predictors, 
                                      alt_predictors=alt_predictors, 
                                      model=model, 
                                      param_grid=param_grid,
                                      bootstrap_iter=bootstrap_iter,
                                      verbose=verbose)
    
    if upper < 0:
        # When the 97.5th percentile of  mse alt model - mse null model 
        # is less than 0, it indicates that the alt model consistently 
        # outperforms the null model.
        result = 'B' # "BIDIRECTED (REJECT NULL)"
    else:
        result = 'U' # "UNDIRECTED (FAIL TO REJECT NULL)"

    return lower, upper, result

if __name__ == "__main__":
    ''' STEP 1: Greate graph '''
    NUM_OF_VERTICES = 30000
    BURN_IN = 100
    BOOTSTRAP_ITER = 50
    VERBOSE = True
    MIN_NB = 1
    MAX_NB = 8

    network = create_random_network(n=NUM_OF_VERTICES, min_neighbors=MIN_NB, max_neighbors=MAX_NB)

    ''' STEP 2: Create data '''
    # edge_types = {'L' : ['U', {'prob_v_given_boundary':dg.sample_given_boundary_continuous, 'verbose':VERBOSE, 'burn_in':BURN_IN}]}
                   #'A' : ['U', {'sample_given_boundary':dg.sample_given_boundary_continuous, 'verbose':VERBOSE, 'burn_in':BURN_IN}]}
    #               'Y' : ['U', {'sample_given_boundary':dg.prob_v_given_boundary_3, 'verbose':VERBOSE, 'burn_in':BURN_IN}]}
    edge_types = {'L' : ['B', {'U_dist':dg.U_dist_1, 'f':dg.f_1}]}
    
    sample = dg.sample_L_A_Y(n_samples=1, network=network, edge_types=edge_types)[0]
    #sample = dg.sample_biedge_L_layer_cont(network=network, max_neighbors=MAX_NB)

    ''' STEP 3: Create and prepare data '''
    ind_set = maximal_n_apart_independent_set(graph=network, n=5, verbose=False)
    print("size of ind set", len(ind_set))
    df = prepare_data(sample, ind_set, network)

    ''' STEP 4: Perform nonparametric test '''
    model = RandomForestRegressor() 
    param_grid = {
        'n_estimators': [100, 500],  
        'max_depth': [None, 20],
        'min_samples_split': [2, 10]
    }
    lower, upper, result = test_edge_type(layer="L", dataset=df, bootstrap_iter=BOOTSTRAP_ITER, model=model, param_grid=param_grid, verbose=VERBOSE)
    print(lower, upper, result)
