from util import create_random_network, kth_order_neighborhood
import data_generator as dg
from maximal_independent_set import maximal_n_apart_independent_set
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
import warnings

# Filter out the UserWarning raised by the Ridge model
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn.linear_model._ridge')

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


# def _remove_overlaps(X_train, X_val, X_test, y_train, y_val, y_test):
#     """
#     Remove overlapping indices among training, validation, and test sets.

#     Args:
#     X_train, X_val, X_test: DataFrames representing the feature sets.
#     y_train, y_val, y_test: Series representing the target sets.

#     Returns:
#     Adjusted DataFrames and Series without overlapping indices.
#     """
#     # Identify overlapping row indices among the sets
#     overlapping_indices = (set(X_train.index) & set(X_val.index)) | \
#                           (set(X_train.index) & set(X_test.index)) | \
#                           (set(X_val.index) & set(X_test.index))
#     overlapping_indices = list(overlapping_indices)

#     # Append overlapping rows from validation and test sets to the training set
#     X_train = pd.concat([X_train, X_val.loc[X_val.index.intersection(overlapping_indices)], 
#                                   X_test.loc[X_test.index.intersection(overlapping_indices)]])
#     y_train = pd.concat([y_train, y_val.loc[y_val.index.intersection(overlapping_indices)], 
#                                   y_test.loc[y_test.index.intersection(overlapping_indices)]])

#     # Remove overlapping rows from validation and test sets
#     X_val = X_val.drop(index=overlapping_indices, errors='ignore')
#     X_test = X_test.drop(index=overlapping_indices, errors='ignore')
#     y_val = y_val.drop(index=overlapping_indices, errors='ignore')
#     y_test = y_test.drop(index=overlapping_indices, errors='ignore')

#     return X_train, X_val, X_test, y_train, y_val, y_test


# def diff_test_accuracy(X, y, null_predictors, alt_predictors, model, param_grid, test_size=0.2, val_size=0.15, random_state=0):
    
#     # Split the data into training + validation, and test sets
#     X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

#     # Further split the training data into training and validation sets
#     X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, random_state=random_state)

#     # Remove overlapping indices
#     X_train, X_val, X_test, y_train, y_val, y_test = _remove_overlaps(X_train, X_val, X_test, y_train, y_val, y_test)

#     # Convert the type of Y
#     y_train = np.ravel(y_train)
#     y_val = np.ravel(y_val)
#     y_test = np.ravel(y_test)

#     # Function to get all combinations of parameters in param_grid
#     def param_combinations(param_grid):
#         keys = param_grid.keys()
#         values = param_grid.values()
#         for instance in product(*values):
#             yield dict(zip(keys, instance))

#     # Function to train model and return MSE on validation set for given parameters
#     def train_and_evaluate(predictors, params):
#         model.set_params(**params)
#         model.fit(X_train[predictors], y_train)
#         y_pred_val = model.predict(X_val[predictors])
#         return mean_squared_error(y_val, y_pred_val)

#     # Evaluate and find the best model for null_predictors and alt_predictors
#     def find_best_model(predictors):
#         best_mse = float('inf')
#         best_params = None
#         for params in param_combinations(param_grid):
#             mse = train_and_evaluate(predictors, params)
#             if mse < best_mse:
#                 best_mse = mse
#                 best_params = params
#         return best_params

#     # Find the best model for null_predictors
#     best_params_null = find_best_model(null_predictors)
#     model.set_params(**best_params_null)
#     model.fit(X_train[null_predictors], y_train)
#     mse_null_test = mean_squared_error(y_test, model.predict(X_test[null_predictors]))

#     # Find the best model for alt_predictors
#     best_params_alt = find_best_model(alt_predictors)
#     model.set_params(**best_params_alt)
#     model.fit(X_train[alt_predictors], y_train)
#     mse_alt_test = mean_squared_error(y_test, model.predict(X_test[alt_predictors]))

#     return mse_alt_test - mse_null_test


# def diff_test_accuracy(X, y, null_predictors, alt_predictors, model, param_grid, test_size=0.3, random_state=0):
#     '''
#     Using random forest classifier with grid search for parameter tuning to 
#     train a null model using null_predictors and an alternative model using 
#     alt_predictors. 
    
#     Return the test accuracy of alternative model minus that of the null model.
#     '''

#     # prepare training and testing set
#     y = y.astype(float)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

#     # Identify overlapping row indices
#     overlapping_indices = set(X_train.index) & set(X_test.index)

#     # Remove overlapping rows from the training set
#     X_train = X_train.drop(index=overlapping_indices)
#     y_train = y_train.drop(index=overlapping_indices)
#     print(X_train.index)
#     print(len(list(X_train.index)), len(set(X_train.index)))

#     # Convert the type of Y
#     y_train = np.ravel(y_train)
#     y_test = np.ravel(y_test)

#     # If the model is a linear regression model, skip parameter tuning
#     if isinstance(model, LinearRegression):
#         null_model = model.fit(X_train[null_predictors], y_train)
#         mse_null = mean_squared_error(y_test, null_model.predict(X_test[null_predictors]))

#         alt_model = model.fit(X_train[alt_predictors], y_train)
#         mse_alt = mean_squared_error(y_test, alt_model.predict(X_test[alt_predictors]))
        
#         # print("MSE Null:", mse_null)
#         # print("MSE Alt:", mse_alt)
#         # print("MSE Alt - MSE Null =", mse_alt - mse_null)
        
#         return mse_alt - mse_null

#     grid_search = GridSearchCV(estimator=model, 
#                                param_grid=param_grid, 
#                                cv=10, 
#                                n_jobs=-1, 
#                                scoring="neg_mean_squared_error")
 
#     grid_search.fit(X_train[null_predictors], y_train)

#     print("\nAll results:")
#     for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
#         print("Mean squared error: {:.3f} for {}".format(-mean_score, params))


#     best_null_model = grid_search.best_estimator_
#     y_pred = best_null_model.predict(X_test[null_predictors])
#     mse_null = mean_squared_error(y_test, y_pred)

#     print("best params: ", grid_search.best_params_)
    
#     grid_search.fit(X_train[alt_predictors], y_train)

#     print("\nAll results:")
#     for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
#         print("Mean squared error: {:.3f} for {}".format(-mean_score, params))

#     best_alt_model = grid_search.best_estimator_
#     y_pred = best_alt_model.predict(X_test[alt_predictors])
#     mse_alt = mean_squared_error(y_test, y_pred)

#     print("best params: ", grid_search.best_params_)

#     print("MSE Null:", mse_null)
#     print("MSE Alt:", mse_alt)
#     print("MSE Alt - MSE Null =", mse_alt - mse_null)
    
#     return mse_alt - mse_null


def _tune_hyperparams(X_train, y_train, X_val, y_val, predictors, model, param_grid):
    """
    Find the best hyperparameters using the validation set.
    """
    best_mse = float('inf')
    best_params = None  
    for params in ParameterGrid(param_grid):
        model.set_params(**params)
        model.fit(X_train[predictors], np.ravel(y_train))
        mse_val = mean_squared_error(y_val, model.predict(X_val[predictors]))
        if mse_val < best_mse:
            best_mse = mse_val
            best_params = params
    return best_params, best_mse

def _train_and_eval(X_train, y_train, X_test, y_test, predictors, model, best_params):
    """
    Train the model with the best parameters and evaluate on the test set.
    """
    model.set_params(**best_params)
    model.fit(X_train[predictors], np.ravel(y_train))
    return mean_squared_error(y_test, model.predict(X_test[predictors]))

def diff_test_accuracy(X_train, y_train, X_val, y_val, X_test, y_test, 
                       null_predictors, alt_predictors, model, param_grid):
    """
    Train a null model and an alternative model. 
    Calculate and return the difference in test MSE.
    """
    best_params_null, _ = _tune_hyperparams(X_train, y_train, X_val, y_val, 
                                            null_predictors, model, param_grid)
    mse_test_null = _train_and_eval(X_train, y_train, X_test, y_test, 
                                    null_predictors, model, best_params_null)

    best_params_alt, _ = _tune_hyperparams(X_train, y_train, X_val, y_val, 
                                           alt_predictors, model, param_grid)
    mse_test_alt = _train_and_eval(X_train, y_train, X_test, y_test, 
                                   alt_predictors, model, best_params_alt)

    print("MSE Test Null:", mse_test_null)
    print("MSE Test Alt:", mse_test_alt)
    print("MSE Test Alt - MSE Test Null =", mse_test_alt - mse_test_null)

    return mse_test_alt - mse_test_null

def nonparametric_test(X, y, null_predictors, alt_predictors, model, param_grid,
                        test_size=0.2, val_size=0.2, bootstrap_iter=100, 
                        percentile_lower=2.5, percentile_upper=97.5, verbose=False):
    diff_test_mses = []

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=42)
    
    combined_train = pd.concat([X_train, y_train], axis=1)
    
    for i in range(bootstrap_iter):
        if verbose and i % 10 == 0:
            print("testing iteration:", i)

        # Bootstrap (sample with replacement) a new dataset from the training data
        bootstrapped_combined_train = combined_train.sample(n=len(combined_train), replace=True, random_state=i)

        bootstrapped_X_train = bootstrapped_combined_train[X_train.columns]
        bootstrapped_y_train = bootstrapped_combined_train[y_train.columns]
        
        # Compute the test statistic
        diff_test_mse = diff_test_accuracy(X_train=bootstrapped_X_train, 
                                           y_train=bootstrapped_y_train, 
                                           X_val=X_val, 
                                           y_val=y_val, 
                                           X_test=X_test, 
                                           y_test=y_test,
                                           null_predictors=null_predictors, 
                                           alt_predictors=alt_predictors,
                                           model=model, 
                                           param_grid=param_grid)
        
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
    NUM_OF_VERTICES = 100000
    BURN_IN = 200
    BOOTSTRAP_ITER = 40
    VERBOSE = True
    MIN_NB = 1
    MAX_NB = 6

    network = create_random_network(n=NUM_OF_VERTICES, min_neighbors=MIN_NB, max_neighbors=MAX_NB)

    ''' STEP 2: Create data '''
    # edge_types = {'L' : ['U', {'sample_given_boundary':dg.sample_given_boundary_continuous, 'verbose':VERBOSE, 'burn_in':BURN_IN}],
    #                'A' : ['U', {'sample_given_boundary':dg.sample_given_boundary_continuous, 'verbose':VERBOSE, 'burn_in':BURN_IN}],
    #               'Y' : ['U', {'sample_given_boundary':dg.sample_given_boundary_continuous, 'verbose':VERBOSE, 'burn_in':BURN_IN}]}
    edge_types = {'L' : ['B', {'U_dist':dg.U_dist_1, 'f':dg.f_non_linear}],
                  'A' : ['B', {'U_dist':dg.U_dist_1, 'f':dg.f_non_linear}],
                  'Y' : ['B', {'U_dist':dg.U_dist_1, 'f':dg.f_non_linear}]}
    
    sample = dg.sample_L_A_Y(n_samples=1, network=network, edge_types=edge_types)[0]
    print("sampling done")
    #sample = dg.sample_biedge_L_layer_cont(network=network, max_neighbors=MAX_NB)

    ''' STEP 3: Create and prepare data '''
    ind_set = maximal_n_apart_independent_set(graph=network, n=5, verbose=False)
    print("size of ind set", len(ind_set))
    df = prepare_data(sample, ind_set, network)

    ''' STEP 4: Perform nonparametric test '''
    # model = RandomForestRegressor() 
    # param_grid = {
    #     'n_estimators': [100],  
    #     'max_depth': [None, 10],
    #     'min_samples_split': [2, 20],
    #     'min_samples_leaf': [1, 10]
    # }
    model = KernelRidge()
    param_grid = {
        'alpha': [0.1, 1.0, 10.0],
        'kernel': ['linear', 'poly', 'rbf'],
        'degree': [2, 3],  # For polynomial kernel
        'coef0': [0.1, 1.0],  # For polynomial kernel
        'gamma': [0.01, 0.1, 1.0],  # For RBF kernel
    }
    lower, upper, result = test_edge_type(layer="L", dataset=df, bootstrap_iter=BOOTSTRAP_ITER, model=model, param_grid=param_grid, verbose=VERBOSE)
    print("L result: ", lower, upper, result)
    # lower, upper, result = test_edge_type(layer="A", dataset=df, bootstrap_iter=BOOTSTRAP_ITER, model=model, param_grid=param_grid, verbose=VERBOSE)
    # print("A result: ", lower, upper, result)
    # lower, upper, result = test_edge_type(layer="Y", dataset=df, bootstrap_iter=BOOTSTRAP_ITER, model=model, param_grid=param_grid, verbose=VERBOSE)
    # print("Y result: ", lower, upper, result)
