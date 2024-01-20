from util import create_random_network, kth_order_neighborhood
import data_generator as dg
from maximal_independent_set import maximal_n_apart_independent_set
from scipy.special import expit
import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

def U_dist():
    '''
    Define the distribution: U ~ U_dist.
    '''
    return np.random.normal(0, 1)

def f(U_values):
    '''
    Define the function f that calculates V based on its neighboring U values
    and returns a binary value.
    '''
    noise = np.random.normal(0, 0.1)
    linear_sum = 17 * sum(U_values) + noise
    prob = expit(linear_sum)  # Sigmoid function to get a value between 0 and 1
    return np.random.binomial(1, prob)  # Sample from a Bernoulli distribution

def prob_v_given_neighbors(data, params=[0.25, 0.3]):
    '''
    Define the parametric form for the conditional probability P(V_i = 1 | -V_i)
    using only the V_neighbors as input. V_i is a binary variable that is either
    0 or 1. The parameters a0 and a1 are hard-coded inside the function.

    Params:
        - V_neighbors: array-like, containing the values of V's neighbors
    
    Return:
        - a float that represents the conditional probability
    '''
    # Parameters can be hard-coded or defined as constants elsewhere
    a0 = params[0]
    a1 = params[1]

    V_nb_values = data["V_nb_values"]

    return expit(a0 + a1 * np.sum(V_nb_values))

def prepare_data(dataset, ind_set, network, max_neighbors):
    '''
    Prepare data from dataset using ind_set into the forat below. 
    Each vertex in ind_set should take one row in the output dataframe.

    Params:
        - dataset: a pandas dataframe with three columns ('L', 'A', 'Y') and each
            row is data from one subject in the graph
        - ind_set: independent set of the graph
        - graph
    '''
    
    df = pd.DataFrame(columns=['id', 'L', 'L_1nb', 'L_2nb'])
    
    for node in ind_set:
        row = {'id': int(node), 'L': dataset.iloc[node]['L']}

        # Rank node's neighbors from highest to lowest degree
        neighbors_ordered_by_degree = sorted(network[node], key=lambda x: len(network[x]), reverse=True)
        
        # L_1nb: Build a vector of length max_neighbors and assign L values
        L_1nb_values = [dataset.loc[neighbor]['L'] for neighbor in neighbors_ordered_by_degree]
        row['L_1nb'] = L_1nb_values + [-1] * (max_neighbors - len(L_1nb_values)) # use -1 to denote none
        
        # L_2nb: Count values in 2nd order neighborhood
        second_order_neighbors_of_node = kth_order_neighborhood(network, node, 2)
        L_2nb_values = []
        for neighbor in neighbors_ordered_by_degree:
            zero_count = 0
            one_count = 0
            for i in set(network[neighbor]) & second_order_neighbors_of_node:
                if dataset.iloc[i]['L'] == 0:
                    zero_count += 1
                else:
                    one_count += 1
            L_2nb_values.append(zero_count)
            L_2nb_values.append(one_count)
        
        row['L_2nb'] = L_2nb_values + [-1, -1] * (max_neighbors - len(L_1nb_values))
        
        df = df.append(row, ignore_index=True)
    
    return df

def flatten_and_split_lists(df, columns):
    # Find the maximum length of lists in each column
    max_lengths = df.iloc[0][columns].apply(lambda x: len(x))

    # Fill the new columns with values from the original lists
    for col in columns:
        expanded_cols = pd.DataFrame(df[col].tolist(), columns=[f'{col}_{i}' for i in range(max_lengths[col])])
        expanded_cols = expanded_cols.reset_index(drop=True)  # Reset index to align with the original DataFrame
        df = pd.concat([df.reset_index(drop=True), expanded_cols], axis=1)


    # Drop the original columns with lists
    df = df.drop(columns=columns)

    return df


def create_new_predictors(predictors, X_columns):
    new_predictors = []

    for predictor in predictors:
        matching_cols = [col for col in X_columns if col.startswith(predictor)]
        new_predictors.extend(matching_cols)

    return new_predictors

def diff_test_accuracy(X, y, null_predictors, alt_predictors, test_size=0.3, random_state=0):
    '''
    Using random forest classifier with grid search for parameter tuning to 
    train a null model using null_predictors and an alternative model using 
    alt_predictors. 
    
    Return the test accuracy of alternative model minus that of the null model.
    '''
    # prepare training and testing set

    y = y.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Identify overlapping row indices
    overlapping_indices = set(X_train.index) & set(X_test.index)

    # Remove overlapping rows from the training set
    X_train = X_train.drop(index=overlapping_indices)
    y_train = y_train.drop(index=overlapping_indices)
   

    # Convert the type of Y
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    # Flatten and split lists in X_train and X_test
    list_columns = alt_predictors  # Add more columns if needed
    X_train = flatten_and_split_lists(X_train, list_columns)
    X_test = flatten_and_split_lists(X_test, list_columns)
    # print(X_train)

    param_grid = {
        'n_estimators': [1000]
    }

    # create new null predictors and alt predictors
    null_predictors = create_new_predictors(null_predictors, X_train.columns)
    alt_predictors = create_new_predictors(alt_predictors, X_train.columns)
    # print(null_predictors)
    # print(alt_predictors)
    # Define the parameter grid
    # param_grid = {
    #     'n_estimators': [1000],
    #     # 'max_depth': [None, 10, 20],
    #     # 'min_samples_split': [2, 5, 10]
    # }
    
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=2, n_jobs=-1)

    # train and test null model
    grid_search.fit(X_train[null_predictors], y_train)
    best_null_model = grid_search.best_estimator_
    y_pred = best_null_model.predict(X_test[null_predictors])
    accuracy_null = accuracy_score(y_test, y_pred)
    
    # train and test alternative model
    grid_search.fit(X_train[alt_predictors], y_train)
    best_alt_model = grid_search.best_estimator_
    y_pred = best_alt_model.predict(X_test[alt_predictors])
    accuracy_alt = accuracy_score(y_test, y_pred)
    
    print(accuracy_alt - accuracy_null, "=", accuracy_alt, "-", accuracy_null)
    return accuracy_alt - accuracy_null

def nonparametric_test(X, y, null_predictors, alt_predictors, bootstrap_iter=100, percentile_lower=2.5, percentile_upper=97.5):
    diff_test_accuracies = []
    combined = pd.concat([X, y], axis=1)
    
    for i in range(bootstrap_iter):
        if i % 10 == 0:
            print(i)
        # bootstrap (sample w replacement) a new dataset from the original X, y
        bootstrapped_combined = combined.sample(n=len(combined), replace=True, random_state=i)

        bootstrapped_X = bootstrapped_combined[X.columns]
        bootstrapped_y = bootstrapped_combined[y.columns]
        
        # compute the test statistic using the bootstrapped dataset
        diff_test_acc = diff_test_accuracy(bootstrapped_X, bootstrapped_y, 
                                           null_predictors=null_predictors, 
                                           alt_predictors=alt_predictors,
                                           random_state=i)
        diff_test_accuracies.append(diff_test_acc)
    
    return np.percentile(diff_test_accuracies, [percentile_lower, percentile_upper])

def test_edge_type(layer, dataset, bootstrap_iter):
    if layer == "L":
        null_predictors = ['L_1nb']
        alt_predictors = ['L_1nb', 'L_2nb']
    if layer == "A":
        null_predictors = ['L', 
                           'L_1nb_1_count', 'L_1nb_0_count', 
                           'L_2nb_1_count', 'L_2nb_0_count',
                           'A_1nb_1_count', 'A_1nb_0_count']
        alt_predictors = ['L', 
                          'L_1nb_1_count', 'L_1nb_0_count', 
                          'L_2nb_1_count', 'L_2nb_0_count',
                          'L_3nb_1_count', 'L_3nb_0_count',
                          'A_1nb_1_count', 'A_1nb_0_count',
                          'A_2nb_1_count', 'A_2nb_0_count']
    if layer == "Y":
        null_predictors = ['L', 'A',
                           'L_1nb_1_count', 'L_1nb_0_count', 
                           'L_2nb_1_count', 'L_2nb_0_count',
                           'A_1nb_1_count', 'A_1nb_0_count',
                           'A_2nb_1_count', 'A_2nb_0_count',
                           'Y_1nb_1_count', 'Y_1nb_0_count']
        alt_predictors = ['L', 'A',
                           'L_1nb_1_count', 'L_1nb_0_count', 
                           'L_2nb_1_count', 'L_2nb_0_count',
                           'L_3nb_1_count', 'L_3nb_0_count',
                           'A_1nb_1_count', 'A_1nb_0_count',
                           'A_2nb_1_count', 'A_2nb_0_count',
                           'A_3nb_1_count', 'A_3nb_0_count',
                           'Y_1nb_1_count', 'Y_1nb_0_count',
                           'Y_2nb_1_count', 'Y_2nb_0_count']

    y = pd.DataFrame(dataset[layer])
    X = dataset.drop([layer, 'id'], axis=1)
    print(X[alt_predictors], y, null_predictors, alt_predictors)
    lower, upper = nonparametric_test(X, y, 
                                      null_predictors=null_predictors, 
                                      alt_predictors=alt_predictors, 
                                      bootstrap_iter=bootstrap_iter)
    print("Lower: ", lower)
    print("Upper: ", upper)
    if lower <= 0 <= upper:
        print(layer, " - ", layer)
    else:
        print(layer, "<->", layer)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', None)
    # pd.set_option('display.max_colwidth', None)

    ''' STEP 1: Greate graph '''
    NUM_OF_VERTICES = 100000
    BURN_IN = 200
    BOOTSTRAP_ITER = 100
    VERBOSE = True
    MAX_NBS = 7

    network = create_random_network(n=NUM_OF_VERTICES, min_neighbors=1, max_neighbors=MAX_NBS)

    ''' STEP 2: Create data '''
    #edge_types = {'L' : ['U', {'prob_v_given_boundary':dg.prob_v_given_boundary_1, 'verbose':VERBOSE, 'burn_in':BURN_IN}]}
    #               'A' : ['U', {'prob_v_given_boundary':dg.prob_v_given_boundary_2, 'verbose':VERBOSE, 'burn_in':BURN_IN}], 
    #               'Y' : ['U', {'prob_v_given_boundary':dg.prob_v_given_boundary_3, 'verbose':VERBOSE, 'burn_in':BURN_IN}]}
    edge_types = {'L' : ['B', {'U_dist':dg.U_dist_1, 'f':dg.f_1, 'verbose':VERBOSE}]}
                #   'A' : ['U', {'prob_v_given_boundary':dg.prob_v_given_boundary_2, 'verbose':VERBOSE, 'burn_in':BURN_IN}], 
                #   'Y' : ['U', {'prob_v_given_boundary':dg.prob_v_given_boundary_3, 'verbose':VERBOSE, 'burn_in':BURN_IN}]}
    
    sample = dg.sample_L_A_Y(n_samples=1, network=network, edge_types=edge_types)[0]
    
    ''' STEP 3: Create and prepare data '''
    ind_set = maximal_n_apart_independent_set(graph=network, n=5, verbose=False)
    print("Size of ind set: ", len(ind_set))
    df = prepare_data(sample, ind_set, network, max_neighbors=MAX_NBS)
    print(df)
    ''' STEP 4: Perform nonparametric test '''
    test_edge_type(layer="L", dataset=df, bootstrap_iter=BOOTSTRAP_ITER)
    # test_edge_type(layer="A", dataset=df, bootstrap_iter=BOOTSTRAP_ITER)
    # test_edge_type(layer="Y", dataset=df, bootstrap_iter=BOOTSTRAP_ITER)


# # next step: create a wrapper function to test for all three layers. 
# # modify step 4. 