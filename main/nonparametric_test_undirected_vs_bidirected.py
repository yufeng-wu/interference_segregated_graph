from util import create_random_network
import data_generator
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

def prepare_data(dataset, ind_set, graph):
    '''
    Prepare data from dataset using ind_set into the forat below. 
    Each vertex in ind_set should take one row in the output dataframe.
    
    Output format:
    
    | vertex_id | value | count_nb_value_is_1 | count_nb_value_is_0 | count_nbnb_value_is_1 | count_nbnb_value_is_0 |
    |  (int)    | 1/0   | (int)               | (int)               | (int)                 | (int)                 |
    '''
    
    df = pd.DataFrame(columns=['id', 'value', 
                               'count_nb_value_is_1', 'count_nb_value_is_0',
                               'count_nbnb_value_is_1', 'count_nbnb_value_is_0'])
    
    for node in ind_set:
        # create a list to store the values of node's neighbors
        nb_values = [dataset[nb] for nb in graph[node]]
        
        # create a list to store the values of {node's neighbors' neighbors} \ node's neighbors
        node_nbnb = set()
        for nb in graph[node]:
            node_nbnb.update(graph[nb])
            
        # remove the current vertex and its neighbors
        node_nbnb.difference_update(graph[node])
        node_nbnb.discard(node)

        nbnb_values = [dataset[element] for element in node_nbnb]
        
        row = pd.Series({'id':int(node), 
                         'value':dataset[node], 
                         'count_nb_value_is_1':sum(nb_values),
                         'count_nb_value_is_0':len(nb_values) - sum(nb_values),
                         'count_nbnb_value_is_1':sum(nbnb_values),
                         'count_nbnb_value_is_0':len(nbnb_values) - sum(nbnb_values)})
        df = df.append(row, ignore_index=True)
    
    return df

def diff_test_accuracy(X, y, null_predictors, alt_predictors, test_size=0.3, random_state=0):
    '''
    Using random forest classifier with grid search for parameter tuning to 
    train a null model using null_predictors and an alternative model using 
    alt_predictors. 
    
    Return the test accuracy of alternative model minus that of the null model.
    '''
    # prepare training and testing set
    y = y.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, np.ravel(y), test_size=test_size, random_state=random_state)

    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 500],
        'max_depth': [None, 20],
        'min_samples_split': [2, 5]
    }

    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=3, n_jobs=-1)

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
    
    print(accuracy_alt - accuracy_null)
    return accuracy_alt - accuracy_null



# def diff_test_accuracy(X, y, null_predictors, alt_predictors, test_size=0.3, random_state=0):
#     '''
#     Using random forest classifier to train a null model using null_predictors and 
#     an alternative model using alt_predictors. Return the test accuracy of alternative
#     model minus that of the null model.
#     '''
#     # prepare training and testing set
#     y = y.astype(int)
#     X_train, X_test, y_train, y_test = train_test_split(X, np.ravel(y), test_size=test_size, random_state=random_state)

#     # train and test null model
#     model_null = RandomForestClassifier(n_estimators=1000, max_depth=100)
#     model_null.fit(X_train[null_predictors], y_train)
#     y_pred = model_null.predict(X_test[null_predictors])
#     accuracy_null = accuracy_score(y_test, y_pred)
    
#     # train and test alternative model
#     model_alt = RandomForestClassifier(n_estimators=1000, max_depth=100)
#     model_alt.fit(X_train[alt_predictors], y_train)
#     y_pred = model_alt.predict(X_test[alt_predictors])
#     accuracy_alt = accuracy_score(y_test, y_pred)
    
#     return accuracy_alt - accuracy_null

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
        diff_test_acc = diff_test_accuracy(bootstrapped_X, bootstrapped_y, null_predictors, alt_predictors)
        diff_test_accuracies.append(diff_test_acc)
    
    return np.percentile(diff_test_accuracies, [percentile_lower, percentile_upper])

if __name__ == "__main__":

    # sys.setrecursionlimit(4000)
    warnings.filterwarnings('ignore')

    ''' STEP 1: Greate graph '''
    NUM_OF_VERTICES = 800
    UG_BURN_IN_PERIOD = 100
    BOOTSTRAP_ITER = 200

    # graph = create_cycle_graph(NUM_OF_VERTICES)
    network = create_random_network(n=NUM_OF_VERTICES, min_neighbors=1, max_neighbors=8)

    ''' STEP 2: Create data '''
    sample = data_generator.sample_biedge_layer(network=network, 
                                                sample={}, 
                                                layer='L', 
                                                U_dist=data_generator.U_dist_1, 
                                                f=data_generator.f_1)

    ''' STEP 3: Create and prepare data '''
    ind_set = maximal_n_apart_independent_set(graph=network, n=5, verbose=False)
    print("Size of ind set: ", len(ind_set))

    df = prepare_data(sample, ind_set, network)
    print(df)
    y = pd.DataFrame(df['value'])
    X = df.drop(['value', 'id'], axis=1)
    
    ''' STEP 4: Perform nonparametric test '''
    null_predictors = ['count_nb_value_is_1', 'count_nb_value_is_0']
    alt_predictors = ['count_nb_value_is_1', 'count_nb_value_is_0', 
                      'count_nbnb_value_is_1', 'count_nbnb_value_is_0']
    lower, upper = nonparametric_test(X, y, 
                             null_predictors=null_predictors, 
                             alt_predictors=alt_predictors,
                             bootstrap_iter=BOOTSTRAP_ITER)
    print("Lower: ", lower)
    print("Upper: ", upper)
    if lower <= 0 <= upper:
        print("The data is generated from an undirected graph")
    else:
        print("The data is generated from a bidirected graph")