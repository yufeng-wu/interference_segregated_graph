from util import create_cycle_graph, graph_to_edges
from data_generator import sample_from_BG, sample_from_UG
from maximal_independent_set import maximal_n_apart_independent_set
from scipy.special import expit
from scipy.optimize import minimize
from scipy.stats import chi2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

def U_dist():
    '''
    Define the distribution: U ~ U_dist.
    '''
    return np.random.normal(-0.2, 1)

def f(U_values):
    '''
    Define the function f that calculates V based on its neighboring U values
    and returns a binary value.
    '''
    noise = np.random.normal(0, 1)
    linear_sum = 5*sum(U_values) + noise
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
                               'count_nb_value_is_1', 'count_nb_value_is_0', 'avg_nb_value', 
                               'count_nbnb_value_is_1', 'count_nbnb_value_is_0', 'avg_nbnb_value'
                              ])
    
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
                         'avg_nb_value':sum(nb_values)/len(nb_values),
                         'count_nbnb_value_is_1':sum(nbnb_values),
                         'count_nbnb_value_is_0':len(nbnb_values) - sum(nbnb_values),
                         'avg_nbnb_value':sum(nbnb_values)/len(nbnb_values)
                        })
        df = df.append(row, ignore_index=True)
    
    return df

def diff_test_accuracy(X, y, null_predictors, alt_predictors, test_size=0.3, random_state=42):
    '''
    Using random forest classifier to train a null model using null_predictors and 
    an alternative model using alt_predictors. Return the test accuracy of alternative
    model minus that of the null model.
    '''
    # prepare training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, np.ravel(y), test_size=test_size, random_state=random_state)
    
    # train and test null model
    model_null = RandomForestClassifier()
    model_null.fit(X_train[null_predictors], y_train)
    y_pred = model_null.predict(X_test[null_predictors])
    accuracy_null = accuracy_score(y_test, y_pred)
    
    # train and test alternative model
    model_alt = RandomForestClassifier()
    model_alt.fit(X_train[alt_predictors], y_train)
    y_pred = model_alt.predict(X_test[alt_predictors])
    accuracy_alt = accuracy_score(y_test, y_pred)
    
    return accuracy_alt - accuracy_null

def nonparametric_test(X, y, null_predictors, alt_predictors, bootstrap_iter=100, percentile_lower=2.5, percentile_upper=97.5):
    diff_test_accuracies = []
    
    for i in range(bootstrap_iter):
        if i % 10 == 0:
            print(i)
        # bootstrap (sample w replacement) a new dataset from the original X, y
        combined = pd.concat([X, y], axis=1)
        bootstrapped_combined = combined.sample(n=len(combined), replace=True, random_state=i)

        bootstrapped_X = bootstrapped_combined[X.columns]
        bootstrapped_y = bootstrapped_combined[y.columns]
        
        # compute the test statistic using the bootstrapped dataset
        diff_test_acc = diff_test_accuracy(bootstrapped_X, bootstrapped_y, null_predictors, alt_predictors)
        diff_test_accuracies.append(diff_test_acc)
    
    return np.percentile(diff_test_accuracies, [percentile_lower, percentile_upper])

if __name__ == "__main__":


    ''' STEP 1: Greate graph '''
    NUM_OF_VERTICES = 10000
    UG_BURN_IN_PERIOD = 300
    BOOTSTRAP_ITER = 200

    # TODO: next step -- create a random graph
    graph = create_cycle_graph(NUM_OF_VERTICES)


    ''' STEP 2: Create data '''
    sample = sample_from_BG(edges=graph_to_edges(graph), 
                               U_dist=U_dist, 
                               f=f)
    
    sample = sample_from_UG(graph, prob_v_given_neighbors, verbose=True, burn_in=UG_BURN_IN_PERIOD)

    ''' STEP 3: Create and prepare data '''
    ind_set = maximal_n_apart_independent_set(graph, 
                                          n=5, 
                                          available_vertices=set(graph.keys()),
                                          approx=True,
                                          is_cycle_graph=True)
    df = prepare_data(sample, ind_set, graph)
    y = pd.DataFrame(df['value'])
    X = df.drop(['value', 'id'], axis=1)


    ''' STEP 4: Perform nonparametric test '''
    null_predictors = ['count_nb_value_is_1', 'count_nb_value_is_0']
    alt_predictors = ['count_nb_value_is_1', 'count_nb_value_is_0', 
                      'count_nbnb_value_is_1', 'count_nbnb_value_is_0']
    print(nonparametric_test(X, y, 
                             null_predictors=null_predictors, 
                             alt_predictors=alt_predictors,
                             bootstrap_iter=BOOTSTRAP_ITER))