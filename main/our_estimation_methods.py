import random

from sklearn.ensemble import RandomForestClassifier
from maximal_independent_set import maximal_n_apart_independent_set
from autog import *

import pandas as pd
import numpy as np

from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

def true_causal_effects_B_B(network_adj_mat, params_L, params_Y,
                            n_simulations=100):
    L_samples = [biedge_sample_L(network_adj_mat, params_L, const_var=True) for 
                 _ in range(n_simulations)]
    
    n_unit = len(network_adj_mat)
    A1 = np.array([1] * n_unit)
    A0 = np.array([0] * n_unit)
    
    contrasts = [biedge_sample_Y(network_adj_mat, L, A1, params_Y) - 
                 biedge_sample_Y(network_adj_mat, L, A0, params_Y)
                 for L in L_samples]
    
    return np.mean(contrasts)

def estimate_causal_effects_B_B(network_dict, network_adj_mat, L, A, Y, 
                                n_simulations=100):
    # 1) get iid realizations of p(L)
    L_est = estimate_biedge_L_params(network_dict, L, A, Y)
    L_draws = [biedge_sample_L(network_adj_mat, L_est, const_var=True) for 
                 _ in range(n_simulations)]
    
    # 2) build a ML model to estimate E[Y_i | A_i, A_Ni, L_i, L_Ni]
    model = build_EYi_model(network_dict, L, A, Y)
    
    # 3) estimate network causal effects using empirical estimate of p(L)
    #    and model
    contrasts = estimate_causal_effect_biedge_Y_helper(network_dict, model, L_draws)
    return np.mean(contrasts)

def ricf(L1, L2, max_iter, var):

    def least_squares_loss(params, L, Z, var_index):
        n, _ = L.shape
        return 0.5 * n * np.linalg.norm(L[:, var_index] - np.dot(Z, params)) ** 2
        # return 0.5 / n * np.linalg.norm(L[:, var_index] - np.dot(Z, params)) ** 2

    d = 2 # number of variables in the graphical model for RICF estimates
    eps_L1 = L1 - np.mean(L1)
    eps_L2 = L2 - np.mean(L2)

    L_df = pd.DataFrame({'L1': eps_L1, 'L2': eps_L2})

    # random guess for cov mat
    cov_mat = np.array([[0.0, 0.0],
                        [0.0, 0.0]])

    var_mat = np.array([[var, 0.0],
                        [0.0, var]])

    for _ in range(max_iter):

        for var_index in [0, 1]:
            omega = cov_mat + var_mat
            omega_minusi = np.delete(omega, var_index, axis=0)
            omega_minusii = np.delete(omega_minusi, var_index, axis=1)
            omega_minusii_inv = np.linalg.inv(omega_minusii)

            epsilon = L_df.values
            epsilon_minusi = np.delete(epsilon, var_index, axis=1)

            Z_minusi = epsilon_minusi @ omega_minusii_inv.T
            Z = np.insert(Z_minusi, var_index, 0, axis=1)

            sol = minimize(least_squares_loss,
                            np.zeros(d),
                            args=(L_df.values, Z, var_index))

            # update covariance matrix according to the solution
            cov_mat[:, var_index] = sol.x
            cov_mat[var_index, :] = sol.x

            # this is a trivial update for graphs with only bidirected edges
            var_mat[var_index, var_index] = var 

    return cov_mat

def estimate_biedge_L_params(network_dict, L, A, Y):
    
    def build_dataset(ind_set, L):
        df = pd.DataFrame(L.T, columns=["L_i"])
        return df.loc[list(ind_set)]
    
    def create_edge_graph(graph_dict):
        '''
        An edge graph L(G) of a graph G is a graph such that each vertex of L(G)
        represents an edge of G, and two vertices of L(G) are adjacent if and only
        if their corresponding edges in G share a common vertex.
        '''
        G = nx.Graph(graph_dict)
        LG = nx.line_graph(G)
        return nx.to_dict_of_lists(LG)
    
    # find a 1-hop independent set to estimate mean of L_i
    ind_set = maximal_n_apart_independent_set(network_dict, n=1)
    data = build_dataset(ind_set, L)

    # de-mean the data for easier estimate of covariance
    est_mean = np.mean(data["L_i"]) 
    data["L_i"] -= est_mean 
    
    # Estimate covariance using RICF
    # find structures like "Li <-> Lj", but it could be other structures
    # such as chain of length three.
    edge_graph_dict = create_edge_graph(network_dict)
    ind_set_2_hop_edge_graph = maximal_n_apart_independent_set(edge_graph_dict, n=2)

    L1 = []
    L2 = []

    for edge in ind_set_2_hop_edge_graph:
        v1, v2 = edge
        # randomly append v1, v2 into L1, L2
        if random.random() < 0.5:
            L1.append(v1)
            L2.append(v2)
        else:
            L1.append(v2)
            L2.append(v1)
    
    est_var = np.var(data["L_i"]) # close-form MLE estimate
    est_cov_mat = ricf(L1, L2, max_iter=30, var=est_var)
    est_cov = est_cov_mat[0][1] # get the covariance between Li and Lj

    return est_cov, est_var, est_mean

def causal_effects_B_U(network_adj_mat, params_L, params_Y, burn_in=200, 
                            n_simulations=100):
    contrasts = []
    n_unit = len(network_adj_mat)
    
    for i in range(n_simulations):
        L = biedge_sample_L(network_adj_mat, params_L, const_var=True)
        
        A1 = np.array([1] * n_unit)
        A0 = np.array([0] * n_unit)
        
        Y_A1 = gibbs_sample_Y(network_adj_mat, L, A1, params_Y, burn_in=burn_in)
        Y_A0 = gibbs_sample_Y(network_adj_mat, L, A0, params_Y, burn_in=burn_in)
        
        contrasts.append(np.mean(Y_A1 - Y_A0))
    
    return np.mean(contrasts)
    

def true_causal_effects_U_B(network_adj_mat, params_L, params_Y, burn_in=200, 
                       n_simulations=100):
    ''' L layer is Undirected (---), Y layer is Bidirected (<->) '''
    L_samples = gibbs_sample_L(network_adj_mat, params_L, burn_in, n_draws=n_simulations, select_every=3)
    
    n_unit = len(network_adj_mat)
    A1 = np.array([1] * n_unit)
    A0 = np.array([0] * n_unit)
    
    contrasts = [biedge_sample_Y(network_adj_mat, L, A1, params_Y) - 
                 biedge_sample_Y(network_adj_mat, L, A0, params_Y)
                 for L in L_samples]
    
    return np.mean(contrasts)

def estimate_causal_effects_U_B(network_dict, network_adj_mat, L, A, Y, 
                               n_draws_from_pL, gibbs_select_every, burn_in):
    ''' 
    L layer is Undirected (---), Y layer is Bidirected (<->) 
    
    Inputs:
        - network_dict
        - network_adj_mat
        
        - gibbs_select_every: select every gibbs_select_every-th element of 
        the Gibbs samples to reduce auto-correlation.
    '''

    # 1) estimate the parameters to resample L layer
    # params_L = minimize(npll_L, x0=np.random.uniform(-1, 1, 2), 
    #                     args=(L, network_adj_mat)).x
    params_L = np.array([-0.3, 0.4]) # give it true params_L
    print("params L:", params_L)
    # 2) build a ML model to estimate E[Y_i | A_i, A_Ni, L_i, L_Ni]
    model = build_EYi_model(network_dict, L, A, Y)

    # 3) use params_L and model to estimate causal effects:
    #   - first, get independent realizations of p(L) using Gibbs sampling
    #     and thin auto-correlation 
    L_draws = gibbs_sample_L(network_adj_mat=network_adj_mat, params=params_L, 
                             burn_in=burn_in, n_draws=n_draws_from_pL,
                             select_every=gibbs_select_every)

    # a list of lists
    # each inner list is the estimated individual-level constrast between
    # pred_Y_i_given_intervention_1 - pred_Y_i_given_intervention_0
    contrasts = estimate_causal_effect_biedge_Y_helper(network_dict, model, L_draws)
    
    return np.mean(contrasts)

def estimate_causal_effect_biedge_Y_helper(network_dict, model, L_draws):
    contrasts = []
    
    for L_draw in L_draws:
        l_j_sums = {i: np.sum([L_draw[nb] for nb in network_dict[i]]) 
                    for i in network_dict}
        
        # order of the features: a_i  l_i  l_j_sum  a_j_sum
        feature_vals_1 = np.array([
            [1, L_draw[i], l_j_sums[i], 1 * len(network_dict[i])]
            for i in network_dict
        ])
        feature_vals_0 = np.array([
            [0, L_draw[i], l_j_sums[i], 0 * len(network_dict[i])]
            for i in network_dict
        ])
        
        # convert to DataFrames with named columns
        feature_vals_1_df = pd.DataFrame(feature_vals_1, columns=['a_i', 'l_i', 'l_j_sum', 'a_j_sum'])
        feature_vals_0_df = pd.DataFrame(feature_vals_0, columns=['a_i', 'l_i', 'l_j_sum', 'a_j_sum'])
        
        # the two variables below are vectors with n rows
        pred_Y_intervene_A1 = model.predict_proba(feature_vals_1_df)[:, 1]
        pred_Y_intervene_A0 = model.predict_proba(feature_vals_0_df)[:, 1]
        
        contrasts.append(pred_Y_intervene_A1 - pred_Y_intervene_A0)
    return contrasts

def build_EYi_model(network_dict, L, A, Y):
    ind_set_1_hop = maximal_n_apart_independent_set(network_dict, n=1)
    df = assemble_estimation_df(network_dict, ind_set_1_hop, L, A, Y)

    target = df['y_i']
    features = df.drop(['i', 'y_i'], axis=1) 
    
    model = RandomForestClassifier(n_estimators=100) # smaller C means more regularization #RandomForestClassifier(n_estimators=50) # LogisticRegression()
    model.fit(features, target)
    
    # evaluate model
    predicted = model.predict(features)
    accuracy = accuracy_score(target, predicted)

    print("naive:", max(np.mean(target), 1-np.mean(target)), "Model Accuracy:", accuracy)

    return model