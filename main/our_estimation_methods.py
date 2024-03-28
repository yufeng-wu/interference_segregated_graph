from maximal_independent_set import maximal_n_apart_independent_set
from autog import *

import pandas as pd
import numpy as np

from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

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
    params_L = minimize(npll_L, x0=np.random.uniform(-1, 1, 2), 
                        args=(L, network_adj_mat)).x

    # 2) build a ML model to estimate E[Y_i | A_i, A_Ni, L_i, L_Ni]
    ind_set_1_hop = maximal_n_apart_independent_set(network_dict, n=1)
    df = assemble_estimation_df(network_dict, ind_set_1_hop, L, A, Y)

    features = df.drop(['i', 'y_i'], axis=1) 
    target = df['y_i']

    model = LogisticRegression(max_iter=1000)
    model.fit(features, target)

    # 3) use params_L and model to estimate causal effects:
    #   - first, get independent realizations of p(L) using Gibbs sampling
    #     and thin auto-correlation 
    L_draws = gibbs_sample_L(network_adj_mat=network_adj_mat, params=params_L, 
                             burn_in=burn_in, n_draws=n_draws_from_pL,
                             select_every=gibbs_select_every)

    # a list of lists
    # each inner list is the estimated individual-level constrast between
    # pred_Y_i_given_intervention_1 - pred_Y_i_given_intervention_0
    contrasts = []
    
    for L_draw in L_draws:
        l_j_sums = {i: np.sum([L_draw[nb] for nb in network_dict[i]]) 
                    for i in network_dict}
        
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
        pred_Y_intervene_A1 = model.predict(feature_vals_1_df)
        pred_Y_intervene_A0 = model.predict(feature_vals_0_df)
        
        contrasts.append(pred_Y_intervene_A1 - pred_Y_intervene_A0)

    return np.mean(contrasts)