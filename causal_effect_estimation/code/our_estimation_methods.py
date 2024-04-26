import sys
sys.path.append("../../")
from infrastructure.maximal_independent_set import maximal_n_hop_independent_set
from infrastructure.data_generator import *
from infrastructure.network_utils import kth_order_neighborhood
from autog_estimation_methods import npll_L
import pandas as pd
import numpy as np
import random
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
from tqdm import tqdm
from pygam import LogisticGAM, s, l
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

def ricf(L1, L2, num_iter, var, max_degree_of_network):
    '''
    RICF stands for Residual Iterative Conditional Fitting, a method
    from the paper "Computing maximum likelihood estimates in recursive 
    linear models with correlated errors" by Drton, Eichler, and Richardson.
    RICF is used to estimate the covariance matrix of a graphical model 
    that specifies a multivariate normal distribution.
    
    This function estimates the covariance matrix of the joint distribution
    L1 <-> L2, which is assumed to be generated from a multivariate normal
    distribution.
    
    Args:
        L1: a list of n independent realizations of L1.
        L2: a list of n independent realizations of L2.
        max_iter: number of iterations to run the optimization.
        var: the estimated, shared variance of L1 and L2. var(L1) = var(L2)
            is by assumption.
        max_degree_of_network: largest degree among all vertices in the network.
            We need this information to ensure diagonal dominance throughout 
            the optimization process, which is a sufficient condition for the 
            positive definiteness of the estimated covariance matrix.
    
    Returns:
        A 2x2 numpy array representing the estimated covariance matrix of the
        joint distribution L1 <-> L2.
    '''

    def least_squares_loss(params, L, Z, var_index):
        n, _ = L.shape
        return 0.5 / n * np.linalg.norm(L[:, var_index] - np.dot(Z, params)) ** 2
        
    d = 2 # number of variables in the graphical model for RICF estimates
    eps_L1 = L1 - np.mean(L1)
    eps_L2 = L2 - np.mean(L2)

    L_df = pd.DataFrame({'L1': eps_L1, 'L2': eps_L2})

    # random guess for cov mat
    cov_mat = np.array([[0.0, 0.0],
                        [0.0, 0.0]])

    var_mat = np.array([[var, 0.0],
                        [0.0, var]])

    for _ in range(num_iter):
        for var_index in [0, 1]:
            omega = cov_mat + var_mat
            omega_minusi = np.delete(omega, var_index, axis=0)
            omega_minusii = np.delete(omega_minusi, var_index, axis=1)
            omega_minusii_inv = np.linalg.inv(omega_minusii)

            epsilon = L_df.values
            epsilon_minusi = np.delete(epsilon, var_index, axis=1)

            Z_minusi = epsilon_minusi @ omega_minusii_inv.T
            Z = np.insert(Z_minusi, var_index, 0, axis=1)
            
            # bounds are to ensure positive definiteness, and we also add/minus 
            # a small constant in case the rounding goes the wrong way
            bound = (-var/float(max_degree_of_network) + 1e-10, 
                      var/float(max_degree_of_network) - 1e-10)
            
            # getting the solution from five random initializations
            # and pick the one with the smallest loss
            best_solution = None
            best_loss = np.inf
            for _ in range(5):
                # minimize by first setting a random start within the bounds
                sol = minimize(least_squares_loss, 
                               x0=np.random.uniform(low=bound[0], high=bound[1], size=d),
                               args=(L_df.values, Z, var_index),
                               method='L-BFGS-B',
                               bounds=[bound]*d)
                if sol.fun < best_loss:
                    best_loss = sol.fun
                    best_solution = sol
            
            # update covariance matrix according to the best solution
            cov_mat[:, var_index] = cov_mat[var_index, :] = best_solution.x
            
            # this is a trivial update for graphs with only bidirected edges
            var_mat[var_index, var_index] = var 

    return cov_mat

def estimate_biedge_L_params(network_dict, L, max_degree_of_network):

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
    ind_set = maximal_n_hop_independent_set(network_dict, n=1)
    data = build_dataset(ind_set, L)
    
    est_var = np.var(data["L_i"]) # close-form MLE estimate

    # de-mean the data for easier estimate of covariance
    est_mean = np.mean(data["L_i"]) 
    data["L_i"] -= est_mean 
    
    # Estimate covariance using RICF
    # find structures like "Li <-> Lj", but it could be other structures
    # such as chain of length three.
    edge_graph_dict = create_edge_graph(network_dict)
    ind_set_2_hop_edge_graph = maximal_n_hop_independent_set(edge_graph_dict, n=2)

    L1 = []
    L2 = []

    for edge in ind_set_2_hop_edge_graph:
        v1, v2 = edge
 
        # randomly append v1, v2 into L1, L2
        if random.choice([True, False]):
            L1.append(L[v1])
            L2.append(L[v2])
        else:
            L1.append(L[v2])
            L2.append(L[v1])
    
    est_cov_mat = ricf(L1, L2, num_iter=20, var=est_var, max_degree_of_network=max_degree_of_network)
    est_cov = est_cov_mat[0][1] # get the covariance between Li and Lj
    # est_cov = np.cov(L1, L2)[0][1]
    return est_cov, est_var, est_mean

def causal_effects_B_U(network_adj_mat, params_L, params_Y, burn_in, 
                            n_simulations):
    '''
    can evaluate both true and estimated causal effects
    '''
    Ls = biedge_sample_Ls(network_adj_mat, params_L, n_draws=n_simulations)

    As_1 = np.ones(Ls.shape)
    As_0 = np.zeros(Ls.shape)

    Ys_A1 = gibbs_sample_Ys(network_adj_mat, Ls, As_1, params_Y, burn_in=burn_in)
    Ys_A0 = gibbs_sample_Ys(network_adj_mat, Ls, As_0, params_Y, burn_in=burn_in)
    return np.mean(Ys_A1 - Ys_A0)

def true_causal_effects_B_B(network_adj_mat, params_L, params_Y,
                            n_simulations):
    ''' vectorized '''
    # dimension of Ls is n_simulations x n_units
    Ls = biedge_sample_Ls(network_adj_mat, params_L, n_draws=n_simulations)

    As_1 = np.ones(Ls.shape)
    As_0 = np.zeros(Ls.shape)

    # dimension of Ys_A1 and Ys_A0 are both n_simulations x n_units
    Ys_A1 = biedge_sample_Ys(network_adj_mat, Ls, As_1, params_Y)
    Ys_A0 = biedge_sample_Ys(network_adj_mat, Ls, As_0, params_Y)
    
    return np.mean(Ys_A1 - Ys_A0)

def estimate_causal_effects_B_B(network_dict, network_adj_mat, L, A, Y, 
                                max_degree_of_network, n_simulations):
    # 1) get iid realizations of p(L)
    L_est = estimate_biedge_L_params(network_dict, L, max_degree_of_network)

    Ls = biedge_sample_Ls(network_adj_mat, L_est, n_draws=n_simulations)
    
    # 2) build a ML model to estimate E[Y_i | A_i, A_Ni, L_i, L_Ni]
    model = build_EYi_model(L, A, Y, network_adj_mat, network_dict)
    
    # compare raw accuracy vs. model accuracy
    # majority_class = np.argmax(np.bincount(Y))
    # naive_accuracy = np.mean(Y == majority_class)
    
    # df = biedge_Y_df_builder(network_dict, network_dict.keys(), L, A, Y)
    # model_accuracy = np.mean((model.predict_proba(df)[:, 1] >0.5) == Y)
    # print(f"Naive Accuracy: {naive_accuracy:.3f}", f"Model Accuracy: {model_accuracy:.3f}")

    # 3) estimate network causal effects using empirical estimate of p(L)
    #    and model
    contrasts = estimate_causal_effect_biedge_Y_helper(network_dict, model, Ls)
    return np.mean(contrasts)

def true_causal_effects_U_B(network_adj_mat, params_L, params_Y, burn_in, 
                       n_simulations, gibbs_select_every):
    # dimension of Ls is n_simulations x n_units
    Ls = gibbs_sample_Ls(network_adj_mat, params_L, burn_in, 
                        n_draws=n_simulations, select_every=gibbs_select_every)
    
    As_1 = np.ones(Ls.shape)
    As_0 = np.zeros(Ls.shape)
    
    # dimension of Ys_A1 and Ys_A0 are both n_simulations x n_units
    Ys_A1 = biedge_sample_Ys(network_adj_mat, Ls, As_1, params_Y)
    Ys_A0 = biedge_sample_Ys(network_adj_mat, Ls, As_0, params_Y)
    
    return np.mean(Ys_A1 - Ys_A0)

def estimate_causal_effects_U_B(network_dict, network_adj_mat, L, A, Y, burn_in,
                               n_simulations, gibbs_select_every):
    ''' 
    Inputs:
        - gibbs_select_every: select every gibbs_select_every-th element of 
        the Gibbs samples to reduce auto-correlation.
    '''

    # 1) estimate the parameters to resample L layer
    params_L = minimize(npll_L, x0=np.random.uniform(-1, 1, 2), 
                        args=(L, network_adj_mat)).x
   
    # 2) build a ML model to estimate E[Y_i | A_i, A_Ni, L_i, L_Ni]
    model = build_EYi_model(L, A, Y, network_adj_mat, network_dict)
    
    # 3) use params_L and model to estimate causal effects:
    #   - first, get independent realizations of p(L) using Gibbs sampling
    #     and thin auto-correlation 
    Ls = gibbs_sample_Ls(network_adj_mat=network_adj_mat, params=params_L, 
                        burn_in=burn_in, n_draws=n_simulations,
                        select_every=gibbs_select_every)

    # a list of lists
    # each inner list is the estimated individual-level constrast between
    # pred_Y_i_given_intervention_1 - pred_Y_i_given_intervention_0
    contrasts = estimate_causal_effect_biedge_Y_helper(network_dict, model, Ls)
    print("CE: ", np.mean(contrasts))
    return np.mean(contrasts)

def estimate_causal_effect_biedge_Y_helper(network_dict, model, L_draws):
    contrasts = []
    
    with tqdm(total=len(L_draws), desc="Processing L_draws", ncols=70) as pbar:
        for L_draw in L_draws:
            l_j_sums = {i: np.sum([L_draw[nb] for nb in network_dict[i]]) 
                        for i in network_dict}
            
            # order of the features: a_i  l_i  l_j_sum  a_j_sum
            feature_vals_1 = np.array([
                [1, L_draw[i], l_j_sums[i], 1 * len(network_dict[i]), len(network_dict[i])]
                for i in network_dict
            ])
            feature_vals_0 = np.array([
                [0, L_draw[i], l_j_sums[i], 0 * len(network_dict[i]), len(network_dict[i])]
                for i in network_dict
            ])
            
            # convert to DataFrames with named columns
            feature_vals_1_df = pd.DataFrame(feature_vals_1, columns=['a_i', 'l_i', 'l_j_sum', 'a_j_sum', 'nb_count'])
            feature_vals_0_df = pd.DataFrame(feature_vals_0, columns=['a_i', 'l_i', 'l_j_sum', 'a_j_sum', 'nb_count'])
            
            # drop nb
            feature_vals_1_df = feature_vals_1_df.drop('nb_count', axis=1)
            feature_vals_0_df = feature_vals_0_df.drop('nb_count', axis=1)
            
            # the two variables below are vectors with n rows
            pred_Y_intervene_A1 = model.predict_proba(feature_vals_1_df)[:, 1]
            pred_Y_intervene_A0 = model.predict_proba(feature_vals_0_df)[:, 1]
            
            contrasts.append(pred_Y_intervene_A1 - pred_Y_intervene_A0)
            pbar.update()
            
    return contrasts

class CustomLogisticRegression:
    def __init__(self, df):
        self.df = df
        # Calculate class weights
        self.class_weights = self.calculate_class_weights(df['y_i'])
        self.params = self.train()

    def calculate_class_weights(self, y):
        weight_for_0 = len(y) / (2.0 * np.sum(y == 0))
        weight_for_1 = len(y) / (2.0 * np.sum(y == 1))
        return {0: weight_for_0, 1: weight_for_1}

    def train(self):
        # use the custom estimator to estimate the parameters for our 
        # logistic regression model. params_logistic_reg is of size 5.
        params_logistic_reg = minimize(self._nll_logistic_regression, 
                                       x0=np.random.uniform(-1, 1, 5)).x
        return params_logistic_reg

    def _nll_logistic_regression(self, params):
        pY1 = expit(params[0] + 
                    params[1]*self.df['l_i'] + 
                    params[2]*self.df['a_i'] + 
                    params[3]*self.df['l_j_sum'] + 
                    params[4]*self.df['a_j_sum'])
                    #  params[5]*self.df['nb_count']))
        pY1 = np.clip(pY1, 1e-10, 1 - 1e-10)
        log_likelihood = self.df['y_i']*np.log(pY1) + (1-self.df['y_i'])*np.log(1-pY1)
        
        # Apply class weights
        # weights = self.df['y_i'].replace(self.class_weights)
        # weighted_log_likelihood = weights * log_likelihood
        
        # Return the negative sum of the weighted log-likelihood
        # return -np.sum(weighted_log_likelihood)
        return -np.sum(log_likelihood)
    
        # pY = self.df['y_i']*pY1 + (1-self.df['y_i'])*(1-pY1)
        # # the expit() function outputs 0.0 when the input is reasonably small, 
        # # so we replace 0 with a small const to ensure numerical stability
        # pY = np.where(pY == 0, 1e-10, pY)
        # return -np.sum(np.log(pY))

    # def _npll_logistic_regression(self, params):
    #     pY1 = expit((params[0] + 
    #                  params[1]*self.L + 
    #                  params[2]*self.A + 
    #                  params[3]*(self.L@self.network_adj_mat) + 
    #                  params[4]*(self.A@self.network_adj_mat) +
    #                  params[5]*np.sum(self.network_adj_mat, axis=1)))
    #     pY = self.Y*pY1 + (1-self.Y)*(1-pY1)
    #     # the expit() function outputs 0.0 when the input is reasonably small, 
    #     # so we replace 0 with a small const to ensure numerical stability
    #     pY = np.where(pY == 0, 1e-10, pY)
    #     return -np.sum(np.log(pY))

    def predict_proba(self, X):
        # X is a pd.DataFrame with certain columns with dimension n x 4
        # p1 is a matrix of size n x 1
        p1 = expit(self.params[0] + 
                   self.params[1]*X['l_i'] + 
                   self.params[2]*X['a_i'] + 
                   self.params[3]*X['l_j_sum'] + 
                   self.params[4]*X['a_j_sum'])
                #    self.params[5]*X['nb_count'])
        # return in the same style as that of a sklearn model
        return np.column_stack((1-p1, p1))

def biedge_Y_df_builder(network, ind_set, L, A, Y):
    '''
    Creates dataframe for causal effect estimation. 
    
    Inputs:
        - network
        - ind_set: a maximal 1-apart independent set obtained from the network
        - sample: a single realization (L, A, Y) of the network where L, A, Y 
                  are vectors of the shape (1, size of network).
    
    Return:
        A pd.DataFrame object that with the following entries for each element 
        of the ind_set:
            'i': id of the subject
            'y_i': the value of Y_i in the network realization
            'a_i': the value of A_i in the network realization
            'l_i': the value of L_i in the network realization
            'l_j_sum': sum of [L_j for j in neighbors of i]
            'a_j_sum': sum of [A_j for j in neighbors of i]
    '''
    data_list = []

    for i in ind_set:
        l_i = L[i]
        a_i = A[i]
        y_i = Y[i]

        # get the neighbors of i as a list
        N_i = kth_order_neighborhood(network, i, 1)

        data_list.append({
            'i' : i,
            'y_i': y_i,
            'a_i': a_i,
            'l_i': l_i,
            'l_j_sum': np.sum([L[j] for j in N_i]),
            'a_j_sum': np.sum([A[j] for j in N_i]),
            # 'nb_count': len(N_i)
        })

    df = pd.DataFrame(data_list) 
    return df   

def build_EYi_model(L, A, Y, network_adj_mat, network_dict):
    
    ind_set_1_hop = maximal_n_hop_independent_set(network_dict, n=1)
    df = biedge_Y_df_builder(network_dict, ind_set_1_hop, L, A, Y)
    return CustomLogisticRegression(df)
    
    # majority_class = np.argmax(np.bincount(Y))
    # naive_accuracy = np.mean(Y == majority_class)
    
    # pY = expit((model.params[0] + model.params[1]*L + model.params[2]*A + 
    #                  model.params[3]*(L@network_adj_mat) + 
    #                  model.params[4]*(A@network_adj_mat)))
    # # now compare the predictoin with Y
    
    # Y_pred = (pY >= 0.5).astype(int)
    # model_accuracy = np.mean(Y_pred == Y)
    #print(f"Naive Accuracy: {naive_accuracy:.3f}", f"Model Accuracy: {model_accuracy:.3f}")

    # OLD IMPLEMENTATION USING ML
    # ind_set_1_hop = maximal_n_hop_independent_set(network_dict, n=1)
    # df = biedge_Y_df_builder(network_dict, ind_set_1_hop, L, A, Y)

    # target = df['y_i']
    # features = df.drop(['i', 'y_i'], axis=1)

    # model = LogisticRegression(class_weight='balanced')
    
    # model.fit(features, target)
    # return model
