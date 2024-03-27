from util import create_random_network, kth_order_neighborhood
from maximal_independent_set import maximal_n_apart_independent_set

from concurrent.futures import ProcessPoolExecutor
import networkx as nx

import pandas as pd
import numpy as np

from scipy.optimize import minimize
from scipy.special import expit

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def ring_adjacency_matrix(num_units):

    network = np.zeros((num_units, num_units))
    for i in range(num_units-1):
        network[i, i+1] = 1
        network[i+1, i] = 1

    network[0, num_units-1] = 1
    network[num_units-1, 0] = 1

    return network

def biedge_sample_L(network, params):
    
    U = np.random.normal(loc=params[0], scale=params[1], size=network.shape)
    U = np.triu(U) + np.triu(U, 1).T # make U symmetric by copying the upper triangular to the lower triangular part
    U = np.where(network == 1, U, network) # apply the network mask

    pL = expit(params[2] + params[3]*U.sum(axis=0)) # pL is a vector 
    L = np.random.binomial(1, pL)

    return L

def biedge_sample_A(network, L, params):

    U = np.random.normal(loc=params[0], scale=params[1], size=network.shape)
    U = np.triu(U) + np.triu(U, 1).T  # make U symmetric
    U = np.where(network == 1, U, network)  # apply network mask

    pA = expit(params[2] + params[3]*L + params[4]*(L@network) + params[5]*U.sum(axis=0))
    A = np.random.binomial(1, pA)

    return A

def biedge_sample_Y(network, L, A, params):

    U = np.random.normal(loc=params[0], scale=params[1], size=network.shape)
    U = np.triu(U) + np.triu(U, 1).T  # make U symmetric
    U = np.where(network == 1, U, network)  # apply network mask

    pY = expit(params[2] + params[3]*L + params[4]*A + params[5]*(L@network) + 
               params[6]*(A@network) + params[7]*U.sum(axis=0))
    
    Y = np.random.binomial(1, pY)

    return Y

def gibbs_sample_L(network_adj_mat, params, burn_in=200, n_draws=1, select_every=1):
    # TODO: this can be changed to a more general version: the user specify 
    # how much to thin autocorrealtion, and this funciton can return a list 
    # of "indepdnet" gibbs sample Ls. The user will also specify how many samples
    # they want.

    L_samples = []
    # initialize a vector of Ls
    L = np.random.binomial(1, 0.5, len(network_adj_mat))

    # keep sampling an L vector till burn in is done
    for gibbs_iter in range(burn_in + n_draws*select_every):
        for i in range(len(network_adj_mat)):
            pLi_given_rest = expit(params[0] + params[1]*np.dot(L, network_adj_mat[i, :]))
            L[i] = np.random.binomial(1, pLi_given_rest)

        if gibbs_iter >= burn_in and gibbs_iter % select_every == 0:
            L_samples.append(L.copy())
    
    return L_samples

def gibbs_sample_A(network, L, params, burn_in=200):

    A = np.random.binomial(1, 0.5, len(network))

    # keep sampling an A vector till burn in is done
    for m in range(burn_in):
        for i in range(len(network)):
            pAi_given_rest = expit(params[0] + params[1]*L[i] +
                                   params[2]*np.dot(A, network[i, :]) +
                                   params[3]*np.dot(L, network[i, :]))
            A[i] = np.random.binomial(1, pAi_given_rest)

    return A

def gibbs_sample_Y(network, L, A, params, burn_in=200):

    Y = np.random.binomial(1, 0.5, len(network))

    # keep sampling an Y vector till burn in is done
    for m in range(burn_in):
        for i in range(len(network)):
            pYi_given_rest = expit(params[0] + params[1]*L[i] + params[2]*A[i] +
                                   params[3]*np.dot(L, network[i, :]) +
                                   params[4]*np.dot(A, network[i, :]) +
                                   params[5]*np.dot(Y, network[i, :]))
            Y[i] = np.random.binomial(1, pYi_given_rest)

    return Y

def npll_L(params, L, network_adj_mat):

    pL1 = expit(params[0] + params[1]*(L@network_adj_mat)) # a length n vector.
    pL = L*pL1 + (1-L)*(1-pL1)
    # TODO: check why that's happening
    pL = np.where(pL == 0, 1e-10, pL) # replace 0 with a small const to ensure numerical stability
    
    # bootstrap pL here
    
    return -np.sum(np.log(pL))

def npll_Y(params, L, A, Y, network):

    pY1 = expit((params[0] + params[1]*L + params[2]*A + params[3]*(L@network) + params[4]*(A@network) + params[5]*(Y@network)))
    pY = Y*pY1 + (1-Y)*(1-pY1)
    pY = np.where(pY == 0, 1e-10, pY)
    return -np.sum(np.log(pY))

def estimate_causal_effects_U_U(network_adj_mat, A_value, params_L, params_Y, 
                                burn_in=200, K=100, N=3):
    '''
    K: number of rows in matrix_Ys
    N: thin the Markov Chain for every N iteration
    '''
    # initialize random values
    Y = np.random.binomial(1, 0.5, len(network_adj_mat))
    L = np.random.binomial(1, 0.5, len(network_adj_mat))

    A = [A_value] * len(L)  # a list of A_value repeated len(L) times
    matrix_Ys = []

    for m in range(burn_in + K*N):
        for i in range(len(network_adj_mat)):
            pLi_given_rest = expit(params_L[0] + params_L[1]*np.dot(L, network_adj_mat[i, :]))
            L[i] = np.random.binomial(1, pLi_given_rest)

            pYi_given_rest = expit(params_Y[0] + params_Y[1]*L[i] + params_Y[2]*A[i] +
                                   params_Y[3]*np.dot(L, network_adj_mat[i, :]) +
                                   params_Y[4]*np.dot(A, network_adj_mat[i, :]) +
                                   params_Y[5]*np.dot(Y, network_adj_mat[i, :]))
            Y[i] = np.random.binomial(1, pYi_given_rest)

        if m > burn_in and m % N == 0:
            matrix_Ys.append(Y.copy())

    return np.mean(matrix_Ys)

def estimate_causal_effects_B_B(network, A_value, params_L, params_Y, K=100):
    matrix_Ys = []

    for _ in range(K):
        L = biedge_sample_L(network, params_L)
        A = np.array([A_value] * len(L))
        Y = biedge_sample_Y(network, L, A, params_Y)

        matrix_Ys.append(Y.copy())
    
    return np.mean(matrix_Ys)


# def bootstrap_causal_effects_U_B(n_units_list, n_bootstraps, true_L, true_A, 
#                                  true_Y, burn_in):
#     '''
#     Bootstrap a confidence interval of causal effects computed using 
#     causal_effects_U_B
#     '''
#     estimates = {}
#     with ProcessPoolExecutor() as executor:
#         for n_units in n_units_list:
#             args = [(n_units, L_edge_type, A_edge_type, Y_edge_type, true_L, 
#                      true_A, true_Y, burn_in) for _ in range(n_bootstraps)]
#             results = executor.map(autog, args)
#             estimates[f'n units {n_units}'] = list(results)

#     return estimates

# def autog(network_adj_mat, L, A, Y, burn_in):
#     ''' 
#     Generate a network realization following L_edge_type, A_edge_type, and 
#     Y_edge_type, then estimate causal effects using auto-g.
    
#     This function is used to demonstrate that auto-g produces consistent 
#     estimates of network causal effects when the DGP follows the "UUU" 
#     specification, but it produces biased estimates when there is 
#     latent homophily at the L or the Y layer. 
#     '''
#     # estimate parameters using autog method
#     params_L = minimize(npll_L, x0=np.random.uniform(-1, 1, 2), args=(L, network_adj_mat)).x
#     params_Y = minimize(npll_Y, x0=np.random.uniform(-1, 1, 6), args=(L, A, Y, network_adj_mat)).x

#     # compute causal effects using estimated parameters
#     Y_A1 = estimate_causal_effects_U_U(network_adj_mat, 1, params_L, params_Y, burn_in=burn_in)
#     Y_A0 = estimate_causal_effects_U_U(network_adj_mat, 0, params_L, params_Y, burn_in=burn_in)
    
#     return Y_A1 - Y_A0

# def autog_wrapper(args_dict):
#     """
#     Wrapper for the autog function to fit the common interface.
#     """
#     return autog(
#         network_adj_mat=args_dict['network_adj_mat'], 
#         L=args_dict['L'], 
#         A=args_dict['A'], 
#         Y=args_dict['Y'], 
#         burn_in=args_dict['burn_in']
#     )

# def estimate_causal_effects_U_B_wrapper(args_dict):
#     """
#     Wrapper for the estimate_causal_effects_U_B function to fit the common interface.
#     """
#     return estimate_causal_effects_U_B(
#         network_dict=args_dict['network_dict'], 
#         network_adj_mat=args_dict['network_adj_mat'], 
#         L=args_dict['L'], 
#         A=args_dict['A'], 
#         Y=args_dict['Y'], 
#         n_draws_from_pL=args_dict['n_draws_from_pL'], 
#         gibbs_select_every=args_dict['gibbs_select_every'], 
#         burn_in=args_dict['burn_in']
#     )

# def consistency_test_helper(args):
#     '''
#     estimate_with_wrapper is a wrapper function, such as autog_wrapper()
#     '''
    
#     estimate_with_wrapper, n_units, L_edge_type, A_edge_type, Y_edge_type, \
#         true_L, true_A, true_Y, burn_in, args_dict = args
    
#     def sample_LAY(network_adj_mat, L_edge_type, A_edge_type, Y_edge_type, 
#                 true_L, true_A, true_Y, burn_in):
#         if L_edge_type == "U":
#             L = gibbs_sample_L(network_adj_mat, params=true_L, burn_in=burn_in, 
#                             n_draws=1, select_every=1)[0]
#         elif L_edge_type == "B":
#             L = biedge_sample_L(network_adj_mat, params=true_L)

#         if A_edge_type == "U":
#             A = gibbs_sample_A(network_adj_mat, L, params=true_A, burn_in=burn_in)
#         elif A_edge_type == "B":
#             A = biedge_sample_A(network_adj_mat, L, params=true_A)

#         if Y_edge_type == "U":
#             Y = gibbs_sample_Y(network_adj_mat, L, A, params=true_Y, burn_in=burn_in)
#         elif Y_edge_type == "B":
#             Y = biedge_sample_Y(network_adj_mat, L, A, params=true_Y)
            
#         return L, A, Y 
    
#     # TODO: should i resample network or no?
    
#     # "sample" a network from the true underlyding distribution of networks 
#     network_dict, network_adj_mat = create_random_network(n_units, 1, 6)
    
#     # sample a realization of L, A, Y from the random network
#     L, A, Y = sample_LAY(network_adj_mat, L_edge_type, A_edge_type, Y_edge_type, true_L, true_A, true_Y, burn_in)
    
#     # add the following information to args_dict
#     args_dict['network_dict'] = network_dict
#     args_dict['network_adj_mat'] = network_adj_mat
#     args_dict['L'] = L
#     args_dict['A'] = A
#     args_dict['Y'] = Y
#     args_dict['burn_in'] = burn_in
    
#     return estimate_with_wrapper(args_dict)

def sample_LAY(network_adj_mat, L_edge_type, A_edge_type, Y_edge_type, 
                true_L, true_A, true_Y, burn_in):
    if L_edge_type == "U":
        L = gibbs_sample_L(network_adj_mat, params=true_L, burn_in=burn_in, 
                        n_draws=1, select_every=1)[0]
    elif L_edge_type == "B":
        L = biedge_sample_L(network_adj_mat, params=true_L)

    if A_edge_type == "U":
        A = gibbs_sample_A(network_adj_mat, L, params=true_A, burn_in=burn_in)
    elif A_edge_type == "B":
        A = biedge_sample_A(network_adj_mat, L, params=true_A)

    if Y_edge_type == "U":
        Y = gibbs_sample_Y(network_adj_mat, L, A, params=true_Y, burn_in=burn_in)
    elif Y_edge_type == "B":
        Y = biedge_sample_Y(network_adj_mat, L, A, params=true_Y)
        
    return L, A, Y 

# def consistency_test(n_units_list, 
#                      L_edge_type, A_edge_type, Y_edge_type, 
#                      true_L, true_A, true_Y, burn_in):
#     '''
#     Create a confidence interval of causal effects computed via 
#     estimate_with_wrapper, using data generated from a graphical model 
#     specified with L_edge_type, A_edge_type, and Y_edge_type.
    
#     Arguments:
#         - estimate_with_wrapper: a function
#         - args_dict: a dictionary of arguments to be passed into estimate_with_wrapper
#     '''
#     estimation_errors = []
    
#     for n_units in n_units_list:
#         network_dict, network_adj_mat = create_random_network(n_units, 1, 6)
#         L, A, Y = sample_LAY(network_adj_mat, L_edge_type, A_edge_type, Y_edge_type, true_L, true_A, true_Y, burn_in)

#         # compute causal effects using true parameters
#         Y_A1_true = estimate_causal_effects_U_U(network_adj_mat, 1, true_L, true_Y, burn_in=burn_in)
#         Y_A0_true = estimate_causal_effects_U_U(network_adj_mat, 0, true_L, true_Y, burn_in=burn_in)
#         causal_effect_true = Y_A1_true - Y_A0_true
        
#         # estimate parameters for the L and Y layers using the autog method
#         est_L = minimize(npll_L, x0=np.random.uniform(-1, 1, 2), args=(L, network_adj_mat)).x
#         est_Y = minimize(npll_Y, x0=np.random.uniform(-1, 1, 6), args=(L, A, Y, network_adj_mat)).x

#         # compute causal effects using estimated parameters
#         Y_A1_est = estimate_causal_effects_U_U(network_adj_mat, 1, est_L, est_Y, burn_in=burn_in)
#         Y_A0_est = estimate_causal_effects_U_U(network_adj_mat, 0, est_L, est_Y, burn_in=burn_in)
#         causal_effect_est = Y_A1_est - Y_A0_est

#         estimation_errors.append(causal_effect_est - causal_effect_true)
    
#     return estimation_errors

# def consistency_test(estimate_with_wrapper, n_bootstraps, n_units_list, 
#                      L_edge_type, A_edge_type, Y_edge_type, 
#                      true_L, true_A, true_Y, burn_in, args_dict):
#     '''
#     Create a confidence interval of causal effects computed via 
#     estimate_with_wrapper, using data generated from a graphical model 
#     specified with L_edge_type, A_edge_type, and Y_edge_type.
    
#     Arguments:
#         - estimate_with_wrapper: a function
#         - args_dict: a dictionary of arguments to be passed into estimate_with_wrapper
#     '''
#     estimates = {}
    
#     with ProcessPoolExecutor() as executor:
#         for n_units in n_units_list:
#             args = [(estimate_with_wrapper, n_units, L_edge_type, A_edge_type, Y_edge_type, 
#                      true_L, true_A, true_Y, burn_in, args_dict) for _ in range(n_bootstraps)]
#             results = executor.map(consistency_test_helper, args)
#             estimates[f'n units {n_units}'] = list(results)
            
#     return estimates

def assemble_estimation_df(network, ind_set, L, A, Y):
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
        })

    df = pd.DataFrame(data_list) 
    return df   


# TODO: check consistency of autog
# TODO: implement estimation strategy (predict Yi using A_Ni,i L_Ni,i using 1 hop data) for UBB and check consistency
# TODO: show that UBB and BBB case, when estimated using autog, is inconsistent

# Question: should the true causal effect of the UBB case andt the BBB case be the same?
# Question: for the UBB case, why does changing the coefficient of U change true causal effect?