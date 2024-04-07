from .network_utils import create_random_network, graph_to_edges
from scipy.special import expit
from scipy.stats import norm, beta
import numpy as np
import pandas as pd
import random
import networkx as nx
from scipy.sparse import csr_matrix
from tqdm import tqdm

def sample_LAY(network_adj_mat, L_edge_type, A_edge_type, Y_edge_type, 
                true_L, true_A, true_Y, burn_in):
    '''
    Sample a single realization of the L, A, Y layers given the network adjacency 
    matrix and the true parameters for each layer.
    
    Args:
        network_adj_mat: a numpy array representing the network adjacency matrix
        L_edge_type: the edge type for the L layer. Can be either "U" or "B",
            where "U" stands for undirected and "B" stands for bidirected.
        A_edge_type: the edge type for the A layer. Can be either "U" or "B"
        Y_edge_type: the edge type for the Y layer. Can be either "U" or "B"
        true_L: a list of parameters for the L layer, whose required length 
            and interpretation depends on the edge type of the L layer.
        true_A: a list of parameters for the A layer, whose required length
            and interpretation depends on the edge type of the A layer.
        true_Y: a list of parameters for the Y layer, whose required length
            and interpretation depends on the edge type of the Y layer.
        burn_in: the number of burn-in iterations to run before sampling the
            L, A, Y layers, if they are sampled using a Gibbs sampler.
    
    Return:
        a tuple of the form (L, A, Y), where L, A, Y are numpy arrays representing
        the sampled realizations of the L, A, Y layers, respectively.
    '''
    if L_edge_type == "U":
        assert len(true_L) == 2, "true_L must be a list of length 2 when L_edge_type is 'U'"
        L = gibbs_sample_L(network_adj_mat, params=true_L, burn_in=burn_in, n_draws=1, select_every=1)[0]
    elif L_edge_type == "B":
        assert len(true_L) == 3, "true_L must be a list of length 3 when L_edge_type is 'B'"
        L = biedge_sample_Ls(network_adj_mat, params=true_L, n_draws=1)[0]

    if A_edge_type == "U":
        assert len(true_A) == 4, "true_A must be a list of length 4 when A_edge_type is 'U'"
        A = gibbs_sample_A(network_adj_mat, L, params=true_A, burn_in=burn_in)
    elif A_edge_type == "B":
        assert len(true_A) == 6, "true_A must be a list of length 6 when A_edge_type is 'B'"
        A = biedge_sample_A(network_adj_mat, L, params=true_A)

    if Y_edge_type == "U":
        assert len(true_Y) == 6, "true_Y must be a list of length 6 when Y_edge_type is 'U'"
        Y = gibbs_sample_Y(network_adj_mat, L, A, params=true_Y, burn_in=burn_in)
    elif Y_edge_type == "B":
        assert len(true_Y) == 8, "true_Y must be a list of length 8 when Y_edge_type is 'B'"
        Y = biedge_sample_Y(network_adj_mat, L, A, params=true_Y)
        
    return L, A, Y 

def biedge_sample_Ls(network_adj_mat, params, n_draws=1):
    '''
    Sample n_draws realization(s) of the L layer assuming the presence of a 
    bidirected edge between the L variables of neighbors of the network.
    
    Args:
        network_adj_mat: a numpy array representing the network adjacency matrix
        params: a list of parameters for the L layer. 
        n_draws: the number of samples to draw from the distribution of L. Default is 1.
        
    Return:
        a n_draws x n_units dimensional numpy array representing the sampled
        realization(s) of the L layer, where the index of each draw corresponds
        to the index of the node in the network.
    '''
    cov, var, mean = params # unpack params
    n_sample = len(network_adj_mat)
    
    cov_mat = np.full(network_adj_mat.shape, cov)
    cov_mat = np.where(network_adj_mat > 0, cov_mat, 0.0)
    np.fill_diagonal(cov_mat, var)
    
    L = np.random.multivariate_normal([mean]*n_sample, cov_mat, size=n_draws)
           
    # try:
    #     L = np.random.multivariate_normal([mean]*n_sample, cov_mat, size=n_draws)
    # except RuntimeWarning as rw:
    #     print("Warning occurred:", rw)
    #     print("COV, VAR, MEAN:", cov, var, mean)
    #     print(cov_mat)
    #     print("MAX DEG", np.max(np.sum(network_adj_mat, axis=1)))
    #     L = []
    
    # else:
    #     mean, std, beta_0, beta_1 = params # unpack params
        
    #     U = np.random.normal(loc=mean, scale=std, size=network_adj_mat.shape)
    #     U = np.triu(U) + np.triu(U, 1).T # make U symmetric by copying the upper triangular to the lower triangular part
    #     U = np.where(network_adj_mat == 1, U, network_adj_mat) # apply the network mask

    #     pL = expit(beta_0 + beta_1*U.sum(axis=0)) # pL is a vector 
    #     L = np.random.binomial(1, pL)

    return L

def biedge_sample_A(network_adj_mat, L, params):
    '''
    Sample a single realization of the A layer assuming the presence of a 
    bidirected edge between the A variables of neighbors of the network.
    
    Args:
        network_adj_mat: a numpy array representing the network adjacency matrix.
        L: a numpy array representing the sampled realization of the L layer.
        params: a list of parameters for the A layer.
        
    Return:
        a length n_units numpy array representing the sampled realization of 
        the A layer, where the index of the array corresponds to the index 
        of the node in the network.
    '''
    U = np.random.normal(loc=params[0], scale=params[1], size=network_adj_mat.shape)
    U = np.triu(U) + np.triu(U, 1).T  # make U symmetric
    U = np.where(network_adj_mat == 1, U, network_adj_mat)  # apply network mask

    pA = expit(params[2] + params[3]*L + params[4]*(L@network_adj_mat) + params[5]*U.sum(axis=0))
    A = np.random.binomial(1, pA)
    return A

def biedge_sample_Y(network_adj_mat, L, A, params):
    '''
    Sample a single realization of the Y layer assuming the presence of a
    bidirected edge between the Y variables of neighbors of the network.
    
    Args:
        network_adj_mat: a numpy array representing the network adjacency matrix.
        L: a numpy array representing the sampled realization of the L layer.
        A: a numpy array representing the sampled realization of the A layer.
        params: a list of parameters for the Y layer.
    
    Return:
        a length n_units numpy array representing the sampled realization of
        the Y layer, where the index of the array corresponds to the index
        of the node in the network.   
    '''
    U = np.random.normal(loc=params[0], scale=params[1], size=network_adj_mat.shape)
    U = np.triu(U) + np.triu(U, 1).T  # make U symmetric
    U = np.where(network_adj_mat == 1, U, 0)  # apply network mask

    pY = expit(params[2] + params[3]*L + params[4]*A + params[5]*(L@network_adj_mat) + 
               params[6]*(A@network_adj_mat) + params[7]*U.sum(axis=0))
    
    Y = np.random.binomial(1, pY)
    return Y

def biedge_sample_Ys(network_adj_mat, Ls, As, params):
    '''
    Sample multiple realizations of the Y layer assuming the presence of a
    bidirected edge between the Y variables of neighbors of the network.
    
    The number of realizations to sample is determined by the number of rows
    in the Ls and As arrays.
    
    Args:
        network_adj_mat: a numpy array representing the network adjacency matrix.
        Ls: a n_simulations x n_units dimensional numpy array representing 
            the sampled realizations of the L layer.
        As: a n_simulations x n_units dimensional numpy array representing
            the sampled realizations of the A layer.
        params: a list of parameters for the Y layer.
        
    Return:
        Ys: a n_simulations x n_units dimensional numpy array representing the
            sampled realizations of the Y layer, where the index of each simulation
            corresponds to the index of the node in the network.
    '''
    # dimension of Us is n_simulations x n_units x n_units
    Us = np.random.normal(loc=params[0], 
                          scale=params[1], 
                          size=(Ls.shape[0], # n_simulations
                                network_adj_mat.shape[0], # n_units
                                network_adj_mat.shape[1])) # n_units
    print("samling of Us done")
    
    Us = np.triu(Us) + np.triu(Us, 1).transpose((0, 2, 1))  # make U symmetric
    Us = np.where(network_adj_mat == 1, Us, 0)  # apply network mask
    
    # dimension of pY is n_simulations x n_units
    pY = expit(params[2] + 
               params[3]*Ls + 
               params[4]*As + 
               params[5]*(Ls@network_adj_mat) + 
               params[6]*(As@network_adj_mat) + 
               params[7]*Us.sum(axis=-1)) # sum across the most inner axis of Us
    
    del Us # delete for memory efficiency
    
    # dimension of Ys is n_simulations x n_units
    Ys = np.random.binomial(1, pY)
    
    del pY # delete for memory efficiency
    return Ys

def gibbs_sample_L(network_adj_mat, params, burn_in=200, n_draws=1, select_every=1):
    '''
    Sample a single realization of the L layer assuming the presence of an
    undirected edge between the L variables of neighbors of the network. 
    
    Args:
        network_adj_mat (numpy.ndarray): The adjacency matrix of the network.
        params (list): The parameters for the Gibbs sampling.
        burn_in (int, optional): The number of iterations to discard at the beginning of the sampling process. Default is 200.
        n_draws (int, optional): The number of samples to draw from the distribution. Default is 1.
        select_every (int, optional): The interval / rate at which samples are selected from the Gibbs chain. Default is 1.
    
    Returns:
        Ls (numpy.ndarray): The sampled L vectors from the Gibbs sampling process.
    '''
    Ls = []
    # initialize a vector of Ls
    L = np.random.binomial(1, 0.5, len(network_adj_mat))

    # keep sampling an L vector till burn in is done
    for gibbs_iter in tqdm(range(burn_in + n_draws*select_every), desc="Gibbs sampling progress"):
        for i in range(len(network_adj_mat)):
            pLi_given_rest = expit(params[0] + params[1]*np.dot(L, network_adj_mat[i, :]))
            L[i] = np.random.binomial(1, pLi_given_rest)

        if gibbs_iter >= burn_in and gibbs_iter % select_every == 0:
            Ls.append(L.copy())
    Ls = np.array(Ls)
    return Ls

def gibbs_sample_A(network_adj_mat, L, params, burn_in=200):
    '''
    Sample a single realization of the A layer assuming the presence of an
    undirected edge between the A variables of neighbors of the network.

    Args:
        network_adj_mat (numpy.ndarray): The network adjacency matrix.
        L (numpy.ndarray): A numpy array representing the sampled realization of the L layer.
        params (list): The parameters for the Gibbs sampling.
        burn_in (int): The number of burn-in iterations. Default is 200.

    Returns:
        A (numpy.ndarray): The sampled A vectors from the Gibbs sampling process.
    '''
    # Initialize A with random binary values
    A = np.random.binomial(1, 0.5, len(network_adj_mat))

    # Perform Gibbs sampling for burn_in iterations
    for m in range(burn_in):
        for i in range(len(network_adj_mat)):
            # Calculate the probability of Ai given the rest of the variables
            pAi_given_rest = expit(params[0] + params[1]*L[i] +
                                   params[2]*np.dot(A, network_adj_mat[i, :]) +
                                   params[3]*np.dot(L, network_adj_mat[i, :]))
            # Sample Ai from a binomial distribution
            A[i] = np.random.binomial(1, pAi_given_rest)
    return A

def gibbs_sample_Y(network_adj_mat, L, A, params, burn_in=200):
    '''
    Sample a single realization of the Y layer assuming the presence of an
    undirected edge between the Y variables of neighbors of the network.

    Args:
        network_adj_mat (numpy.ndarray): The network adjacency matrix.
        L (numpy.ndarray): The sampled realization of the L layer.
        A (numpy.ndarray): The sampled realization of the A layer.
        params (list): The parameters for the Gibbs sampling.
        burn_in (int): The number of burn-in iterations. Default is 200.

    Returns:
        A (numpy.ndarray): The sampled Y vectors from the Gibbs sampling process.
    '''
    Y = np.random.binomial(1, 0.5, len(network_adj_mat))

    # keep sampling an Y vector till burn in is done
    for m in tqdm(range(burn_in), desc="Gibbs sampling progress"):
        for i in range(len(network_adj_mat)):
            pYi_given_rest = expit(params[0] + params[1]*L[i] + params[2]*A[i] +
                                   params[3]*np.dot(L, network_adj_mat[i, :]) +
                                   params[4]*np.dot(A, network_adj_mat[i, :]) +
                                   params[5]*np.dot(Y, network_adj_mat[i, :]))
            Y[i] = np.random.binomial(1, pYi_given_rest)
    return Y

def gibbs_sample_Ys(network_adj_mat, Ls, As, params, burn_in=200):
    '''
    Sample multiple realizations of the Y layer assuming the presence of an
    undirected edge between the Y variables of neighbors of the network.
    
    The number of realizations to sample is determined by the number of rows
    in the Ls and As arrays.
    
    Args:
        network_adj_mat: a numpy array representing the network adjacency matrix.
        Ls: a n_simulations x n_units dimensional numpy array representing 
            the sampled realizations of the L layer.
        As: a n_simulations x n_units dimensional numpy array representing
            the sampled realizations of the A layer.
        params: a list of parameters for the Y layer.
        burn_in: the number of burn-in iterations. Default is 200.
        
    Return:
        Ys: a n_simulations x n_units dimensional numpy array representing the
            sampled realizations of the Y layer, where the index of each simulation
            corresponds to the index of the node in the network.
    '''
    # initialize Ys as a 2D array with the same shape as Ls and As
    Ys = np.random.binomial(1, 0.5, Ls.shape)
    
    # Trick to speed up the sampling process:
    # pre_calculated_values is of shape n_units x n_simulations
    # where pre_calculated_values[i] is a list of pre-calculated values for the 
    # ith unit of the network across all simulations
    pre_calculated_values = [params[0] + 
                            params[1]*Ls[:, i] + 
                            params[2]*As[:, i] +
                            params[3]*np.dot(Ls, network_adj_mat[i, :]) +
                            params[4]*np.dot(As, network_adj_mat[i, :])
                            for i in range(len(network_adj_mat))]

    # keep sampling an Y vector till burn in is done
    with tqdm(total=burn_in*len(network_adj_mat), desc="Sampling progress") as pbar:
        for m in range(burn_in):
            for i in range(len(network_adj_mat)):    
                # pYi_given_rest is a list of probabilities of length n_simulations
                pYi_given_rest = expit(pre_calculated_values[i] + 
                                       params[5]*np.dot(Ys, network_adj_mat[i, :]))

                Ys[:, i] = np.random.binomial(1, pYi_given_rest)
                pbar.update(1)
    return Ys



# TODO: below is the original data generator code for contagion_vs_latent_homophily tests
#       which might be still useful.

# def sample_biedge_layer(network, sample, layer, U_dist, f):
#     '''
#     Modified function to sample from a Bidirected Graph (BG) considering layer-specific dependencies
#     and passing values into the function 'f' as a structured dictionary.

#     Params:
#         - network (dict): Graph structure where the key is a node and the value is a list of its neighbors.
#         - sample (DataFrame): DataFrame containing the current samples of each node for previous (as defined by topological ordering) layers.
#         - layer (str): The layer ('L', 'A', 'Y') currently being sampled.
#         - U_dist (callable): Function to sample the unobserved confounder U.
#         - f (callable): Function to calculate the value of a node given a structured dictionary of parents (pa(V)).

#     Return:
#         - data (dict): Dictionary with the sampled values for each node in the specified layer.
#     '''

#     # Initialize a dictionary to hold the U values for each pair of connected vertices
#     pair_to_U = {}
#     for edge in graph_to_edges(network):
#         U_value = U_dist()
#         pair_to_U[edge] = U_value

#     data = {}

#     for subject in network.keys():
#         pa_values = {
#             'U_values': [pair_to_U[tuple(sorted((subject, neighbor)))] for neighbor in network[subject]],
#             'L_self': None,
#             'A_self': None,
#             'L_neighbors': [],
#             'A_neighbors': []
#         }

#         if layer in ['A', 'Y']:
#             pa_values['L_self'] = sample.loc[subject, 'L']
#             pa_values['L_neighbors'] = [sample.loc[neighbor, 'L'] for neighbor in network[subject]]
        
#         if layer == 'Y':
#             pa_values['A_self'] = sample.loc[subject, 'A']
#             pa_values['A_neighbors'] = [sample.loc[neighbor, 'A'] for neighbor in network[subject]]

#         data[subject] = f(pa_values)

#     return data

# def sample_unedge_layer(network, sample, layer, sample_given_boundary, verbose=False, burn_in=1000):
#     '''
#     Function to sample from an Undirected Graph (UG).

#     Params:
#         - network (dict): Graph structure where the key is a node and the value is a list of its neighbors.
#         - sample (DataFrame): DataFrame containing the current samples of each node for previous (as defined by topological ordering) layers.
#         - layer (str): The layer ('L', 'A', 'Y') currently being sampled.
#         - prob_v_given_boundary (callable): Function to calculate the conditional probability of a node given its neighbors.
#         - verbose (bool): Flag for printing progress messages.
#         - burn_in (int): Number of iterations for the Gibbs sampling 'burn-in' period.

#     Return:
#         - data (dict): Dictionary with the sampled values for each node in the specified layer.
#     '''
#     # generate random initial values for variables at the current layer
#     # V_DOMAIN = [1, 0]
#     # current_layer = {vertex: random.choice(V_DOMAIN) for vertex in network.keys()}

#     # generate random initial values for variables at the current layer
#     current_layer = {vertex: np.random.normal(loc=0, scale=1) for vertex in network.keys()}

#     for i in range(burn_in):
#         if verbose:
#             print("[PROGRESS] Sample from UG burning in:", i, "/", burn_in)
#         for subject in network.keys():
#             boundary_values = {
#                 'L_self': None,
#                 'L_neighbors': [],
#                 'A_self': None,
#                 'A_neighbors': [],
#                 'Y_neighbors': [],
#             }

#             if layer == "L":
#                 boundary_values['L_neighbors'] = [current_layer[neighbor] for neighbor in network[subject]]

#             if layer == "A":
#                 boundary_values['L_self'] = sample.loc[subject, 'L']
#                 boundary_values['L_neighbors'] = [sample.loc[neighbor, 'L'] for neighbor in network[subject]]
#                 boundary_values['A_neighbors'] = [current_layer[neighbor] for neighbor in network[subject]]

#             if layer == 'Y':
#                 boundary_values['L_self'] = sample.loc[subject, 'L']
#                 boundary_values['L_neighbors'] = [sample.loc[neighbor, 'L'] for neighbor in network[subject]]
#                 boundary_values['A_self'] = sample.loc[subject, 'A']
#                 boundary_values['A_neighbors'] = [sample.loc[neighbor, 'A'] for neighbor in network[subject]]
#                 boundary_values['Y_neighbors'] = [current_layer[neighbor] for neighbor in network[subject]]

#             current_layer[subject] = sample_given_boundary(boundary_values) # np.random.choice(V_DOMAIN, size=1, p=np.array([p, 1-p]))[0]

#     return current_layer # return the sampled data for THE SPECIFIED LAYER

# def sample_biedge_L_layer_cont(network, max_neighbors):
#     adjacency_matrix = csr_matrix(nx.adjacency_matrix(nx.from_dict_of_lists(network)))
#     c = np.random.uniform(0, 1 / max_neighbors)
#     covariance_matrix = c * adjacency_matrix
#     np.fill_diagonal(covariance_matrix.toarray(), 1)  # Convert to dense array for fill_diagonal

#     # Generate a standard normal sample for each node
#     standard_normal_samples = np.random.normal(size=adjacency_matrix.shape[0])

#     # Transform the standard normal samples
#     sample = covariance_matrix @ standard_normal_samples
    
#     df = pd.DataFrame({"L": sample})

#     return df

# def sample_L_A_Y(n_samples, network, edge_types):
#     '''
#     A function to draw n_samples samples from the joint distribution p(V) corresponding
#     to a graph specified by 
#     '''
#     # List to store each sample as a DataFrame
#     samples = []

#     for _ in range(n_samples):
#         # Initialize DataFrame for the current sample
#         sample = pd.DataFrame(index=network.keys(), columns=['L', 'A', 'Y'])

#         # Sample each layer
#         for layer in ['L', 'A', 'Y']:
#             try:
#                 edge_type, args = edge_types[layer]
#             except:
#                 continue
#             if edge_type == 'B':
#                 sample[layer] = sample_biedge_layer(network=network, 
#                                                     sample=sample, 
#                                                     layer=layer, 
#                                                     U_dist=args['U_dist'], 
#                                                     f=args['f'])
#             elif edge_type == 'U':
#                 sample[layer] = sample_unedge_layer(network=network,
#                                                     sample=sample,
#                                                     layer=layer,
#                                                     sample_given_boundary=args['sample_given_boundary'],
#                                                     verbose=args['verbose'],
#                                                     burn_in=args['burn_in'])

#         # Add the current sample DataFrame to the list
#         samples.append(sample.copy())

#     return samples

# def U_dist_1():
#     return np.random.normal(0, 1)

# def f_1(pa_values):
#     weighted_sum = 0
#     weights = {
#         'U_values': 5,
#         'L_self': 0.2,
#         'A_self': -0.3,
#         'L_neighbors': 0.1,
#         'A_neighbors': -0.2
#     }

#     for key, values in pa_values.items():
#         if values is not None and values != []:
#             if isinstance(values, list):
#                 weighted_sum += weights[key] * sum(values)
#             else:
#                 weighted_sum += weights[key] * values
    
#     noise = np.random.normal(0, 1)
#     return weighted_sum + noise

# def f_non_linear(pa_values):
#     weighted_sum = 0
#     weights = {
#         'U_values': 5,
#         'L_self': 0.2,
#         'A_self': -0.3,
#         'L_neighbors': 0.1,
#         'A_neighbors': -0.2
#     }

#     for key, values in pa_values.items():
#         if values is not None and values != []:
#             if isinstance(values, list):
#                 weighted_sum += weights[key] * (sum(values)) ** 2
#             else:
#                 weighted_sum += weights[key] * (values) ** 2
    
#     noise = np.random.normal(0, 1)
#     return weighted_sum + noise

# def f_binary(pa_values):
#     weighted_sum = 0
#     weights = {
#         'U_values': 5,
#         'L_self': 0.2,
#         'A_self': -0.3,
#         'L_neighbors': 0.1,
#         'A_neighbors': -0.2
#     }

#     for key, values in pa_values.items():
#         if values is not None and values != []:
#             if isinstance(values, list):
#                 weighted_sum += weights[key] * sum(values)
#             else:
#                 weighted_sum += weights[key] * values
    
#     noise = np.random.normal(0, 1)
#     p = expit(weighted_sum + noise)
#     return int(np.random.uniform() < p)

# def prob_v_given_boundary_1(boundary_values):
#     weighted_sum = 0
#     weights = {
#         'Y_neighbors': 1.0,
#         'L_self': 1,
#         'A_self': 3.0,
#         'L_neighbors': 0.5,
#         'A_neighbors': -2.0
#     }

#     for key, values in boundary_values.items():
#         if values is not None and values != []:
#             if isinstance(values, list):
#                 weighted_sum += weights[key] * sum(values) - 0.5
#             else:
#                 weighted_sum += weights[key] * values
#     return expit(weighted_sum)

# def prob_v_given_boundary_2(boundary_values):
#     weighted_sum = 0
#     weights = {
#         'Y_neighbors': 0.4,
#         'L_self': -2.0,
#         'A_self': -1.4,
#         'L_neighbors': 1.4,
#         'A_neighbors': -2.4
#     }
#     for key, values in boundary_values.items():
#         if values is not None and values != []:
#             if isinstance(values, list):
#                 weighted_sum += weights[key] * sum(values)
#             else:
#                 weighted_sum += weights[key] * values
#     return expit(weighted_sum)

# def prob_v_given_boundary_3(boundary_values):
#     weighted_sum = 0
#     weights = {
#         'Y_neighbors': 0.2,
#         'L_self': -0.8,
#         'A_self': 1.7,
#         'L_neighbors': -0.31,
#         'A_neighbors': 0.4
#     }
#     for key, values in boundary_values.items():
#         if values is not None and values != []:
#             if isinstance(values, list):
#                 weighted_sum += weights[key] * sum(values)
#             else:
#                 weighted_sum += weights[key] * values
#     return expit(weighted_sum)

# def sample_given_boundary_continuous(boundary_values):
#     '''
#     Note: This can't be any random function. 
#           Check Lauritzen chain graph paper page 342.
#     '''
#     weighted_sum = 0
#     weights = {
#         'Y_neighbors': -0.1, # this need to be controlled
#         'L_self': 0.8,
#         'A_self': 1.7,
#         'L_neighbors': -0.1, # this need to be controlled
#         'A_neighbors': -0.1 # this need to be controlled
#     }
    
#     for key, values in boundary_values.items():
#         if values is not None and values != []:
#             if isinstance(values, list):
#                 weighted_sum += weights[key] * sum(values)
#             else:
#                 weighted_sum += weights[key] * values

#     return weighted_sum + np.random.normal(0, 1)


# def sample_given_boundary_binary(boundary_values):
#     '''
#     Note: This can't be any random function. 
#           Check Lauritzen chain graph paper page 342.
#     '''
#     weighted_sum = 0
#     weights = {
#         'Y_neighbors': -0.1, # this need to be controlled
#         'L_self': 0.8,
#         'A_self': 1.7,
#         'L_neighbors': -0.1, # this need to be controlled
#         'A_neighbors': -0.1 # this need to be controlled
#     }
    
#     for key, values in boundary_values.items():
#         if values is not None and values != []:
#             if isinstance(values, list):
#                 weighted_sum += weights[key] * sum(values)
#             else:
#                 weighted_sum += weights[key] * values

#     noise = np.random.normal(0, 0.1)
#     p = expit(weighted_sum + noise)
#     return int(np.random.uniform() < p)