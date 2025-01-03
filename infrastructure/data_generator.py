from scipy.special import expit
import numpy as np
from tqdm import tqdm
from scipy.stats import multivariate_normal

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
        L = gibbs_sample_Ls(network_adj_mat, params=true_L, burn_in=burn_in, n_draws=1, select_every=1)[0]
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
    
    # Generate samples using the 'rvs' method
    mvn_distribution = multivariate_normal(mean=[mean]*n_sample, cov=cov_mat)
    L = mvn_distribution.rvs(size=n_draws)
    if n_draws == 1:
        L = [L]

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

    pY = expit(params[2] + params[3]*L + params[4]*A + 
               params[5]*(L@network_adj_mat) + 
               params[6]*(A@network_adj_mat) + 
               params[7]*U.sum(axis=0))
    
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
    n_units = network_adj_mat.shape[0]
    n_total_simulations = Ls.shape[0]

    Ys_list = []
    
    # due to memory constraints of our lab machine, sampling is done in several 
    # batches where each batch contains at most 100 simulations.
    # the user of this function should be aware of this limitation.
    for i in range(0, n_total_simulations, 100):
        end_idx = min(i + 100, n_total_simulations)
        current_Ls = Ls[i:end_idx]
        current_As = As[i:end_idx]
        
        # dimension of Us is current_Ls.shape[0] (at most 100) x n_units x n_units
        Us = np.random.normal(loc=params[0], scale=params[1], 
                              size=(current_Ls.shape[0], n_units, n_units))
        Us = np.triu(Us) + np.triu(Us, 1).transpose((0, 2, 1))
        Us = np.where(network_adj_mat == 1, Us, 0)

        # dimension of pY is current_Ls.shape[0] x n_units
        pY = expit(params[2] + params[3]*current_Ls + params[4]*current_As + 
                   params[5]*(current_Ls @ network_adj_mat) + 
                   params[6]*(current_As @ network_adj_mat) + 
                   params[7]*Us.sum(axis=-1)) # sum across the most inner axis of Us

        Ys_list.append(np.random.binomial(1, pY))
        del Us, pY # delete intermiate values for memory efficiency

    # concatenate all batches of samples together
    return np.concatenate(Ys_list, axis=0)

def gibbs_sample_Ls(network_adj_mat, params, burn_in=200, n_draws=1, 
                   select_every=1, data_type="binary"):
    '''
    Sample a single realization of the L layer assuming the presence of an
    undirected edge between the L variables of neighbors of the network. 
    
    Args:
        network_adj_mat (numpy.ndarray): The adjacency matrix of the network.
        params (list): The parameters for the Gibbs sampling.
        burn_in (int, optional): The number of iterations to discard at the beginning of the sampling process. Default is 200.
        n_draws (int, optional): The number of samples to draw from the distribution. Default is 1.
        select_every (int, optional): The interval / rate at which samples are selected from the Gibbs chain. Default is 1.
        data_type (str, optional): The type of data to sample. Can be either "binary" or "continuous". Default is "binary".
        
    Returns:
        Ls (numpy.ndarray): The sampled L vectors from the Gibbs sampling process.
    '''
    Ls = []
    # initialize a vector of Ls
    if data_type == "binary":
        assert len(params) == 2, "params must be a list of length 2 when data_type is 'binary'"
        L = np.random.binomial(1, 0.5, len(network_adj_mat))
    elif data_type == "continuous":
        assert len(params) == 3, "params must be a list of length 3 when data_type is 'continuous'"
        L = np.random.normal(loc=params[0], scale=params[2], size=len(network_adj_mat))
    else:
        raise ValueError("data_type must be either 'binary' or 'continuous'")
    
    # keep sampling an L vector till burn in is done
    for gibbs_iter in tqdm(range(burn_in + n_draws*select_every), desc="Gibbs sampling progress"):
        for i in range(len(network_adj_mat)):
            if data_type == "binary":
                pLi_given_rest = expit(params[0] + params[1]*np.dot(L, network_adj_mat[i, :]))
                L[i] = np.random.binomial(1, pLi_given_rest)
            elif data_type == "continuous":
                L[i] = np.random.normal(loc=params[0] + params[1]*np.dot(L, network_adj_mat[i, :]), scale=params[2])
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

    # keep sampling a Y vector till burn in is done
    with tqdm(total=burn_in*len(network_adj_mat), desc="Sampling progress") as pbar:
        for m in range(burn_in):
            for i in range(len(network_adj_mat)):    
                # pYi_given_rest is a list of probabilities of length n_simulations
                pYi_given_rest = expit(pre_calculated_values[i] + 
                                       params[5]*np.dot(Ys, network_adj_mat[i, :]))

                Ys[:, i] = np.random.binomial(1, pYi_given_rest)
                pbar.update(1)
    return Ys

def f_binary(pa_values):
    weighted_sum = 0
    weights = {
        'U_values': 5,
        'L_self': 0.2,
        'A_self': -0.3,
        'L_neighbors': 0.1,
        'A_neighbors': -0.2
    }

    for key, values in pa_values.items():
        if values is not None and values != []:
            if isinstance(values, list):
                weighted_sum += weights[key] * sum(values)
            else:
                weighted_sum += weights[key] * values
    
    p = expit(weighted_sum)
    return int(np.random.uniform() < p)

def sample_given_boundary_binary(boundary_values):
    ''' 
    Note: This can't be any random function. 
          Check Lauritzen chain graph paper page 342.
    '''
    weighted_sum = 0
    weights = {
        'Y_neighbors': -0.1, # this need to be controlled
        'L_self': 0.8,
        'A_self': 1.7,
        'L_neighbors': -0.1, # this need to be controlled
        'A_neighbors': -0.1 # this need to be controlled
    }
    
    for key, values in boundary_values.items():
        if values is not None and values != []:
            if isinstance(values, list):
                weighted_sum += weights[key] * sum(values)
            else:
                weighted_sum += weights[key] * values

    p = expit(weighted_sum)
    return int(np.random.uniform() < p)