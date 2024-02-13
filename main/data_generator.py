'''
This code:

1. samples a single realization of a given undirected graph
2. samples a single realization of a given bidirected graph
'''

from util import create_random_network, graph_to_edges
from scipy.special import expit
from scipy.stats import norm, beta
import numpy as np
import pandas as pd
import random
import networkx as nx
from scipy.sparse import csr_matrix

def sample_biedge_layer(network, sample, layer, U_dist, f):
    '''
    Modified function to sample from a Bidirected Graph (BG) considering layer-specific dependencies
    and passing values into the function 'f' as a structured dictionary.

    Params:
        - network (dict): Graph structure where the key is a node and the value is a list of its neighbors.
        - sample (DataFrame): DataFrame containing the current samples of each node for previous (as defined by topological ordering) layers.
        - layer (str): The layer ('L', 'A', 'Y') currently being sampled.
        - U_dist (callable): Function to sample the unobserved confounder U.
        - f (callable): Function to calculate the value of a node given a structured dictionary of parents (pa(V)).

    Return:
        - data (dict): Dictionary with the sampled values for each node in the specified layer.
    '''

    # Initialize a dictionary to hold the U values for each pair of connected vertices
    pair_to_U = {}
    for edge in graph_to_edges(network):
        U_value = U_dist()
        pair_to_U[edge] = U_value

    data = {}

    for subject in network.keys():
        pa_values = {
            'U_values': [pair_to_U[tuple(sorted((subject, neighbor)))] for neighbor in network[subject]],
            'L_self': None,
            'A_self': None,
            'L_neighbors': [],
            'A_neighbors': []
        }

        if layer in ['A', 'Y']:
            pa_values['L_self'] = sample.loc[subject, 'L']
            pa_values['L_neighbors'] = [sample.loc[neighbor, 'L'] for neighbor in network[subject]]
        
        if layer == 'Y':
            pa_values['A_self'] = sample.loc[subject, 'A']
            pa_values['A_neighbors'] = [sample.loc[neighbor, 'A'] for neighbor in network[subject]]

        data[subject] = f(pa_values)

    return data

def sample_unedge_layer(network, sample, layer, sample_given_boundary, verbose=False, burn_in=1000):
    '''
    Function to sample from an Undirected Graph (UG).

    Params:
        - network (dict): Graph structure where the key is a node and the value is a list of its neighbors.
        - sample (DataFrame): DataFrame containing the current samples of each node for previous (as defined by topological ordering) layers.
        - layer (str): The layer ('L', 'A', 'Y') currently being sampled.
        - prob_v_given_boundary (callable): Function to calculate the conditional probability of a node given its neighbors.
        - verbose (bool): Flag for printing progress messages.
        - burn_in (int): Number of iterations for the Gibbs sampling 'burn-in' period.

    Return:
        - data (dict): Dictionary with the sampled values for each node in the specified layer.
    '''
    # generate random initial values for variables at the current layer
    # V_DOMAIN = [1, 0]
    # current_layer = {vertex: random.choice(V_DOMAIN) for vertex in network.keys()}

    # generate random initial values for variables at the current layer
    current_layer = {vertex: np.random.normal(loc=0, scale=1) for vertex in network.keys()}

    for i in range(burn_in):
        if verbose:
            print("[PROGRESS] Sample from UG burning in:", i, "/", burn_in)
        for subject in network.keys():
            boundary_values = {
                'L_self': None,
                'L_neighbors': [],
                'A_self': None,
                'A_neighbors': [],
                'Y_neighbors': [],
            }

            if layer == "L":
                boundary_values['L_neighbors'] = [current_layer[neighbor] for neighbor in network[subject]]

            if layer == "A":
                boundary_values['L_self'] = sample.loc[subject, 'L']
                boundary_values['L_neighbors'] = [sample.loc[neighbor, 'L'] for neighbor in network[subject]]
                boundary_values['A_neighbors'] = [current_layer[neighbor] for neighbor in network[subject]]

            if layer == 'Y':
                boundary_values['L_self'] = sample.loc[subject, 'L']
                boundary_values['L_neighbors'] = [sample.loc[neighbor, 'L'] for neighbor in network[subject]]
                boundary_values['A_self'] = sample.loc[subject, 'A']
                boundary_values['A_neighbors'] = [sample.loc[neighbor, 'A'] for neighbor in network[subject]]
                boundary_values['Y_neighbors'] = [current_layer[neighbor] for neighbor in network[subject]]

            current_layer[subject] = sample_given_boundary(boundary_values) # np.random.choice(V_DOMAIN, size=1, p=np.array([p, 1-p]))[0]

    return current_layer # return the sampled data for THE SPECIFIED LAYER

def sample_biedge_L_layer_cont(network, max_neighbors):
    adjacency_matrix = csr_matrix(nx.adjacency_matrix(nx.from_dict_of_lists(network)))
    c = np.random.uniform(0, 1 / max_neighbors)
    covariance_matrix = c * adjacency_matrix
    np.fill_diagonal(covariance_matrix.toarray(), 1)  # Convert to dense array for fill_diagonal

    # Generate a standard normal sample for each node
    standard_normal_samples = np.random.normal(size=adjacency_matrix.shape[0])

    # Transform the standard normal samples
    sample = covariance_matrix @ standard_normal_samples
    
    df = pd.DataFrame({"L": sample})

    return df

def sample_L_A_Y(n_samples, network, edge_types):
    '''
    A function to draw n_samples samples from the joint distribution p(V) corresponding
    to a graph specified by 
    '''
    # List to store each sample as a DataFrame
    samples = []

    for _ in range(n_samples):
        # Initialize DataFrame for the current sample
        sample = pd.DataFrame(index=network.keys(), columns=['L', 'A', 'Y'])

        # Sample each layer
        for layer in ['L', 'A', 'Y']:
            try:
                edge_type, args = edge_types[layer]
            except:
                continue
            if edge_type == 'B':
                sample[layer] = sample_biedge_layer(network=network, 
                                                    sample=sample, 
                                                    layer=layer, 
                                                    U_dist=args['U_dist'], 
                                                    f=args['f'])
            elif edge_type == 'U':
                sample[layer] = sample_unedge_layer(network=network,
                                                    sample=sample,
                                                    layer=layer,
                                                    sample_given_boundary=args['sample_given_boundary'],
                                                    verbose=args['verbose'],
                                                    burn_in=args['burn_in'])

        # Add the current sample DataFrame to the list
        samples.append(sample.copy())

    return samples

def U_dist_1():
    return np.random.normal(0, 1)

def f_1(pa_values):
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
    
    noise = np.random.normal(0, 1)
    return weighted_sum + noise

def f_non_linear(pa_values):
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
                weighted_sum += weights[key] * (sum(values)) ** 2
            else:
                weighted_sum += weights[key] * (values) ** 2
    
    noise = np.random.normal(0, 1)
    return weighted_sum + noise

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
    
    noise = np.random.normal(0, 1)
    p = expit(weighted_sum + noise)
    return int(np.random.uniform() < p)

def prob_v_given_boundary_1(boundary_values):
    weighted_sum = 0
    weights = {
        'Y_neighbors': 1.0,
        'L_self': 1,
        'A_self': 3.0,
        'L_neighbors': 0.5,
        'A_neighbors': -2.0
    }

    for key, values in boundary_values.items():
        if values is not None and values != []:
            if isinstance(values, list):
                weighted_sum += weights[key] * sum(values) - 0.5
            else:
                weighted_sum += weights[key] * values
    return expit(weighted_sum)

def prob_v_given_boundary_2(boundary_values):
    weighted_sum = 0
    weights = {
        'Y_neighbors': 0.4,
        'L_self': -2.0,
        'A_self': -1.4,
        'L_neighbors': 1.4,
        'A_neighbors': -2.4
    }
    for key, values in boundary_values.items():
        if values is not None and values != []:
            if isinstance(values, list):
                weighted_sum += weights[key] * sum(values)
            else:
                weighted_sum += weights[key] * values
    return expit(weighted_sum)

def prob_v_given_boundary_3(boundary_values):
    weighted_sum = 0
    weights = {
        'Y_neighbors': 0.2,
        'L_self': -0.8,
        'A_self': 1.7,
        'L_neighbors': -0.31,
        'A_neighbors': 0.4
    }
    for key, values in boundary_values.items():
        if values is not None and values != []:
            if isinstance(values, list):
                weighted_sum += weights[key] * sum(values)
            else:
                weighted_sum += weights[key] * values
    return expit(weighted_sum)

def sample_given_boundary_continuous(boundary_values):
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

    return weighted_sum + np.random.normal(0, 1)


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

    noise = np.random.normal(0, 0.1)
    p = expit(weighted_sum + noise)
    return int(np.random.uniform() < p)