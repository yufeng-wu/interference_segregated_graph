'''
This code:

1. samples a single realization of a given undirected graph
2. samples a single realization of a given bidirected graph
'''

from util import create_random_network, graph_to_edges
from scipy.special import expit
import numpy as np
import pandas as pd
import random

# def sample_from_BG(edges, U_dist, f):
#     '''
#     Given a graph structure, sample a single realization by assuming that there
#     is an unobserved confounder, U, between each pair of connected vertices. 

#     The value of each U is sampled from U_dist.

#     Params:
#         - edges list(tuple(int, int)): a list of edges; each tuple denotes an 
#             edge between two vertices in the graph.
#         - U_dist (callable): a function that returns a sample from the distribution
#             of U when called.
#         - f (callable): the function V = f(U_1, U_2, ..., U_n) + noise
#             that takes a list of U values and returns a value V.  
    
#     Return:
#         - sample (dict: int -> float): sample[vertex_i] = the value of vertex_i

#     Assumptions:
#         - the U's are realizations of the same distribution. (We have to assume this! otherwise misspecification.)
#         - the V's are generated from the same parametric form.
#     '''

#     # Initialize a dictionary to hold the U values "at" each edge
#     edge_to_U = {}

#     for edge in edges:
#         # sample a U value from U_dist
#         U_value = U_dist()
#         # save that in a map: each edge -> U value "associated with" that edge.
#         edge_to_U[edge] = U_value

#     # Initialize a dictionary to hold the V values for each vertex
#     sample = {}

#     graph = edges_to_graph(edges)

#     # For each vertex, find all edges that involve this vertex
#     for vertex in graph.keys():
#         U_values = []
#         for neighbor in graph[vertex]:
#             try:
#                 U_value = edge_to_U[(vertex, neighbor)]
#             except: # try both orders
#                 U_value = edge_to_U[(neighbor, vertex)]
#             U_values.append(U_value)
        
#         # Calculate the value of V using the function f and add normally distributed noise
#         V_value = f(U_values)
        
#         sample[vertex] = V_value

#     return sample


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

def sample_unedge_layer(network, sample, layer, prob_v_given_boundary, verbose=False, burn_in=1000):
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
    V_DOMAIN = [1, 0]
    current_layer = {vertex: random.choice(V_DOMAIN) for vertex in network.keys()}

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

            p = prob_v_given_boundary(boundary_values)

            current_layer[subject] = np.random.choice(V_DOMAIN, size=1, p=np.array([p, 1-p]))[0]

    return current_layer # return the sampled data for THE SPECIFIED LAYER

# def sample_from_UG(network, sample, prob_v_given_boundary, verbose=False, burn_in=1000):
#     '''
#     "boundary" is defined as the boundary in LWF chain graph model for a node. 
#     '''
#     V_DOMAIN = [1, 0]

#     # Initialize vertices with random values chosen from V_DOMAIN
#     v_values = {vertex: random.choice(V_DOMAIN) for vertex in network.keys()}

#     # List to store samples for a particular node, say node 0
#     # node_0_samples = []

#     # Gibbs sampling
#     for i in range(burn_in):
#         if verbose:
#             print("[PROGRESS] Sample from UG burning in:", i, "/", burn_in)
#         for v in network.keys():
#             Y_neighbors = network[v] # returns a list
#             v_neighbor_values = [v_values[neighbor] for neighbor in Y_neighbors]

#             p = prob_v_given_boundary(data={"V_nb_values":v_neighbor_values})

#             # sample a new value for the current node based on conditional proba
#             v_values[v] = np.random.choice(V_DOMAIN, size=1, p=np.array([p, 1-p]))[0]
        
#         # Store the value of node 0 at this iteration
#         # node_0_samples.append(v_values[0])

#     return v_values


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
                                                    prob_v_given_boundary=args['prob_v_given_boundary'],
                                                    verbose=args['verbose'],
                                                    burn_in=args['burn_in'])

        # Add the current sample DataFrame to the list
        samples.append(sample)

    return samples

def U_dist_1():
    return np.random.normal(0, 1)

def f_1(pa_values):
    weighted_sum = 0
    weights = {
        'U_values': -21,
        'L_self': 2,
        'A_self': 3,
        'L_neighbors': 1,
        'A_neighbors': -2
    }

    for key, values in pa_values.items():
        if values is not None and values != []:
            if isinstance(values, list):
                weighted_sum += weights[key] * sum(values)
            else:
                weighted_sum += weights[key] * values
    
    prob = expit(weighted_sum)
    return np.random.binomial(1, prob)  

def prob_v_given_boundary_1(boundary_values):
    '''
    Calculate the conditional probability of a node value given the values of its neighbors and its own values from other layers.

    Params:
        - boundary_values (dict): Dictionary containing 'Y_neighbors', 'L_self', 'A_self', 'L_neighbors', and 'A_neighbors'.

    Return:
        float: Conditional probability value between 0 and 1.
    '''
    weighted_sum = 0
    weights = {
        'Y_neighbors': 1.0,
        'L_self': 0.1,
        'A_self': 3.0,
        'L_neighbors': 0.2,
        'A_neighbors': -2.0
    }
    for key, values in boundary_values.items():
        if values is not None and values != []:
            if isinstance(values, list):
                weighted_sum += weights[key] * sum(values)
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

if __name__ == '__main__':
    NUM_OF_VERTICES = 100
    VERBOSE = True
    BURN_IN = 1

    network = create_random_network(n=NUM_OF_VERTICES, min_neighbors=0, max_neighbors=5)
    
    edge_types = {'L' : ['U', {'prob_v_given_boundary':prob_v_given_boundary_1, 'verbose':VERBOSE, 'burn_in':BURN_IN}], 
                  'A' : ['U', {'prob_v_given_boundary':prob_v_given_boundary_2, 'verbose':VERBOSE, 'burn_in':BURN_IN}], 
                  'Y' : ['U', {'prob_v_given_boundary':prob_v_given_boundary_3, 'verbose':VERBOSE, 'burn_in':BURN_IN}]}
    
    samples = sample_L_A_Y(n_samples=1, network=network, edge_types=edge_types)
    