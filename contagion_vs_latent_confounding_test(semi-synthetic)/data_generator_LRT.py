'''
This data generator is used to generate the semi-synthetic data using network 
structure from raw_data folder. 

The sampling procedure and parameters are exactly the same as the synthetic data
version (data_generator_LRT.py) except that the network structure is loaded from
the raw_data folder.
'''

# data generator for the liklihood ratio test (LRT)
import sys
sys.path.append("..")
from infrastructure.network_utils import *
from infrastructure.maximal_independent_set import *

from scipy.special import expit
import numpy as np
import pandas as pd
import pickle

def sample_biedge_layer(network_dict, sample, layer, U_dist, f):
    '''
    Function to sample from a segregated graph with only bidirected edges.

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
    for edge in graph_to_edges(network_dict):
        U_value = U_dist()
        pair_to_U[edge] = U_value

    data = {}

    for subject in network_dict.keys():
        pa_values = {
            'U_values': [pair_to_U[tuple(sorted((subject, neighbor)))] for neighbor in network_dict[subject]],
            'L_self': None,
            'A_self': None,
            'L_neighbors': [],
            'A_neighbors': []
        }

        if layer in ['A', 'Y']:
            pa_values['L_self'] = sample.loc[subject, 'L']
            pa_values['L_neighbors'] = [sample.loc[neighbor, 'L'] for neighbor in network_dict[subject]]
        
        if layer == 'Y':
            pa_values['A_self'] = sample.loc[subject, 'A']
            pa_values['A_neighbors'] = [sample.loc[neighbor, 'A'] for neighbor in network_dict[subject]]

        data[subject] = f(pa_values)

    return data

def sample_unedge_layer(network_dict, sample, layer, sample_given_boundary, verbose=False, burn_in=1000):
    '''
    Function to sample from an undirected graph.

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
    current_layer = {vertex: random.choice(V_DOMAIN) for vertex in network_dict.keys()}

    for i in range(burn_in):
        if verbose:
            print("[PROGRESS] Sample from UG burning in:", i, "/", burn_in)
        for subject in network_dict.keys():
            boundary_values = {
                'L_self': None,
                'L_neighbors': [],
                'A_self': None,
                'A_neighbors': [],
                'Y_neighbors': [],
            }

            if layer == "L":
                boundary_values['L_neighbors'] = [current_layer[neighbor] for neighbor in network_dict[subject]]

            if layer == "A":
                boundary_values['L_self'] = sample.loc[subject, 'L']
                boundary_values['L_neighbors'] = [sample.loc[neighbor, 'L'] for neighbor in network_dict[subject]]
                boundary_values['A_neighbors'] = [current_layer[neighbor] for neighbor in network_dict[subject]]

            if layer == 'Y':
                boundary_values['L_self'] = sample.loc[subject, 'L']
                boundary_values['L_neighbors'] = [sample.loc[neighbor, 'L'] for neighbor in network_dict[subject]]
                boundary_values['A_self'] = sample.loc[subject, 'A']
                boundary_values['A_neighbors'] = [sample.loc[neighbor, 'A'] for neighbor in network_dict[subject]]
                boundary_values['Y_neighbors'] = [current_layer[neighbor] for neighbor in network_dict[subject]]

            current_layer[subject] = sample_given_boundary(boundary_values)

    return current_layer # return the sampled data for THE SPECIFIED LAYER

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
        if values:
            if isinstance(values, list):
                weighted_sum += weights[key] * sum(values)
            else:
                weighted_sum += weights[key] * values
    
    p = expit(weighted_sum)
    return int(np.random.uniform() < p)

def U_dist():
    return np.random.normal(0, 1)

def sample_given_boundary_binary(boundary_values):
    ''' 
    Note: This can't be any random function. 
          See Lauritzen chain graph paper page 342.
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
        if values:
            if isinstance(values, list):
                weighted_sum += weights[key] * sum(values)
            else:
                weighted_sum += weights[key] * values

    p = expit(weighted_sum)
    return int(np.random.uniform() < p)

def main():
    
    # NETWORK_NAME = "HR_edges" 
    NETWORK_NAME = "HU_edges" 
    # NETWORK_NAME = "RO_edges"  
    burn_in = 500 
    
    # load real-life social network and process it into a dictionary
    network = pd.read_csv(f"./raw_data/{NETWORK_NAME}.csv")
    network_dict = {}
    for index, row in network.iterrows():
        network_dict.setdefault(row['node_1'], []).append(row['node_2'])
        network_dict.setdefault(row['node_2'], []).append(row['node_1'])
    
    # save the network_dict and a maximal 6-degree separated set 
    # (aka. a maximal 5-hop independent set) ind_set to intermediate_data folder
    with open(f"./intermediate_data/{NETWORK_NAME}/{NETWORK_NAME}_network.pkl", "wb") as file:
        pickle.dump(network_dict, file) 
        
    # find 20 maximal 5-hop independent sets and save the largest one
    for i in range(20):
        ind_set = maximal_n_hop_independent_set(network_dict, n=5, verbose=False) 
        if i == 0:
            max_ind_set = ind_set
        elif len(ind_set) > len(max_ind_set):
            max_ind_set = ind_set
    
    pd.DataFrame(list(max_ind_set), columns=["subject"]).to_csv(f"./intermediate_data/{NETWORK_NAME}/{NETWORK_NAME}_5_ind_set.csv", index=False)
    
    # sample from the UUU model
    UUU_sample = pd.DataFrame(index=network_dict.keys(), columns=['L', 'A', 'Y'])
    UUU_sample['L'] = sample_unedge_layer(network_dict=network_dict,
                                      sample=UUU_sample,
                                      layer='L',
                                      sample_given_boundary=sample_given_boundary_binary,
                                      verbose=True,
                                      burn_in=burn_in)
    UUU_sample['A'] = sample_unedge_layer(network_dict=network_dict,
                                      sample=UUU_sample,
                                      layer='A',
                                      sample_given_boundary=sample_given_boundary_binary,
                                      verbose=True,
                                      burn_in=burn_in)
    UUU_sample['Y'] = sample_unedge_layer(network_dict=network_dict,
                                      sample=UUU_sample,
                                      layer='Y',
                                      sample_given_boundary=sample_given_boundary_binary,
                                      verbose=True,
                                      burn_in=burn_in)   
    
    # save the UUU_sample to intermediate_data folder
    df = pd.DataFrame(UUU_sample)
    df.to_csv(f"./intermediate_data/{NETWORK_NAME}/{NETWORK_NAME}_UUU_sample.csv", index=True)
    
    # sample from the BBB model
    BBB_sample = pd.DataFrame(index=network_dict.keys(), columns=['L', 'A', 'Y'])
    BBB_sample['L'] = sample_biedge_layer(network_dict=network_dict, 
                                            sample=BBB_sample, 
                                            layer='L', 
                                            U_dist=U_dist, 
                                            f=f_binary)
    BBB_sample['A'] = sample_biedge_layer(network_dict=network_dict, 
                                            sample=BBB_sample, 
                                            layer='A', 
                                            U_dist=U_dist, 
                                            f=f_binary)
    BBB_sample['Y'] = sample_biedge_layer(network_dict=network_dict, 
                                            sample=BBB_sample, 
                                            layer='Y', 
                                            U_dist=U_dist, 
                                            f=f_binary)

    df = pd.DataFrame(BBB_sample)
    df.to_csv(f"./intermediate_data/{NETWORK_NAME}/{NETWORK_NAME}_BBB_sample.csv", index=True)

if __name__ == "__main__":
    main()