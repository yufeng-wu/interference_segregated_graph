'''
This code:

1. samples a single realization of a given undirected graph
2. samples a single realization of a given bidirected graph
'''

from util import edges_to_graph, find_cliques
import numpy as np
import random

def sample_from_BG(edges, U_dist, f):
    '''
    Given a graph structure, sample a single realization by assuming that there
    is an unobserved confounder, U, between each pair of connected vertices. 

    The value of each U is sampled from U_dist.

    Params:
        - edges list(tuple(int, int)): a list of edges; each tuple denotes an 
            edge between two vertices in the graph.
        - U_dist (callable): a function that returns a sample from the distribution
            of U when called.
        - f (callable): the function V = f(U_1, U_2, ..., U_n) + noise
            that takes a list of U values and returns a value V.  
    
    Return:
        - sample (dict: int -> float): sample[vertex_i] = the value of vertex_i

    Assumptions:
        - the U's are realizations of the same distribution. (We have to assume this! otherwise misspecification.)
        - the V's are generated from the same parametric form.
    '''

    # Initialize a dictionary to hold the U values "at" each edge
    edge_to_U = {}

    for edge in edges:
        # sample a U value from U_dist
        U_value = U_dist()
        # save that in a map: each edge -> U value "associated with" that edge.
        edge_to_U[edge] = U_value

    # Initialize a dictionary to hold the V values for each vertex
    sample = {}

    graph = edges_to_graph(edges)

    # For each vertex, find all edges that involve this vertex
    for vertex in graph.keys():
        U_values = []
        for neighbor in graph[vertex]:
            try:
                U_value = edge_to_U[(vertex, neighbor)]
            except: # try both orders
                U_value = edge_to_U[(neighbor, vertex)]
            U_values.append(U_value)
        
        # Calculate the value of V using the function f and add normally distributed noise
        V_value = f(U_values)
        
        sample[vertex] = V_value

    return sample

def sample_from_UG(graph, prob_v_given_neighbors, verbose=False, burn_in=1000):
    '''
    Maybe we should just specify a cond_proba_v_given_all_else() function with set parameters, as long as it's consistent w/ the UG!
    
    parametric_form:
        -   P(V_i = v_i | -V_i)
          = P(V_i = v_i | nb(V_i)) 
          = expit(a0 + a1 * \sum_{V_j s.t. V_j \in nb(V_i)} V_j)
    '''
    V_DOMAIN = [1, 0]

    # Initialize vertices with random values chosen from V_DOMAIN
    v_values = {vertex: random.choice(V_DOMAIN) for vertex in graph.keys()}

    # Gibbs sampling
    for i in range(burn_in):
        if verbose and (i % 100 == 0):
            print("[PROGRESS] Sample from UG burning in:", i, "/", burn_in)
        for v in graph.keys():
            v_neighbors = graph[v] # returns a list
            v_neighbor_values = [v_values[neighbor] for neighbor in v_neighbors]

            # P(V = 1 | nb(V))
            p = prob_v_given_neighbors(v_neighbor_values)

            # always interpret of p as P(V=1 | Y=y).
            if v_values[v] == 0:
                p = 1 - p

            # sample a new value for the current node based on conditional proba
            v_values[v] = np.random.choice(V_DOMAIN, size=1, p=np.array([p, 1-p]))[0]

    return v_values


if __name__ == '__main__':
    pass