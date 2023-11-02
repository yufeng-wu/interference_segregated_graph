'''
This code:

1. samples a single realization of a given undirected graph
2. samples a single realization of a given bidirected graph
'''

from util import edges_to_graph
import numpy as np

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
        - the U's are realizations of the same distribution.
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
        V_value = f(U_values) + np.random.normal()
        
        sample[vertex] = V_value

    return sample

def sample_from_UG(graph):
    pass

if __name__ == '__main__':
    pass