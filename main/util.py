from collections import defaultdict
import networkx as nx
import random
import pandas as pd

def kth_order_neighborhood(network, node, k):
    if k == 0:
        return {node}
    
    neighbors = set([node])
    visited = set([node])

    for _ in range(k):
        temp_neighbors = set()
        for neighbor in neighbors:
            temp_neighbors.update(set(network[neighbor]))
        temp_neighbors -= visited
        neighbors = temp_neighbors
        visited.update(temp_neighbors)

    return neighbors

def create_random_network(n, min_neighbors, max_neighbors):
    # Ensure that the sum of the degree sequence is even
    while True:
        degree_sequence = [random.randint(min_neighbors, max_neighbors) for _ in range(n)]
        if sum(degree_sequence) % 2 == 0:
            break

    # Create a graph using the configuration model
    g = nx.configuration_model(degree_sequence)

    # Remove parallel edges and self-loops
    g = nx.Graph(g)
    g.remove_edges_from(nx.selfloop_edges(g))

    return nx.to_dict_of_lists(g)

def random_network_adjacency_matrix(n, min_neighbors, max_neighbors):

    # Ensure that the sum of the degree sequence is even
    while True:
        degree_sequence = [random.randint(min_neighbors, max_neighbors) for _ in range(n)]
        if sum(degree_sequence) % 2 == 0:
            break

    # Create a graph using the configuration model
    g = nx.configuration_model(degree_sequence)

    # Remove parallel edges and self-loops
    g = nx.Graph(g)
    g.remove_edges_from(nx.selfloop_edges(g))

    # Convert to adjacency matrix
    adj_matrix = nx.adjacency_matrix(g).toarray()

    return adj_matrix

def create_cycle_graph(n):
    '''
    Create a cycle graph with n vertices.
    '''
    graph = {}
    for i in range(n):
        # Connect each node to its adjacent nodes, wrapping around at the ends
        graph[i] = [(i - 1) % n, (i + 1) % n]
    return graph

def edges_to_graph(edges):
    '''
    Convert edge representation of a graph into a vertex representation.

    Params:
        - edges list(tuple(int, int)): a list of edges; each tuple denotes an 
            edge between two vertices in the graph.
    
    Return:
        - graph (dict): graph[v_i] = a list of vertices connected with v_i
    '''

    graph = defaultdict(list)

    for i in range(len(edges)):
        v1, v2 = edges[i]
        graph[v1].append(v2)
        graph[v2].append(v1)
    
    return graph

def graph_to_edges(graph):
    '''
    Convert a vertex representation of a graph into an edge representation.

    Params:
        - graph (dict): graph[v_i] = a list of vertices connected with v_i

    Return:
        - edges list(tuple(int, int)): a list of edges; each tuple denotes an 
            edge between two vertices in the graph.
    '''
    
    edges = set() # use a set to avoid duplicates
    for vertex, neighbors in graph.items():
        for neighbor in neighbors:
            # Add a tuple of vertices to the set, 
            # with the smaller vertex first to avoid duplicates
            edge = tuple(sorted((vertex, neighbor)))
            edges.add(edge)
            
    # Convert to a list before returning
    return list(edges)
