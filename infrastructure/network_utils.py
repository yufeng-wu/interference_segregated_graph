from collections import defaultdict
import networkx as nx
import random
import numpy as np

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

def create_random_network(n, avg_degree, max_degree):
    g = nx.fast_gnp_random_graph(n, avg_degree / (n - 1))
    for node in g.nodes:
        while g.degree[node] > max_degree:
            neighbors = list(g.neighbors(node))
            g.remove_edge(node, random.choice(neighbors))
    adj_matrix = nx.adjacency_matrix(g).toarray()
    return nx.to_dict_of_lists(g), adj_matrix

# def create_random_network(n, min_neighbors, max_neighbors):
#     while True:
#         degree_sequence = [random.randint(min_neighbors, max_neighbors) for _ in range(n)]
#         if sum(degree_sequence) % 2 == 0:
#             break
#     g = nx.configuration_model(degree_sequence)
#     g = nx.Graph(g)
#     g.remove_edges_from(nx.selfloop_edges(g))
#     adj_matrix = nx.adjacency_matrix(g).toarray()

#     return nx.to_dict_of_lists(g), adj_matrix

def create_cycle_graph(n):
    '''
    Create a cycle graph with n vertices.
    '''
    graph = {}
    for i in range(n):
        # Connect each node to its adjacent nodes, wrapping around at the ends
        graph[i] = [(i - 1) % n, (i + 1) % n]
    return graph

def ring_adjacency_matrix(num_units):
    network = np.zeros((num_units, num_units))
    for i in range(num_units-1):
        network[i, i+1] = 1
        network[i+1, i] = 1

    network[0, num_units-1] = 1
    network[num_units-1, 0] = 1

    return network

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
