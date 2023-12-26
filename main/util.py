from collections import defaultdict

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
    for vertex, neighbours in graph.items():
        for neighbour in neighbours:
            # Add a tuple of vertices to the set, 
            # with the smaller vertex first to avoid duplicates
            edge = tuple(sorted((vertex, neighbour)))
            edges.add(edge)
            
    # Convert to a list before returning
    return list(edges)

def bron_kerbosch_all_cliques(R, P, X, graph, cliques):
    # Report the current set R as a clique
    if R:
        cliques.append(R)
    for v in list(P):
        new_R = R.union({v})
        new_P = P.intersection(graph[v])
        new_X = X.intersection(graph[v])
        bron_kerbosch_all_cliques(new_R, new_P, new_X, graph, cliques)
        P.remove(v)
        X.add(v)

def find_cliques(graph):
    '''
    Find all cliques in the given graph.
    '''
    cliques = []
    P = set(graph.keys())
    R = set()
    X = set()
    bron_kerbosch_all_cliques(R, P, X, graph, cliques)
    return cliques

def print_cliques(cliques):
    cliques_by_size = defaultdict(list)
    for clique in cliques:
        cliques_by_size[len(clique)].append(clique)
    for size in cliques_by_size.keys():
        print(len(cliques_by_size[size]), "cliques of size", size, ": ")
        print(cliques_by_size[size])