import random

def get_vertices_within_n_layers(graph, start_vertex, n):
    """
    Get all vertices within 'n' layers from 'start_vertex'.
    Only consider vertices that are in 'active_nodes'.
    """

    visited = set([start_vertex])
    frontier = set([start_vertex])

    for _ in range(n):
        next_frontier = set()
        for vertex in frontier:
            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier

    return visited

# Note: in our paper, we called it an n+1 degree separated set. 
def maximal_n_hop_independent_set(network_dict, n, verbose=False):
    """
    Find an n-apart independent set in the graph.
    """
    independent_set = set()
    active_nodes = set(network_dict.keys())

    while active_nodes:
        if verbose and len(active_nodes) % 1000 < 2:
            print("[PROGRESS] maximal_n_apart_independent_set need to process", 
                  len(active_nodes), "more nodes.")
        
        # sample a random vertex to be the next element in the independent set
        current = random.sample(active_nodes, 1)[0]
        active_nodes.remove(current)
        independent_set.add(current)

        # get all vertices within n layers and mark them as inactive
        to_deactivate = get_vertices_within_n_layers(network_dict, current, n)
        active_nodes.difference_update(to_deactivate)

    return independent_set