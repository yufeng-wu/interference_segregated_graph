'''
Python code to ...

Author: Yufeng Wu
Date: Oct.28th, 2023
Williams College Computer Science

References: 
- Maximal Independent Set algorithm implementation: https://www.geeksforgeeks.org/maximal-independent-set-in-an-undirected-graph/#
'''

from util import create_random_network

# from collections import deque
# import random

# def get_neighbors(graph, current_v, layer):
# 	'''
# 	Helper function to find the neighbors layer layers from current_v
# 	e.g. in A--B--C--D, B is the 1-layer neighbor from A and D is the 3-layer neighbor from A
# 	'''
# 	visited = set()
# 	queue = deque([(current_v, 0)])  # (vertex, layer)
# 	visited.add(current_v)
	
# 	while queue:
# 		vertex, depth = queue.popleft()

# 		# Short-circuit to save runtime
# 		if depth == layer:
# 			# remove current_v before returning so that we only return the neighbors
# 			visited.remove(current_v)
# 			return visited
        
# 		for neighbor in graph[vertex]:
# 			if neighbor not in visited and neighbor in graph.keys():
# 				visited.add(neighbor)
# 				queue.append((neighbor, depth + 1))

# 	# remove current_v before returning so that we only return the neighbors
# 	visited.remove(current_v)
# 	return visited

# def maximal_n_apart_independent_set(graph, n, available_vertices, 
# 									approx, is_cycle_graph=False):
# 	'''
# 	Find the n-apart maximal indepdent set from a given graph using recursion.

# 	The original definition of indepdent set is 1-apart, meaning that the vertices
# 	in the independent set should be separated by at least one vertex. 
# 	n-apart extends this definition and finds the maximal indepdent set where 
# 	every vertex in the set are at least n vertices apart from any other vertex 
# 	in the set in the original graph.

# 	Parameters:
# 	 - graph (dict): a dictionary where graph[Vertex V] 
# 	 				 gives Neighbors of Vertex V as a list. 
# 					 Vertices are represented by integers.
# 	 - n (int): a non-negative int, how many vertices should be apart from 
# 	 			each pair of vertices in the maximal independent set 
# 	 - available_vertices (set): the set of vertices that are still avaiable to 
# 	 							 choose from to form the indepdent set. 
# 								 This field should be set to set(graph.keys()) 
# 								 when this func is called for the first time.

# 	Time complexity: O(2^N) where N is the number of vertices in the graph.
# 	'''

# 	# Short-cut to improve computational efficiency
# 	if is_cycle_graph:
# 		res = []
# 		for i in range(0, len(graph)-n):
# 			if i % (n + 1) == 0:
# 				res.append(i)
# 		return res

# 	def case1():
# 		# Case 1: include current_v in our maximal independent set
# 		# Remove all layer-n neighbors of current_v from the available_vertices set
# 		neighbors = get_neighbors(graph, current_v, n)
# 		available_vertices_res1 = available_vertices.copy()
# 		for neighbor in neighbors:
# 			available_vertices_res1.discard(neighbor)
# 		result_1 = [current_v] + maximal_n_apart_independent_set(graph, n, available_vertices_res1, approx)
# 		return result_1
	
# 	def case2():
# 		# Case 2: not include current_v in our maximal independent set
# 		available_vertices_res2 = available_vertices.copy()
# 		result_2 = maximal_n_apart_independent_set(graph, n, available_vertices_res2, approx)
# 		return result_2

# 	# Base Case
# 	if len(available_vertices) == 0:
# 		return []
	
# 	# Recursive Cases
# 	current_v = available_vertices.pop() # pop a random vertex out of the available set, order not guaranteed

# 	if approx:
# 		options = ["both", "include", "exclude"]
# 		v_count = len(graph.keys())
# 		option = random.choices(options, weights=[v_count, v_count**(1.6), v_count**(1.6)])[0]
		
# 		if option == "both":
# 			res1 = case1()
# 			res2 = case2()
# 		elif option == "include":
# 			res1 = case1()
# 			res2 = []
# 		else:
# 			res1 = []
# 			res2 = case2()
# 	else:
# 		res1 = case1()
# 		res2 = case2()

# 	# Our final result is the one that has more elements in it, return it
# 	if len(res1) > len(res2):
# 		return res1
# 	return res2

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

def maximal_n_apart_independent_set(graph, n, verbose=False):
    """
    Find an n-apart independent set in the graph.
    """
    independent_set = set()
    active_nodes = set(graph.keys())

    while active_nodes:
        if verbose and len(active_nodes) % 1000 < 2:
            print("[PROGRESS] maximal_n_apart_independent_set need to process", len(active_nodes), "more nodes.") 
        #current_vertex = random.choice(tuple(active_nodes))  # Choose a vertex from active nodes
        current_vertex = active_nodes.pop()
		# active_nodes.remove(current_vertex)
        independent_set.add(current_vertex)

        # Get all vertices within n layers and mark them as inactive
        to_deactivate = get_vertices_within_n_layers(graph, current_vertex, n)
        active_nodes.difference_update(to_deactivate)

    return independent_set

if __name__ == '__main__':
    g = create_random_network(n=100, min_neighbors=1, max_neighbors=6)

    n = 5
    result = maximal_n_apart_independent_set(g, n, True)
    # print("Maximal ", n, "-apart independent set: ", result)
    # print("Length: ", len(result))
    # print("Best possible length: ", len(cycle_UG.keys())//6)
    print(result)


# big picture:

# this weekend: 
#   prove the L, A, and Y layers (maybe no both - and <-> edges? cuz it's not defined?)
#   do some code estimation

# next big step:
#   think about estimation
#   proofs of estimation algorithms

# greedy algorithm for finding ind set

# read data application