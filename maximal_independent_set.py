'''
Python code to ...

Author: Yufeng Wu
Date: Oct.28th, 2023
Williams College Computer Science

References: 
- Maximal Independent Set algorithm implementation: https://www.geeksforgeeks.org/maximal-independent-set-in-an-undirected-graph/#
'''

from collections import deque

def get_neighbors(graph, current_v, layer):
	'''
	Helper function to find the neighbors layer layers from current_v
	e.g. in A--B--C--D, B is the 1-layer neighbor from A and D is the 3-layer neighbor from A
	'''
	visited = set()
	queue = deque([(current_v, 0)])  # (vertex, layer)
	visited.add(current_v)
	
	while queue:
		vertex, depth = queue.popleft()

		# Short-circuit to save runtime
		if depth == layer:
			# remove current_v before returning so that we only return the neighbors
			visited.remove(current_v)
			return visited
        
		for neighbor in graph[vertex]:
			if neighbor not in visited and neighbor in graph.keys():
				visited.add(neighbor)
				queue.append((neighbor, depth + 1))

	# remove current_v before returning so that we only return the neighbors
	visited.remove(current_v)
	return visited

def maximal_n_apart_independent_set(graph, n, available_vertices):
	'''
	Find the n-apart maximal indepdent set from a given graph using recursion.

	The original definition of indepdent set is 1-apart, meaning that the vertices
	in the independent set should be separated by at least one vertex. 
	n-apart extends this definition and finds the maximal indepdent set where 
	every vertex in the set are at least n vertices apart from any other vertex 
	in the set in the original graph.

	Parameters:
	 - graph (dict): a dictionary where graph[Vertex V] 
	 				 gives Neighbors of Vertex V as a list. 
					 Vertices are represented by integers.
	 - n (int): a non-negative int, how many vertices should be apart from 
	 			each pair of vertices in the maximal independent set 
	 - available_vertices (set): the set of vertices that are still avaiable to 
	 							 choose from to form the indepdent set. 
								 This field should be set to graph.keys() by the 
								 caller when this func is called for the first time.

	Time complexity: O(2^N) where N is the number of vertices in the graph.
	'''

	# Base Case
	if len(available_vertices) == 0:
		return []
	
	# Recursive Cases
	current_v = available_vertices.pop() # pop a random vertex out of the available set, order not guaranteed
	
	# Case 1: include current_v in our maximal independent set
	# Remove all layer-n neighbors of current_v from the available_vertices set
	neighbors = get_neighbors(graph, current_v, n)
	available_vertices_res1 = available_vertices.copy()
	for neighbor in neighbors:
		available_vertices_res1.discard(neighbor)
	res1 = [current_v] + maximal_n_apart_independent_set(graph, n, available_vertices_res1)

	# Case 2: not include current_v in our maximal independent set
	available_vertices_res2 = available_vertices.copy()
	res2 = maximal_n_apart_independent_set(graph, n, available_vertices_res2)

	# Our final result is the one that has more elements in it, return it
	if len(res1) > len(res2):
		return res1
	return res2
	
# Defines edges
# E = [(1, 2),
# 	(2, 3),
# 	(3, 4),
# 	(4, 5),
# 	(5, 6),
# 	(6, 7),
# 	(7, 8),
# 	(8, 9),
# 	(9, 1)]

E = [(1, 2),
	(1, 3),
	(2, 4),
	(5, 6),
	(6, 7),
	(4, 8)]

graph = dict([])

# Constructs Graph as a dictionary of the following format:
# graph[Vertex V] = list[Neighbors of Vertex V]
for i in range(len(E)):
	v1, v2 = E[i]
	
	if(v1 not in graph):
		graph[v1] = []
	if(v2 not in graph):
		graph[v2] = []
	
	graph[v1].append(v2)
	graph[v2].append(v1)

graph = {
        0: [30, 1],
        1: [0, 2],
        2: [1, 3],
        3: [2, 4],
        4: [3, 5],
        5: [4, 6],
        6: [5, 7],
        7: [6, 8],
        8: [7, 9],
        9: [8, 10],
        10: [9, 11],
        11: [10, 12],
        12: [11, 13],
        13: [12, 14],
        14: [13, 15],
        15: [14, 16],
        16: [15, 17],
        17: [16, 18],
        18: [17, 19],
        19: [18, 20],
        20: [19, 21],
        21: [20, 22],
        22: [21, 23],
        23: [22, 24],
        24: [23, 25],
        25: [24, 26],
        26: [25, 27],
        27: [26, 28],
        28: [27, 29],
        29: [28, 30],
        30: [29, 0]
    }

n = 5
res = maximal_n_apart_independent_set(graph, n, set(graph.keys()))
print("Maximal ", n, "-apart independent set: ", res)
