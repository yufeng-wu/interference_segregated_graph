from util import create_cycle_graph, graph_to_edges
from data_generator import sample_from_BG, sample_from_UG
import numpy as np

# Define the U distribution sampling function
def U_dist():
    return np.random.normal(0, 1)

# Define the function f that calculates V based on U values
def f(U_values):
    noise = np.random.normal(0, 0.1)
    return sum(U_values) + noise

if __name__ == '__main__':
    graph = create_cycle_graph(1000)
    BG_sample = sample_from_BG(edges=graph_to_edges(graph), U_dist=U_dist, f=f)
    print(BG_sample)