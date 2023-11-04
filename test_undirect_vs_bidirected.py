from util import create_cycle_graph, graph_to_edges
from data_generator import sample_from_BG, sample_from_UG
from maximal_independent_set import maximal_n_apart_independent_set
from scipy.special import expit
import numpy as np

def U_dist():
    '''
    Define the distribution: U ~ U_dist.
    '''
    return np.random.normal(0, 1)

def f(U_values):
    '''
    Define the function f that calculates V based on its neighboring U values
    and returns a binary value.
    '''
    noise = np.random.normal(0, 0.1)
    linear_sum = sum(U_values) + noise
    prob = expit(linear_sum)  # Sigmoid function to get a value between 0 and 1
    return np.random.binomial(1, prob)  # Sample from a Bernoulli distribution

    # '''
    # Define the function f that calculates V based on its neighboring U values
    # '''
    # noise = np.random.normal(0, 0.1)
    # return sum(U_values) + noise

def prob_v_given_neighbors(V_neighbors):
    """
    Define the parametric form for the conditional probability P(V_i = 1 | -V_i)
    using only the V_neighbors as input. V_i is a binary variable that is either
    0 or 1. The parameters a0 and a1 are hard-coded inside the function.

    Params:
        - V_neighbors: array-like, containing the values of V's neighbors
    
    Return:
        - a float that represents the conditional probability
    """
    # Parameters can be hard-coded or defined as constants elsewhere
    a0 = 0.5  
    a1 = 0.8
    return expit(a0 + a1 * np.sum(V_neighbors))

if __name__ == '__main__':
    # STEP 1: Greate graph
    graph = create_cycle_graph(1000)

    # STEP 2: Generate data assuming that the edges are undirected 
    sample_UG = sample_from_UG(graph=graph, 
                               prob_v_given_neighbors=prob_v_given_neighbors,
                               verbose=True,
                               burn_in=1000)
    # print(sample_UG)

    # STEP 3: Generate data assuming that the edges are bidirected 
    sample_BG = sample_from_BG(edges=graph_to_edges(graph), 
                               U_dist=U_dist, 
                               f=f)
    # print(sample_BG)

    # STEP 4: Get independent set from graph
    ind_set = maximal_n_apart_independent_set(graph, 
                                              n=5, 
                                              available_vertices=set(graph.keys()),
                                              approx=True)
    print(ind_set)

    # STEP 5: Likelihood Ratio Test


    # Map from clique size to clique potential function
    # clique_potentials = {
    #     1 : lambda a: a + 10,
    #     2 : lambda a, b: a + 2*b + 5
    # }
    


    

