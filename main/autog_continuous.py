'''
This file implements the Auto-G Computation method proposed by 
Eric J. Tchetgen Tchetgen, Isabel R. Fulcher, and Ilya Shpitser.
'''

import util
import data_generator as dg
import generate_network_sample as gns
import maximal_independent_set
import numpy as np
import pandas as pd
import random
import os
import pickle
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.interpolate import interp1d
import math


''' DATA PREPARATION '''

def df_for_estimation(network, ind_set, sample):
    '''
    Creates dataframe for causal effect estimation. This implementation closely
    follows the Auto-G Computation paper by Eric J. Tchetgen Tchetgen, 
    Isabel R. Fulcher, and Ilya Shpitser.

    Inputs:
        - network
        - ind_set: a maximal 1-apart independent set obtained from the network
        - sample: a single realization (L, A, Y) of the network where L, A, Y 
                  are vectors of the shape (1, size of network).
    
    Return:
        A pd.DataFrame object that with the following entries for each element 
        of the ind_set:
            'i': id of the subject
            'y_i': the value of Y_i in the network realization
            'a_i': the value of A_i in the network realization
            'l_i': the value of L_i in the network realization
            'l_j_list': [L_j for j in neighbors of i]
            'y_j_list': [Y_j for j in neighbors of i]
            'adjusted_sum_l_j': the sum of elements in l_j_list where each term 
                is first multiplied by a normalizing constant 
                (= 1 / count of i's neighbors) and then summed together.
            'adjusted_sum_a_j': the sum of elements in a_j_list where each term 
                is first multiplied by a normalizing constant 
                (= 1 / count of i's neighbors) and then summed together.
            'w_ij_y': 1 / count of i's neighbors
            'w_ij_l': 1 / count of i's neighbors
    '''
    
    data_list = []
    
    for i in ind_set['subject']:
        # extract the relevant rows from sample based on index i. 
        row = sample.loc[i]
        
        # assuming 'Y', 'A', and 'L' are column names in 'sample'
        l_i = row['L']
        a_i = row['A']
        y_i = row['Y']
        
        # get the neighbors of i
        N_i = util.kth_order_neighborhood(network, i, 1)
        normalizing_weight = 1 #/ len(N_i) #TODO: think about this weight term
        
        # Get a list with sample.loc[j]['Y'] for each j in N_i
        l_j = [sample.loc[j]['L'] for j in N_i]
        a_j = [sample.loc[j]['A'] for j in N_i]
        y_j = [sample.loc[j]['Y'] for j in N_i]
        
        # sum over l_j and then mutiply with normalizing_weight
        sum_l_j = sum(l_j)
        adjusted_sum_l_j = sum_l_j * normalizing_weight

        sum_a_j = sum(a_j)
        adjusted_sum_a_j = sum_a_j * normalizing_weight
        
        w_ij_y = normalizing_weight
        w_ij_l = normalizing_weight
        
        data_list.append({
            'i' : i,
            'y_i': y_i,
            'a_i': a_i,
            'l_i': l_i,
            'l_j_list': l_j,
            'y_j_list': y_j,  
            'adjusted_sum_l_j': adjusted_sum_l_j,
            'adjusted_sum_a_j': adjusted_sum_a_j,
            'w_ij_y': w_ij_y,
            'w_ij_l': w_ij_l 
        })
    
    df = pd.DataFrame(data_list) 
    return df     


''' AUTO GAUSSIAN MODEL '''

''' 
Functions defined for the L layer
Parameters: sigma2_l, mu_l, weight_l
'''
def H_i(l_i, params):
    sigma2_l, mu_l = params[:-1]
    return -(1/(2*sigma2_l)) * (l_i - 2*mu_l)

def W_i(data_i, params, l_i=None):
    # we allow the caller to pass in l_i separate from data_i
    # so that when evaluating the denominator it is easier.
    if l_i == None:
        l_i = data_i['l_i']
    H_i_output = H_i(l_i=l_i, params=params)
    
    second_term = 0
    weight_l = params[-1]
    for l_j in data_i['l_j_list']:
        second_term += weight_l * l_i * l_j * data_i['w_ij_l']
    
    return l_i * H_i_output + second_term

def f_L_i_given_stuff(data_i, params): 
    '''
    This is the distribution f(L_i | L_-i) with the parametric form specified 
    by the auto-g paper. The distribution has paramters "params."

    Inputs:
        - data_i: data associated with i, which is an element in an 1-apart 
                  maximal independent set of the network. It contains 'l_i',
                  'l_j_list', and 'w_ij_l'.
        - params: parameters of this distribution: sigma2_l, mu_l, weight_l

    The function returns the probability that L_i = l_i given l_-i (as 
    represented with 'l_j_list').
    '''
    numerator = math.exp(W_i(data_i, params))

    # define a lambda function for what we want to integrate over
    integrand = lambda l_i: math.exp(W_i(data_i, params, l_i))

    # perform numerical integration over the range of y_i
    range_min, range_max = -np.inf, np.inf 
    denominator, absolute_error = quad(integrand, range_min, range_max)
    
    return numerator / denominator

''' 
Functions defined for the Y layer
Parameters: beta_0, beta_1, beta_2, beta_3, beta_4, sigma2_y, theta
'''

def G_i(y_i, a_i, l_i, adjusted_sum_a_j, adjusted_sum_l_j, params):
    beta_0, beta_1, beta_2, beta_3, beta_4, sigma2_y = params[:-1]
    mu_y_i = beta_0 + beta_1*a_i + beta_2*l_i + beta_3*adjusted_sum_a_j + beta_4*adjusted_sum_l_j
    return -(1/(2*sigma2_y)) * (y_i - 2*mu_y_i)

def theta_ij(w_ij_y, params):
    theta = params[-1]
    return w_ij_y * theta

def U_i(data_i, params, y_i=None):
    # we allow the caller to pass in y_i separate from data_i
    # so that when evaluating the denominator it is easier.
    if y_i == None:
        y_i = data_i['y_i']
    G_i_output = G_i(y_i=y_i, 
                      a_i=data_i['a_i'], 
                      l_i=data_i['l_i'], 
                      adjusted_sum_a_j=data_i['adjusted_sum_a_j'], 
                      adjusted_sum_l_j=data_i['adjusted_sum_l_j'], 
                      params=params)
    
    second_term = 0
    for y_j in data_i['y_j_list']:
        second_term += y_i * y_j * theta_ij(data_i['w_ij_y'], params)
    
    return y_i * G_i_output + second_term

def f_Y_i_given_stuff(data_i, params):
    '''
    This is the distribution f(Y_i | Y_-i = y_-i, a, l) with 
    the parametric form specified by the auto-g paper. 
    The distribution has paramters "params."

    Inputs:
        - data_i: data associated with i, which is an element in an 1-apart 
                  maximal independent set of the network. It contains 'y_i', 
                  'l_i', 'a_i', 'y_j_list', 'adjusted_sum_a_j', 
                  'adjusted_sum_l_j', and 'w_ij_y'.
        - params: parameters of this distribution: 
                  beta_0, beta_1, beta_2, beta_3, beta_4, sigma2_y, theta

    The function returns the probability that Y_i = y_i given y_-i, a, l 
    (which can be simplified to given the boundary of Y_i in the CG model).
    '''
    numerator = math.exp(U_i(data_i, params))
    
    # define a lambda function for what we want to integrate over
    integrand = lambda y_i: math.exp(U_i(data_i, params, y_i))
    
    # perform numerical integration over the range of y_i
    range_min, range_max = -100, 100#-np.inf, np.inf 
    denominator, absolute_error = quad(integrand, range_min, range_max)
    
    return numerator / denominator


''' ESTIMATE PARAMETERS OF THE TWO DISTRIBUTIONS '''

def nlcl(params, f, est_df):
    '''
    Negative Log Coding-type Likelihood.

    Inputs:
        - params: parameters of the distribution f
        - f: a function, either f_Y_i_given_stuff or f_L_i_given_stuff
        - est_df: a pd.DataFrame object obtained from df_for_estimation()
    
    Return:
        - The Negative Log Coding-type Likelihood to generate a sample of 
          est_df from distribution f with parameters param.
    '''
    log_likelihoods = est_df.apply(lambda row: np.log(f(row, params)), axis=1)
    return -np.sum(log_likelihoods)

def optimize_params(nll_function, initial_params, f, est_df):
    '''
    Function to estimate the parameters of the distribution f using a sample 
    of data est_df.

    Inputs:
        - nll_function: user specified negative log likelihood function.
        - initial_params: a list of numbers representing the inital guess
            of the parameters.
        - f: a function, which is the distribution that we need to estimate
            parameters for. Should be f_Y_i_given_stuff or f_L_i_given_stuff.
        - est_df: a pd.DataFrame object obtained from df_for_estimation()
    
    Return:
        - the estimated parameters.
    '''
    result = minimize(nll_function, x0=initial_params, args=(f, est_df))
    return result.x


''' ESTIMATE NETWORK CAUSAL EFFECTS '''

def approximate_cdf(conditional_pdf, conditionals, x_range, num_points=1000):
    """Numerically approximate the CDF of a given conditional PDF over a specified range."""
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    cdf_values = np.zeros(num_points)
    
    for i, x in enumerate(x_values):
        def integrand(x):
            return conditional_pdf(x, **conditionals)
        
        # Integrate the conditional PDF up to x to approximate the CDF at x
        (cdf_values[i], _) = quad(integrand, x_range[0], x)
    
    return x_values, cdf_values

def draw(conditional_pdf, conditionals, x_range=[-100, 100], num_samples=1):
    """Draw samples from a custom distribution defined by a conditional PDF within a given range."""

    x_values, cdf_values = approximate_cdf(conditional_pdf, conditionals, x_range)
    
    # Create an interpolation function for the inverse CDF
    inverse_cdf = interp1d(cdf_values, x_values, fill_value="extrapolate")
    
    # Draw uniform samples to feed into the inverse CDF
    uniform_samples = np.random.rand(num_samples)
    
    # Get samples from the custom distribution
    custom_samples = inverse_cdf(uniform_samples)
    
    return custom_samples

def gibbs_sampler_1(n_samples, burn_in, network, f_Yi, f_Li, verbose, A_val, mode='true'):
    '''
    Implementation of the Gibbs Sampler I algorithm on p10 of the auto-g paper.

    Inputs:
        - n_samples: number of network realizations to generate
        - burn_in: burn in period
        - network
        - f_Yi: a pdf that we can draw value from.
                Could be a wrapper function of f_Y_i_given_stuff with 
                estimated parameters.
        - f_Li: a pdf that we can draw value from.
                Could be a wrapper function of f_L_i_given_stuff with 
                estimated parameters.
        - verbose: True or False
        - A_val: the value of A_i that we set / intervene for each A_i.
                 should be either 0 or 1 as we are working with binary 
                 treatments.
    
    Return:
        - n_samples samples of realization of the network after burn_in period.
    '''
    
    assert mode in ['true', 'estimate']

    samples = []
    N = len(network) # number of subjects in the network
    
    # produce initial random guess
    sample = pd.DataFrame(index=network.keys(), columns=['L', 'A', 'Y'])

    # generate initial values for variables 
    sample['L'] = {vertex: np.random.normal(0, 1) for vertex in network.keys()}
    sample['A'] = {vertex: A_val for vertex in network.keys()}
    sample['Y'] = {vertex: np.random.normal(0, 1) for vertex in network.keys()}

    for m in range(n_samples + burn_in):
        if verbose and m % 1000 == 0:
            print("progress: ", m / (n_samples + burn_in))
            
        i = (m % N) # the paper has "+1" but i don't +1 because the index of subjects in my network starts from 0
        # draw L_i ~ f(L_i | L_-i)
        boundary_values_L = {
            'L_neighbors': [sample.loc[neighbor, 'L'] for neighbor in network[i]]
        }

        if mode == 'true':
            new_Li = f_Li(boundary_values_L)
        else:
            new_Li = draw(conditional_pdf=f_Li, conditionals=boundary_values_L, num_samples=1)[0]
        
        # draw Y_i ~ f(Y_i | Y_-i)
        boundary_values_Y = {
            'L_self': sample.loc[i, 'L'],
            'L_neighbors': [sample.loc[neighbor, 'L'] for neighbor in network[i]],
            'A_self': sample.loc[i, 'A'],
            'A_neighbors': [sample.loc[neighbor, 'A'] for neighbor in network[i]],
            'Y_neighbors': [sample.loc[neighbor, 'Y'] for neighbor in network[i]],
        }

        if mode == 'true':
            new_Yi = f_Yi(boundary_values_Y)
        else:
            new_Yi = draw(conditional_pdf=f_Yi, conditionals=boundary_values_Y, num_samples=1)[0]

        sample.loc[i, 'L'] = new_Li
        sample.loc[i, 'Y'] = new_Yi

        if m >= burn_in:
            samples.append(sample.copy())

    return samples

def beta_i_a(samples, i, select_for_every):
    '''
    Compute beta_i(a) as defined in p10 of the auto-g paper.

    Inputs:
        - samples: samples from gibbs_sampler_1.
        - i: id of the subject.
        - select_for_every: compute the causal effect using samples for 
            every select_for_every iterations to "thin" autocorrelation.
    
    Return:
        - beta_i(a)
    '''
    selected_Y_values = []
    
    # iterate through samples, selecting every nth sample where n is select_for_every
    for sample_idx in range(0, len(samples), select_for_every):
        selected_sample = samples[sample_idx]
        selected_Y_values.append(selected_sample.loc[i, 'Y'])
        
    average_Y = float(sum(selected_Y_values) / len(selected_Y_values))
    return average_Y


def main():

    NUM_OF_SUBJECTS = 1000
    # generate a small network n
    #network = util.create_random_network(n=NUM_OF_SUBJECTS, min_neighbors=1, max_neighbors=6)

    '''Evaluate True Network Causal Effects '''

    # evaluate true causal effect using true f_Yi, true f_Li, n, and A_val (via beta_i_a and gibbs sampler 1)
    # true_f_Yi = dg.sample_given_boundary_continuous
    # true_f_Li = dg.sample_given_boundary_continuous

    # Y_A1 = gibbs_sampler_1(n_samples=10000, burn_in=10000, network=network, 
    #                        f_Yi=true_f_Yi, f_Li=true_f_Li, verbose=True, 
    #                        A_val=1, mode='true')
    
    # beta_alpha = np.mean([beta_i_a(Y_A1, i, 3) for i in range(0, NUM_OF_SUBJECTS)])
    # print("beta(alpha) =", beta_alpha)


    '''Estimate Network Causal Effects Via Auto-G '''

    # edge_types = gns.generate_edge_types("UUU")

    # # Sample a single realization from the specified Graphical Model
    # GM_sample = dg.sample_L_A_Y(n_samples=1, network=network, edge_types=edge_types)[0]

    # ind_set = maximal_independent_set.maximal_n_apart_independent_set(network, 1)
    # ind_set = pd.DataFrame(list(ind_set), columns=["subject"])
    # est_df = df_for_estimation(network=network, ind_set=ind_set, sample=GM_sample)

    # with open('./autog_continuous_data/network.pkl', 'wb') as file:
    #     pickle.dump(network, file)

    # with open('./autog_continuous_data/ind_set.pkl', 'wb') as file:
    #     pickle.dump(ind_set, file)

    # with open('./autog_continuous_data/est_df.pkl', 'wb') as file:
    #     pickle.dump(est_df, file)


    # read in saved data
    with open('./autog_continuous_data/network.pkl', 'rb') as file:
        network = pickle.load(file)

    # with open('./autog_continuous_data/ind_set.pkl', 'rb') as file:
    #     ind_set = pickle.load(file)

    with open('./autog_continuous_data/est_df.pkl', 'rb') as file:
        est_df = pickle.load(file)

    beta_0, beta_1, beta_2, beta_3, beta_4, sigma2_y, theta = [0.5]*7
    initial_params = [beta_0, beta_1, beta_2, beta_3, beta_4, sigma2_y, theta]
    params_Y = optimize_params(nlcl, initial_params, f_Y_i_given_stuff, est_df)

    print("Estimated Params Y:", params_Y)
        
    sigma2_l, mu_l, weight_l= [0.5]*2
    initial_params = [sigma2_l, mu_l, weight_l]
    params_L = optimize_params(nlcl, initial_params, f_L_i_given_stuff, est_df)

    print("Estimated Params L:", params_L)
    
    def hat_f_Yi(y_i, conditionals):
        '''
        conditional pdf hat_f_Yi that is specified with inputs.

        Returns the conditional probability that Yi = y_i given conditionals.
        '''
        l_i = conditionals['L_self']
        a_i = conditionals['A_self']
            
        normalizing_weight = 1 #/ len(inputs['Y_neighbors'])
            
        l_j = conditionals['L_neighbors']
        a_j = conditionals['A_neighbors']
        y_j = conditionals['Y_neighbors']
        
        sum_l_j = sum(l_j)
        adjusted_sum_l_j = sum_l_j * normalizing_weight

        sum_a_j = sum(a_j)
        adjusted_sum_a_j = sum_a_j * normalizing_weight

        w_ij_y = normalizing_weight
        w_ij_l = normalizing_weight
            
        conditionals = {
            'y_i': y_i,
            'a_i': a_i,
            'l_i': l_i,
            'l_j_list': l_j,
            'y_j_list': y_j,  
            'adjusted_sum_l_j': adjusted_sum_l_j,
            'adjusted_sum_a_j': adjusted_sum_a_j,
            'w_ij_y': w_ij_y,
            'w_ij_l': w_ij_l 
        }
            
        return f_Y_i_given_stuff(conditionals, params_Y)


    def hat_f_Li(l_i, conditionals):
        '''
        conditional pdf hat_f_Li that is specified with inputs.

        Returns the conditional probability that Li = l_i given conditionals.
        '''  
        normalizing_weight = 1 #/ len(inputs['L_neighbors'])
            
        l_j = conditionals['L_neighbors']
        
        sum_l_j = sum(l_j)
        adjusted_sum_l_j = sum_l_j * normalizing_weight

        w_ij_l = normalizing_weight
            
        conditionals = {
            'l_i': l_i,
            'l_j_list': l_j,  
            'adjusted_sum_l_j': adjusted_sum_l_j,
            'w_ij_l': w_ij_l 
        }
            
        return f_L_i_given_stuff(conditionals, params_L)
    
    Y_A1 = gibbs_sampler_1(n_samples=10000, burn_in=10000, network=network, 
                           f_Yi=hat_f_Yi, f_Li=hat_f_Li, verbose=True, 
                           A_val=1, mode='true')
    
    beta_alpha = np.mean([beta_i_a(Y_A1, i, 3) for i in range(0, NUM_OF_SUBJECTS)])
    print("beta(alpha) =", beta_alpha)


if __name__ == "__main__":
    main()





''' SCRATCH WORK BELOW THIS LINE '''

''' AUTO LOGISTIC MODEL '''

''' 
Functions defined for the L layer
Parameters: beta_0, beta_1
'''
# def H_i(l_i, params):
#     beta_0, beta_1 = params
#     return beta_0 + beta_1*l_i

# def W_i(data_i, params, l_i=None):
#     # we allow the caller to pass in l_i separate from data_i
#     # so that when evaluating the denominator it is easier.
#     if l_i == None:
#         l_i = data_i['l_i']
#     H_i_output = H_i(l_i=l_i, params=params)
    
#     second_term = 0
#     for l_j in data_i['l_j_list']:
#         second_term += l_i * l_j * data_i['w_ij_l']
    
#     return l_i * H_i_output + second_term

# def f_L_i_given_stuff(data_i, params): 
#     '''
#     This is the distribution f(L_i | L_-i) with the parametric form specified 
#     by the auto-g paper. The distribution has paramters "params."

#     Inputs:
#         - data_i: data associated with i, which is an element in an 1-apart 
#                   maximal independent set of the network. It contains 'l_i',
#                   'l_j_list', and 'w_ij_l'.
#         - params: parameters of this distribution: beta_0, beta_1

#     The function returns the probability that L_i = l_i given l_-i (as 
#     represented with 'l_j_list').
#     '''
#     numerator = math.exp(W_i(data_i, params))
    
#     denominator = 0
#     for l_i_val in [0, 1]: 
#         # sum over all possible values of l_i (which is 1 and 0 cuz l is binary)
#         denominator += math.exp(W_i(data_i, params, l_i=l_i_val))
    
#     return numerator / denominator

# ''' 
# Functions defined for the Y layer
# Parameters: beta_0, beta_1, beta_2, beta_3, beta_4, theta
# '''
# def G_i(y_i, a_i, l_i, adjusted_sum_a_j, adjusted_sum_l_j, params):
#     beta_0, beta_1, beta_2, beta_3, beta_4 = params[:-1]
#     return beta_0 + beta_1*a_i + beta_2*l_i + beta_3*adjusted_sum_a_j + beta_4*adjusted_sum_l_j

# def theta_ij(w_ij_y, params):
#     theta = params[-1]
#     return w_ij_y * theta

# def U_i(data_i, params, y_i=None):
#     # we allow the caller to pass in y_i separate from data_i
#     # so that when evaluating the denominator it is easier.
#     if y_i == None:
#         y_i = data_i['y_i']
#     G_i_output = G_i(y_i=y_i, 
#                      a_i=data_i['a_i'], 
#                      l_i=data_i['l_i'], 
#                      adjusted_sum_a_j=data_i['adjusted_sum_a_j'], 
#                      adjusted_sum_l_j=data_i['adjusted_sum_l_j'], 
#                      params=params)
    
#     second_term = 0
#     for y_j in data_i['y_j_list']:
#         second_term += y_i * y_j * theta_ij(data_i['w_ij_y'], params)
    
#     return y_i * G_i_output + second_term

# def f_Y_i_given_stuff(data_i, params):
#     '''
#     This is the distribution f(Y_i | Y_-i = y_-i, a, l) with 
#     the parametric form specified by the auto-g paper. 
#     The distribution has paramters "params."

#     Inputs:
#         - data_i: data associated with i, which is an element in an 1-apart 
#                   maximal independent set of the network. It contains 'y_i', 
#                   'l_i', 'a_i', 'y_j_list', 'adjusted_sum_a_j', 
#                   'adjusted_sum_l_j', and 'w_ij_y'.
#         - params: parameters of this distribution: 
#                   beta_0, beta_1, beta_2, beta_3, beta_4, theta

#     The function returns the probability that Y_i = y_i given y_-i, a, l 
#     (which can be simplified to given the boundary of Y_i in the CG model).
#     '''
#     numerator = math.exp(U_i(data_i, params))
    
#     denominator = 0
#     for y_i_val in [0, 1]: 
#         # sum over all possible values of y_i (which is 1 and 0 cuz y is binary)
#         denominator += math.exp(U_i(data_i, params, y_i=y_i_val))
    
#     return numerator / denominator

# def draw(dist, inputs):
#     '''
#     Draw a realization from the distribution dist with inputs inputs.
#     '''
#     proba_of_1 = dist(inputs)
#     return np.random.binomial(1, proba_of_1)