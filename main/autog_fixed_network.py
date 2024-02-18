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
from sklearn.utils import resample
import os
import pickle
from scipy.optimize import minimize
from scipy.integrate import quad
import math
import concurrent.futures


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
        neighbors_count = len(N_i) if len(N_i) > 0 else 0
        normalizing_weight = 1 / neighbors_count # TODO

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


''' AUTO LOGISTIC MODEL '''

''' 
Functions defined for the L layer
Parameters: beta_0
'''
def H_i(l_i, params):
    beta_0 = params[0]
    return beta_0

def W_i(data_i, params, l_i=None):
    # we allow the caller to pass in l_i separate from data_i
    # so that when evaluating the denominator it is easier.
    if l_i == None:
        l_i = data_i['l_i']

    second_term = 0
    for l_j in data_i['l_j_list']:
        second_term += l_i * l_j * data_i['w_ij_l']

    return l_i * H_i(l_i, params) + second_term

def f_L_i_given_stuff(data_i, params): 
    '''
    This is the distribution f(L_i | L_-i) with the parametric form specified 
    by the auto-g paper. The distribution has paramters "params."
    Inputs:
        - data_i: data associated with i, which is an element in an 1-apart 
                  maximal independent set of the network. It contains 'l_i',
                  'l_j_list', and 'w_ij_l'.
        - params: parameters of this distribution: beta_0
    The function returns the probability that L_i = l_i given l_-i (as 
    represented with 'l_j_list').
    '''
    numerator = math.exp(W_i(data_i, params))

    denominator = 0
    for l_i_val in [0, 1]: 
        # sum over all possible values of l_i (which is 1 and 0 cuz l is binary)
        denominator += math.exp(W_i(data_i, params, l_i=l_i_val))

    return numerator / denominator

''' 
Functions defined for the Y layer
Parameters: beta_0, beta_1, beta_2, beta_3, beta_4, theta
'''
def G_i(y_i, a_i, l_i, adjusted_sum_a_j, adjusted_sum_l_j, params):
    # beta_1, beta_2, beta_3, beta_4 = params[:-1]
    # return beta_1*a_i + beta_2*l_i + beta_3*adjusted_sum_a_j + beta_4*adjusted_sum_l_j

    beta_0, beta_1, beta_2, beta_3, beta_4 = params[:-1]
    return beta_0 + beta_1*a_i + beta_2*l_i + beta_3*adjusted_sum_a_j + beta_4*adjusted_sum_l_j

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
                  beta_0, beta_1, beta_2, beta_3, beta_4, theta
    The function returns the probability that Y_i = y_i given y_-i, a, l 
    (which can be simplified to given the boundary of Y_i in the CG model).
    '''
    numerator = math.exp(U_i(data_i, params))

    denominator = 0
    for y_i_val in [0, 1]: 
        # sum over all possible values of y_i (which is 1 and 0 cuz y is binary)
        denominator += math.exp(U_i(data_i, params, y_i=y_i_val))

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

def draw(dist, inputs):
    '''
    Draw a realization from the distribution dist with inputs inputs.
    '''
    proba_of_1 = dist(inputs)
    return np.random.binomial(1, proba_of_1)

def gibbs_sampler_1(n_samples, burn_in, network, f_Yi, f_Li, verbose, A_val, 
                    L_sample=None):
    '''
    Implementation of the Gibbs Sampler I algorithm on p10 of the auto-g paper.
    Inputs:
        - n_samples: number of network realizations to generate
        - burn_in: burn in period
        - network
        - f_Yi: a distribution that we can draw value from.
                Could be a wrapper function of f_Y_i_given_stuff with 
                estimated parameters.
        - f_Li: a distribution that we can draw value from.
                Could be a wrapper function of f_L_i_given_stuff with 
                estimated parameters.
        - verbose: True or False
        - A_val: the value of A_i that we set / intervene for each A_i.
                 should be either 0 or 1 as we are working with binary variables.
        - L_sample: the orignal sample of the L layer of the graph
    
    Return:
        - n_samples samples of realization of the network after burn_in period.
    '''
    samples = []
    N = len(network) 

    # produce initial values
    sample = pd.DataFrame(index=network.keys(), columns=['L', 'A', 'Y'])

    if L_sample is None:
        sample['L'] = {vertex: random.choice([1, 0]) for vertex in network.keys()}
    else:
        sample['L'] = L_sample
    sample['A'] = {vertex: A_val for vertex in network.keys()}
    sample['Y'] = {vertex: random.choice([1, 0]) for vertex in network.keys()}

    for m in range(n_samples + burn_in):
        if verbose and m % 1000 == 0:
            print("progress: ", m / (n_samples + burn_in))

        i = (m % N) # the paper has "+1" but i don't +1 because the index of subjects in my network starts from 0
        
        # draw L_i ~ f(L_i | L_-i)
        if L_sample is None:
            boundary_values_L = {
                'L_neighbors': [sample.loc[neighbor, 'L'] for neighbor in network[i]]
            }
            new_Li = draw(dist=f_Li, inputs=boundary_values_L)

        # draw Y_i ~ f(Y_i | Y_-i)
        boundary_values_Y = {
            'L_self': sample.loc[i, 'L'],
            'L_neighbors': [sample.loc[neighbor, 'L'] for neighbor in network[i]],
            'A_self': sample.loc[i, 'A'],
            'A_neighbors': [sample.loc[neighbor, 'A'] for neighbor in network[i]],
            'Y_neighbors': [sample.loc[neighbor, 'Y'] for neighbor in network[i]],
        }
        new_Yi = draw(dist=f_Yi, inputs=boundary_values_Y)

        # update the newly drawn Li (if L_sample is not passed in) and Yi
        if L_sample is None:
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

def estimate_autog_beta_alpha(est_df_sample, n_samples_autog, burn_in_autog, 
                              network, num_of_subejects, A_val, L_sample):
    '''Estimate Network Causal Effects Via Auto-G '''
    beta_0, beta_1, beta_2, beta_3, beta_4, theta = [0.5]*6
    initial_params = [beta_0, beta_1, beta_2, beta_3, beta_4, theta]
    # beta_1, beta_2, beta_3, beta_4, theta = [0.5]*5 # not estimating beta_0 here
    # initial_params = [beta_1, beta_2, beta_3, beta_4, theta]
    params_Y = optimize_params(nlcl, initial_params, f_Y_i_given_stuff, est_df_sample)
    print("Estimated Params Y:", params_Y)
        
    beta_0 = 0.5
    initial_params = [beta_0]
    params_L = optimize_params(nlcl, initial_params, f_L_i_given_stuff, est_df_sample)
    print("Estimated Params L:", params_L)
    
    def hat_f_Yi(inputs):
        '''
        return: proba of yi = 1
        '''
        y_i = 1
        l_i = inputs['L_self']
        a_i = inputs['A_self']
            
        neighbors_count = len(inputs['Y_neighbors']) if len(inputs['Y_neighbors']) > 0 else 0
        normalizing_weight = 1 / neighbors_count #TODO
            
        l_j = inputs['L_neighbors']
        a_j = inputs['A_neighbors']
        y_j = inputs['Y_neighbors']
        
        sum_l_j = sum(l_j)
        adjusted_sum_l_j = sum_l_j * normalizing_weight

        sum_a_j = sum(a_j)
        adjusted_sum_a_j = sum_a_j * normalizing_weight

        w_ij_y = normalizing_weight
        w_ij_l = normalizing_weight
            
        inputs = {
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
        return f_Y_i_given_stuff(inputs, params_Y)

    def hat_f_Li(inputs):
        '''
        return: proba of li = 1
        '''
        l_i = 1
        neighbors_count = len(inputs['L_neighbors']) if len(inputs['L_neighbors']) > 0 else 0
        normalizing_weight = 1 / neighbors_count #TODO
        l_j = inputs['L_neighbors']
        w_ij_l = normalizing_weight

        inputs = {
            'l_i': l_i,
            'l_j_list': l_j,
            'w_ij_l': w_ij_l 
        }
        return f_L_i_given_stuff(inputs, params_L)

    Y_A1 = gibbs_sampler_1(n_samples=n_samples_autog, burn_in=burn_in_autog, network=network, 
                           f_Yi=hat_f_Yi, f_Li=hat_f_Li, verbose=False, 
                           A_val=A_val, L_sample=L_sample)
    
    autog_beta_alpha = np.mean([beta_i_a(Y_A1, i, 3) for i in range(0, num_of_subejects)])
    print("AUTO-G beta(alpha) =", autog_beta_alpha)

    return autog_beta_alpha

def bootstrap_confidence_interval(data, n_bootstraps, alpha, n_samples_autog, 
                                  burn_in_autog, network, num_of_subejects, 
                                  A_val, L_sample):
    bootstrapped_estimates = []

    for _ in range(n_bootstraps):
        another_sample = resample(data) # sample with replacement
        estimate = estimate_autog_beta_alpha(another_sample, n_samples_autog, 
                    burn_in_autog, network, num_of_subejects, A_val, L_sample)
        bootstrapped_estimates.append(estimate)
    
    # calculate the lower & upper percentiles for confidence interval
    lower = 100 * alpha / 2
    upper = 100 * (1 - alpha / 2)
    confidence_interval = np.percentile(bootstrapped_estimates, [lower, upper])
    return confidence_interval
    

def run_experiment(num_subjects, est_df, true_beta_alpha, network):

    n_samples_autog = 10 * num_subjects
    burn_in_autog  = 10 * num_subjects
    est_df = resample(est_df, n_samples=num_subjects)

    '''Estimate Network Causal Effects Via Auto-G'''

    ## L_sample=GM_sample['L'] : using the original L sample 
    ## L_sample=None : generate new L samples based on the estimated f_Li
    conf_int = bootstrap_confidence_interval(
        data=est_df, n_bootstraps=100, alpha=0.05, n_samples_autog=n_samples_autog, 
        burn_in_autog=burn_in_autog, network=network, num_of_subejects=num_subjects, 
        A_val=1, L_sample=None)
    
    print("AUTO-G Confidence Interval:", conf_int)

    # The modified main function content goes here, using num_subjects instead of a fixed number
    # Return the results instead of printing them directly
    return {
        'num_subjects': num_subjects,
        'true_beta_alpha': true_beta_alpha,
        'confidence_interval': conf_int,
    }

def run_experiments_with_multiprocessing():
    num_subjects_list = [500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
    output_file = '../result/autog_experiments/AUTOG_FIXED_NETWORK.csv'
    pd.DataFrame([], columns=['num_subjects', 'true_beta_alpha', 'confidence_interval']).to_csv(output_file, index=False)

    # Generate a single network realization and keep that fixed for all experiments
    network = util.create_random_network(n=max(num_subjects_list), 
                                         min_neighbors=1, 
                                         max_neighbors=6)

    '''Evaluate True Network Causal Effects '''
    # evaluate true causal effect using true f_Yi, true f_Li, n, and A_val (via beta_i_a and gibbs sampler 1)
    true_f_Yi = dg.sample_given_boundary_binary
    true_f_Li = dg.sample_given_boundary_binary

    Y_A1 = gibbs_sampler_1(n_samples=max(num_subjects_list), 
                           burn_in=10*max(num_subjects_list), 
                           network=network, f_Yi=true_f_Yi, f_Li=true_f_Li, 
                           verbose=False, A_val=1)
    
    true_beta_alpha = np.mean([beta_i_a(Y_A1, i, 3) for i in range(0, max(num_subjects_list))])
    print("TRUE beta(alpha) =", true_beta_alpha)

    edge_types = gns.generate_edge_types("UUU")
    GM_sample = dg.sample_L_A_Y(n_samples=1, network=network, edge_types=edge_types)[0]
    ind_set_1_apart = maximal_independent_set.maximal_n_apart_independent_set(network, 1)
    ind_set_1_apart = pd.DataFrame(list(ind_set_1_apart), columns=["subject"])
    est_df = df_for_estimation(network=network, ind_set=ind_set_1_apart, sample=GM_sample)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_experiment, num_subjects, est_df, true_beta_alpha, network) for num_subjects in num_subjects_list]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            pd.DataFrame([result]).to_csv(output_file, mode='a', index=False, header=False)
    
    print(f"Experiment completed. Results saved to '{output_file}'.")


def main():
    run_experiments_with_multiprocessing()

if __name__ == "__main__":
    main()