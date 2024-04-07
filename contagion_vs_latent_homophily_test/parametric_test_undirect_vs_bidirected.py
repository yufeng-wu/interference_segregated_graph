from network_utils import create_cycle_graph, graph_to_edges
from data_generator import sample_from_BG, sample_from_UG
from maximal_independent_set import maximal_n_apart_independent_set
from scipy.special import expit
from scipy.optimize import minimize
from scipy.stats import chi2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def log_likelihood(data, params, model):
    """
    Compute the log likelihood of the observed data given the parameters.

    Params:
        data (pd.DataFrame): the observed data
        params (list[float]): fitted parameters
        model: function that computes the probability of a single data point
               given the parameters
    
    Returns:
        The log likelihood of the data given the parameters.
    """
    log_likelihood = 0.0

    for index, row in data.iterrows():
        prob = model(data=row, params=params)
        log_likelihood += row['V'] * np.log(prob) + (1 - row['V']) * np.log(1 - prob)

    return log_likelihood

def estimate_parameters(data, model, initial_params):
    """
    Estimate the parameters in model that maximize the 
    likelihood of the observed data.

    Params:
        data (pd.DataFrame): the observed data
        model: function that computes the likelihood of a single data point
               given the parameters
        initial_params (list[float]): initial guess for the parameters. 
                        Array of real elements of size (n,), 
                        where n is the number of independent variables.
    
    Returns:
        The estimated parameters that maximize the likelihood.
    """
    # Define the objective function as the negative log-likelihood
    # because 'minimize' performs minimization
    def objective(params):
        return -log_likelihood(data, params, model)

    # Perform the optimization
    result = minimize(objective, initial_params, method='L-BFGS-B')
    
    # Check if the optimization was successful
    if result.success:
        return result.x
    else:
        raise ValueError(result.message)

def assemble_df(sample, graph, ind_set, nbnb=False):
    V_values = []
    V_nb_values = []
    V_nbnb_values = []

    for v in ind_set:
        V_values.append(sample[v])
        V_nb_values.append([sample[nb] for nb in graph[v]])
        if nbnb:
            # Create a set of next-nearest neighbors,
            # excluding self and direct neighbors
            v_neighbors = set(graph[v])
            v_nbnb = set()
            for v_neighbor in v_neighbors:
                v_nbnb.update(graph[v_neighbor])
            
            # Remove the current vertex and its neighbors
            v_nbnb.difference_update(v_neighbors)
            v_nbnb.discard(v)

            # Add the next-nearest neighbors values to the list
            V_nbnb_values.append([sample[element] for element in v_nbnb])

    # Assemble the DataFrame
    if nbnb:
        df = pd.DataFrame({
            'V': V_values,
            'V_nb_values': V_nb_values,
            'V_nbnb_values': V_nbnb_values
        })
    else:
        df = pd.DataFrame({
            'V': V_values,
            'V_nb_values': V_nb_values
        })
    
    return df

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
    # noise = np.random.normal(0, 0.1)
    linear_sum = 3 * sum(U_values) # + noise
    prob = expit(linear_sum)  # Sigmoid function to get a value between 0 and 1
    return np.random.binomial(1, prob)  # Sample from a Bernoulli distribution

    # '''
    # Define the function f that calculates V based on its neighboring U values
    # '''
    # noise = np.random.normal(0, 0.1)
    # return sum(U_values) + noise

def prob_v_given_neighbors(data, params=[0.2, 0.5]):
    '''
    Define the parametric form for the conditional probability P(V_i = 1 | -V_i)
    using only the V_neighbors as input. V_i is a binary variable that is either
    0 or 1. The parameters a0 and a1 are hard-coded inside the function.

    Params:
        - V_neighbors: array-like, containing the values of V's neighbors
    
    Return:
        - a float that represents the conditional probability
    '''
    # Parameters can be hard-coded or defined as constants elsewhere
    a0 = params[0]
    a1 = params[1]

    V_nb_values = data["V_nb_values"]

    return expit(a0 + a1 * np.sum(V_nb_values))

def prob_v_given_two_layer_neighbors(data, params):
    '''
    The alternative model.
    '''
    # Parameters can be hard-coded or defined as constants elsewhere
    a0 = params[0]
    a1 = params[1]
    a2 = params[2]

    V_nb_values = data["V_nb_values"]
    V_nbnb_values = data["V_nbnb_values"]

    return expit(a0 + a1 * np.sum(V_nb_values) + a2 * np.sum(V_nbnb_values))

def likelihood_ratio_test(data_null, params_null, model_null, data_alt, 
                          params_alt, model_alt, alpha=0.05):
    # Compute the log-likelihoods for each model specification
    null_model_log_likelihood = log_likelihood(data_null, params_null, model_null)
    alt_model_log_likelihood = log_likelihood(data_alt, params_alt, model_alt)

    # Calculate the test statistic
    # Test Statistic = −2( ln(L(null model)) − ln(L(alternative model)) )
    LR_statistic = -2 * (null_model_log_likelihood - alt_model_log_likelihood)

    # Assuming that the test statistic follows a chi-squared distribution
    # The degrees of freedom is the difference in the number of parameters
    # between the two models. 
    degrees_of_freedom = len(params_alt) - len(params_null)

    p_value = chi2.sf(LR_statistic, degrees_of_freedom)

    print(f"Likelihood Ratio Test Statistic: {LR_statistic}")
    print(f"p-value: {p_value}")

    # Conclude based on the p-value
    if p_value < alpha:
        print("The alternative model provides significantly more information.")
    else:
        print("There is NOT enough evidence to suggest that the alternative model provide more information.")

def calculate_residuals(data, params, model):
    """
    Calculate residuals for the model (a pre-specified parametric form) with its
    estimated parameters.
    """
    residuals = []
    for index, row in data.iterrows():
        prob = model(data=row, params=params)
        # For a binary outcome, the residual can be calculated as observed - predicted
        residual = row['V'] - prob
        residuals.append(residual)
    return residuals

def check_residuals(residuals, plot_title):
    """
    Perform residual analysis to check for model misspecification.
    """
    # Convert the list of residuals to a numpy array for better handling
    residuals = np.array(residuals)

    # Check if the mean of residuals is approximately 0
    mean_residual = np.mean(residuals)
    print(f"Mean of residuals: {mean_residual}")

    # Visually inspect for randomness in residuals
    plt.scatter(range(len(residuals)), residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.show()

def fitted_plot_with_jitter(data, params, model, predictor_col='V_nb_values', 
                            outcome_col='V', plot_title='title', 
                            jitter_amount=0.05):
    """
    Plot data and the fitted curve in one plot with jitter added to 
    binary outcome for better visualization.
    """
    # Add jitter to the binary outcome and the predictor
    data_jittered = data.copy()
    data_jittered[outcome_col] = data[outcome_col] + \
        np.random.uniform(-jitter_amount, jitter_amount, size=data.shape[0])
    data_jittered[predictor_col] = data[predictor_col].apply(lambda x: [i+np.random.uniform(-jitter_amount, jitter_amount) for i in x])

    # Generate a range of predictor values for plotting the fitted line
    predictor_values = np.linspace(start=min(data[predictor_col].apply(sum))-1, 
                                   stop=max(data[predictor_col].apply(sum))+1, 
                                   num=100)

    # Calculate the fitted probabilities
    fitted_probs = [model(data={predictor_col: [v] * len(params)}, params=params) for v in predictor_values]
    
    # Plot the observed outcomes with jitter
    plt.scatter(data_jittered[predictor_col].apply(sum), data_jittered[outcome_col], alpha=0.1, label='Observed data')
    # Plot the fitted curve
    plt.plot(predictor_values, fitted_probs, color='red', label='Fitted Curve')

    plt.xlabel('Sum of ' + predictor_col)
    plt.ylabel('Probability of ' + outcome_col + ' with jitter')
    plt.legend()
    plt.show()


if __name__ == "__main__":

    ''' Set up '''
    # Set a new recursion limit
    sys.setrecursionlimit(4000)
    NUM_OF_VERTICES = 10000
    BURN_IN_PERIOD = 200

    ''' STEP 1: Greate graph '''
    graph = create_cycle_graph(NUM_OF_VERTICES)
    
    ''' STEP 2: Generate data assuming that the edges are undirected ''' 
    sample_UG = sample_from_UG(graph=graph, 
                               prob_v_given_neighbors=prob_v_given_neighbors,
                               verbose=True,
                               burn_in=BURN_IN_PERIOD)
    
    ''' STEP 3: Generate data assuming that the edges are bidirected ''' 
    sample_BG = sample_from_BG(edges=graph_to_edges(graph), 
                               U_dist=U_dist, 
                               f=f)
    
    ''' STEP 4: Get independent set from graph ''' 
    ind_set = maximal_n_apart_independent_set(graph, 
                                              n=5, 
                                              available_vertices=set(graph.keys()),
                                              approx=True,
                                              is_cycle_graph=True)
    
    ''' STEP 5: Likelihood Ratio Test for UG '''
    # STEP 5.1: Construct dataframes for both models
    # df with the columns: V, V_nb
    UG_df_v_nb = assemble_df(sample=sample_UG, graph=graph, ind_set=ind_set, nbnb=False)
    # df with the columns: V, V_nb, V_nbnb
    UG_df_v_nb_nbnb = assemble_df(sample=sample_UG, graph=graph, ind_set=ind_set, nbnb=True)

    # STEP 5.2: Estimate params for a given model that maximizes 
    #           the likelihood of observing the sample data.
    UG_nb_params = estimate_parameters(data=UG_df_v_nb, 
                                       model=prob_v_given_neighbors,
                                       initial_params=[0.0, 0.0])
    print("UG_nb_params: ", UG_nb_params)
    UG_nbnb_params = estimate_parameters(data=UG_df_v_nb_nbnb, 
                                         model=prob_v_given_two_layer_neighbors,
                                         initial_params=[0.0, 0.0, 0.0])
    print("UG_nbnb_params: ", UG_nbnb_params)
    # STEP 5.3: Perform LRT
    print("\nLikelihood-Ratio Test for data sampled from a true UG:")
    likelihood_ratio_test(data_null=UG_df_v_nb, 
                          params_null=UG_nb_params, 
                          model_null=prob_v_given_neighbors,
                          data_alt=UG_df_v_nb_nbnb,
                          params_alt=UG_nbnb_params,
                          model_alt=prob_v_given_two_layer_neighbors,
                          alpha=0.05)
    
    
    ''' STEP 6: Likelihood Ratio Test for BG '''
    
    # STEP 6.1: Construct dataframes for both models
    # df with the columns: V, V_nb
    BG_df_v_nb = assemble_df(sample=sample_BG, graph=graph, ind_set=ind_set, nbnb=False)
    # df with the columns: V, V_nb, V_nbnb
    BG_df_v_nb_nbnb = assemble_df(sample=sample_BG, graph=graph, ind_set=ind_set, nbnb=True)

    # STEP 6.2: Estimate params for a given model that maximizes 
    #           the likelihood of observing the sample data.
    BG_nb_params = estimate_parameters(data=BG_df_v_nb, 
                                       model=prob_v_given_neighbors,
                                       initial_params=[0.0, 0.0])
    print("BG_nb_params: ", BG_nb_params)
    BG_nbnb_params = estimate_parameters(data=BG_df_v_nb_nbnb, 
                                         model=prob_v_given_two_layer_neighbors,
                                         initial_params=[0.0, 0.0, 0.0])
    print("BG_nbnb_params: ", BG_nbnb_params)
    # STEP 6.3: Perform LRT
    print("\nLikelihood-Ratio Test for data sampled from a true BG:")
    likelihood_ratio_test(data_null=BG_df_v_nb, 
                          params_null=BG_nb_params, 
                          model_null=prob_v_given_neighbors,
                          data_alt=BG_df_v_nb_nbnb,
                          params_alt=BG_nbnb_params,
                          model_alt=prob_v_given_two_layer_neighbors,
                          alpha=0.05)


    ''' Step 7: Check for Model Misspecification '''
    # Data from true UG
    residuals = calculate_residuals(data=UG_df_v_nb, params=UG_nb_params, 
                                    model=prob_v_given_neighbors)
    check_residuals(residuals, plot_title="Data from true undirected graph")
    fitted_plot_with_jitter(data=UG_df_v_nb, params=UG_nb_params, 
                            model=prob_v_given_neighbors, jitter_amount=0.05,
                            plot_title="Data from true undirected graph")
    
    # Data from true BG
    BG_residuals = calculate_residuals(data=BG_df_v_nb, params=BG_nb_params, 
                                       model=prob_v_given_neighbors)
    check_residuals(BG_residuals, plot_title="Data from true bidirected graph")
    fitted_plot_with_jitter(data=BG_df_v_nb, params=BG_nb_params, 
                            model=prob_v_given_neighbors, jitter_amount=0.05,
                            plot_title="Data from true bidirected graph")
    