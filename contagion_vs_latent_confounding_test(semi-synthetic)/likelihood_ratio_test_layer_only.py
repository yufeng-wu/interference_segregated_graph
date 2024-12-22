'''
Likelihood ratio test for semi-synthetic data.

This script is adapted from the likelihood_ratio_test_layer_only.py script 
in the contagion_vs_latent_confounding_test(synthetic) folder.
'''

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import pickle

# import infrastructure methods
import sys
sys.path.append("../")
from infrastructure.network_utils import *
from infrastructure.data_generator import *
from infrastructure.maximal_independent_set import *

def likelihood_ratio_test(data, Y, Z_set=[], cond_set=[], Y_type="continuous"):
    '''
    Perform a likelihood ratio test for testing the independence Y _||_ Z | cond_set.
    Null model: Y ~ cond_set
    Alternative model: Y ~ Z + cond_set
    
    Args:
    - data: DataFrame containing the data
    - Y: name of the outcome variable (assumed to be continuous unless stated otherwise)
    - Z_set: list of variables to test for independence with Y
    - cond_set: list of variable names that are conditioned on when checking independence
    - Y_type: "continuous" if Y is a continuous variable, otherwise "binary" for a binary variable

    Returns:
    - p-value from the chi-squared distribution
    '''
    # construct the regression model formula
    formula_Y = Y + " ~ 1" 
    if cond_set:
        # add conditioning variables
        formula_Y += " + " + " + ".join(cond_set)  
        
    # fit the null model (without Z_set)
    if Y_type == "continuous":
        model_null = sm.GLM.from_formula(formula=formula_Y, data=data, family=sm.families.Gaussian()).fit()
    elif Y_type == "binary":
        model_null = sm.GLM.from_formula(formula=formula_Y, data=data, family=sm.families.Binomial()).fit()
    else:
        raise ValueError("Invalid Y_type for outcome. Please use 'continuous' or 'binary'.")

    # fit the alternative model (with Z_set)
    formula_Y_alt = formula_Y + " + " + " + ".join(Z_set)
    if Y_type == "continuous":
        model_alt = sm.GLM.from_formula(formula=formula_Y_alt, data=data, family=sm.families.Gaussian()).fit()
    elif Y_type == "binary":
        model_alt = sm.GLM.from_formula(formula=formula_Y_alt, data=data, family=sm.families.Binomial()).fit()

    # the test statistic 2*(loglikelihood_alt - loglikelihood_null) is 
    # chi2 distributed with degree of freedom being the number of variables in Z_set
    test_statistic = 2 * (model_alt.llf - model_null.llf) 
    p_value = 1 - stats.chi2.cdf(x=test_statistic, df=len(Z_set))

    return p_value

def prepare_test_df(sample, network_dict, ind_set_5_hop, get_row):
    '''
    This method is different that from the synthetic data version because 
    in the synthetic version, the ith unit's data is exactly at the ith row of 
    the dataset. 
    
    However, in the semi-synthetic version (this version), the units in the data 
    is not ordered. 
    
    get_row is a dictionary that maps the unit index to the row index
    of the sample dataframe. 
    '''
    
    L_vals = []
    L_1nb_vals = []
    L_2nb_vals = []
    L_3nb_vals = []
    
    A_vals = []
    A_1nb_vals = []
    A_2nb_vals = []
    A_3nb_vals = []
    
    Y_vals = []
    Y_1nb_vals = []
    Y_2nb_vals = []

    for i in ind_set_5_hop: # i refers to the index of the unit
        if not network_dict[i]: 
            # skip nodes with no neighbors
            continue
        
        # Extract values for the current unit
        L_vals.append(sample.iloc[get_row[i]]['L'])
        A_vals.append(sample.iloc[get_row[i]]['A'])
        Y_vals.append(sample.iloc[get_row[i]]['Y'])
        
        # get the neighbors of the current unit
        neighbors = network_dict[i]
        second_order_nbs = kth_order_neighborhood(network_dict, i, 2)
        third_order_nbs = kth_order_neighborhood(network_dict, i, 3)

        # Append values for neighbors if they exist in the sample
        L_1nb_vals.append([sample.iloc[get_row[j]]['L'] for j in neighbors])
        L_2nb_vals.append([sample.iloc[get_row[j]]['L'] for j in second_order_nbs])
        L_3nb_vals.append([sample.iloc[get_row[j]]['L'] for j in third_order_nbs])

        A_1nb_vals.append([sample.iloc[get_row[j]]['A'] for j in neighbors])
        A_2nb_vals.append([sample.iloc[get_row[j]]['A'] for j in second_order_nbs])
        A_3nb_vals.append([sample.iloc[get_row[j]]['A'] for j in third_order_nbs])

        Y_1nb_vals.append([sample.iloc[get_row[j]]['Y'] for j in neighbors])
        Y_2nb_vals.append([sample.iloc[get_row[j]]['Y'] for j in second_order_nbs])

    df = pd.DataFrame({
        'L': L_vals,
        'L_1nb_sum': [sum(ls) for ls in L_1nb_vals],
        'L_2nb_sum': [sum(ls) for ls in L_2nb_vals],
        'L_3nb_sum': [sum(ls) for ls in L_3nb_vals],
        'A': A_vals,
        'A_1nb_sum': [sum(ls) for ls in A_1nb_vals],
        'A_2nb_sum': [sum(ls) for ls in A_2nb_vals],
        'A_3nb_sum': [sum(ls) for ls in A_3nb_vals],
        'Y': Y_vals,
        'Y_1nb_sum': [sum(ls) for ls in Y_1nb_vals],
        'Y_2nb_sum': [sum(ls) for ls in Y_2nb_vals],
    })
    
    return df

if __name__ == "__main__":
    # set up 
    NETWORK_NAME = "HR_edges" 
    n_trials = 100 # number of trials to run for each n_units
    sys.setrecursionlimit(4000) # set a new recursion limit
    
    # load the data
    with open(f"./intermediate_data/{NETWORK_NAME}/{NETWORK_NAME}_network.pkl", 'rb') as file:
        network_dict = pickle.load(file)
    BBB_sample = pd.read_csv(f"./intermediate_data/{NETWORK_NAME}/{NETWORK_NAME}_BBB_sample.csv")
    UUU_sample = pd.read_csv(f"./intermediate_data/{NETWORK_NAME}/{NETWORK_NAME}_UUU_sample.csv")
    ind_set_full = pd.read_csv(f"./intermediate_data/{NETWORK_NAME}/{NETWORK_NAME}_5_ind_set.csv")['subject'].tolist()
    
    # set n_units_list based on the size of ind_set_full 
    n_units_list = [i for i in range(100, len(ind_set_full) + 1, 100)] # start; end; step;
    
    # initialize dataframes to store type I error rates and power
    L_results = pd.DataFrame(columns=["n_units", "type_I_error_rate", "power"])
    A_results = pd.DataFrame(columns=["n_units", "type_I_error_rate", "power"])
    Y_results = pd.DataFrame(columns=["n_units", "type_I_error_rate", "power"])
    
    # the Unnamed: 0 column is the index of the unit
    BBB_sample.rename(columns={'Unnamed: 0': 'unit_index'}, inplace=True)
    UUU_sample.rename(columns={'Unnamed: 0': 'unit_index'}, inplace=True)
    
    # prepare mapping between the index of the unit and the row index in the UUU and BBB samples
    # this is needed because the UUU and BBB samples are not ordered by the unit index.
    # this preparetion is for faster processing in later code. 
    BBB_unit_index_to_row_index = {row['unit_index']: idx for idx, row in BBB_sample.iterrows()}
    UUU_unit_index_to_row_index = {row['unit_index']: idx for idx, row in UUU_sample.iterrows()}
    
    for n_units in n_units_list:
        # a list of True/False indicating whether the test correctly concludes
        # bidirected edges or when the L layer has bidirected edges.
        pred_correct_when_L_biedge = []
        
        # a list of True/False indicating whether the test correctly concludes
        # undirected edges or when the L layer has undirected edges.
        pred_correct_when_L_udedge = []
        
        # similar to the above, but for the A layer
        pred_correct_when_A_biedge = []
        pred_correct_when_A_udedge = []
        
        # similar to the above, but for the Y layer
        pred_correct_when_Y_biedge = []
        pred_correct_when_Y_udedge = []
        
        for i in range(n_trials):
            print(f"n_units: {n_units}, trial: {i}")
            ind_set_5_hop = random.sample(ind_set_full, n_units)
            
            BBB_test_df = prepare_test_df(BBB_sample, network_dict, ind_set_5_hop, BBB_unit_index_to_row_index)
            UUU_test_df = prepare_test_df(UUU_sample, network_dict, ind_set_5_hop, UUU_unit_index_to_row_index)
            
            # p-value of L layer test when the true model have bidirected edges (<->)
            p_value_L_biedge = likelihood_ratio_test(data=BBB_test_df, 
                                                     Y="L", 
                                                     Z_set=["L_2nb_sum"], 
                                                     cond_set=["L_1nb_sum"],
                                                     Y_type="binary")
            
            # p-value of L layer test when the true model have undirected edges (-)
            p_value_L_udedge = likelihood_ratio_test(data=UUU_test_df, 
                                                     Y="L", 
                                                     Z_set=["L_2nb_sum"], 
                                                     cond_set=["L_1nb_sum"],
                                                     Y_type="binary")
            
            # p-value of A layer test when the true model have bidirected edges (<->)
            p_value_A_biedge = likelihood_ratio_test(data=BBB_test_df, 
                                                     Y="A", 
                                                     Z_set=["A_2nb_sum"], 
                                                     cond_set=["A_1nb_sum", "L", "L_1nb_sum", "L_2nb_sum", "L_3nb_sum"],
                                                     Y_type="binary")

            # p-value of A layer test when the true model have undirected edges (-)
            p_value_A_udedge = likelihood_ratio_test(data=UUU_test_df, 
                                                     Y="A", 
                                                     Z_set=["A_2nb_sum"], 
                                                     cond_set=["A_1nb_sum", "L", "L_1nb_sum", "L_2nb_sum", "L_3nb_sum"],
                                                     Y_type="binary")
            
            # p-value of Y layer test when the true model have bidirected edges (<->)
            p_value_Y_biedge = likelihood_ratio_test(data=BBB_test_df, 
                                                     Y="Y", 
                                                     Z_set=["Y_2nb_sum"], 
                                                     cond_set=["L", "L_1nb_sum", "L_2nb_sum", "L_3nb_sum", "A", "A_1nb_sum", "A_2nb_sum", "A_3nb_sum", "Y_1nb_sum"],
                                                     Y_type="binary")
            
            # p-value of Y layer test when the true model have undirected edges (-)
            p_value_Y_udedge = likelihood_ratio_test(data=UUU_test_df, 
                                                     Y="Y", 
                                                     Z_set=["Y_2nb_sum"], 
                                                     cond_set=["L", "L_1nb_sum", "L_2nb_sum", "L_3nb_sum", "A", "A_1nb_sum", "A_2nb_sum", "A_3nb_sum", "Y_1nb_sum"],
                                                     Y_type="binary")
            
            # determine whether the test draws the correct conclusion
            pred_correct_when_L_biedge.append(p_value_L_biedge < 0.05)
            pred_correct_when_L_udedge.append(p_value_L_udedge >= 0.05)
            
            pred_correct_when_A_biedge.append(p_value_A_biedge < 0.05)
            pred_correct_when_A_udedge.append(p_value_A_udedge >= 0.05)
            
            pred_correct_when_Y_biedge.append(p_value_Y_biedge < 0.05)
            pred_correct_when_Y_udedge.append(p_value_Y_udedge >= 0.05)
            
        # calculate type I error rates and power for the L, A, Y layers
        type_I_L = 1 - np.mean(pred_correct_when_L_udedge)
        power_L = np.mean(pred_correct_when_L_biedge)
        
        type_I_A = 1 - np.mean(pred_correct_when_A_udedge)
        power_A = np.mean(pred_correct_when_A_biedge)
        
        type_I_Y = 1 - np.mean(pred_correct_when_Y_udedge)
        power_Y = np.mean(pred_correct_when_Y_biedge)
        
        # store the results into dataframes
        L_results = pd.concat([L_results, pd.DataFrame({"n_units": [n_units], "type_I_error_rate": [type_I_L], "power": [power_L]})], ignore_index=True)
        A_results = pd.concat([A_results, pd.DataFrame({"n_units": [n_units], "type_I_error_rate": [type_I_A], "power": [power_A]})], ignore_index=True)
        Y_results = pd.concat([Y_results, pd.DataFrame({"n_units": [n_units], "type_I_error_rate": [type_I_Y], "power": [power_Y]})], ignore_index=True)
    
    # save the results as csv files
    L_results.to_csv(f"./result/{NETWORK_NAME}/{NETWORK_NAME}_L_results_layer_only.csv", index=False)
    A_results.to_csv(f"./result/{NETWORK_NAME}/{NETWORK_NAME}_A_results_layer_only.csv", index=False)
    Y_results.to_csv(f"./result/{NETWORK_NAME}/{NETWORK_NAME}_Y_results_layer_only.csv", index=False)
