'''
Likelihood ratio test for semi-synthetic data.

Different from likelihood_ratio_test.py in this folder, the following program
selects 100 random maximal 6-degree separated set (5-hop independent set) from
a given network, and compute the type I error rate and power of the test.
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

def prepare_test_df(sample, network_dict):
    unit_ids = []
    
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
    
    network_size = len(network_dict)
    counter = 0

    for i in network_dict.keys():
        
        # progress tracker
        counter += 1
        if counter % 1000 == 0:
            print("[Already Processed]:", counter / network_size)
            
        neighbors = network_dict[i]
        second_order_nbs = kth_order_neighborhood(network_dict, i, 2)
        third_order_nbs = kth_order_neighborhood(network_dict, i, 3)
        
        # skip the unit if its neighbors or second order neighbors are empty
        # because our test requires at least one neighbor and one second order neighbor
        if not neighbors or not second_order_nbs:
            continue
        
        unit_ids.append(i)
        
        L_vals.append(sample['L'][i])
        L_1nb_vals.append([sample['L'][j] for j in neighbors])
        L_2nb_vals.append([sample['L'][j] for j in second_order_nbs])
        L_3nb_vals.append([sample['L'][j] for j in third_order_nbs])
        
        A_vals.append(sample['A'][i])
        A_1nb_vals.append([sample['A'][j] for j in neighbors])
        A_2nb_vals.append([sample['A'][j] for j in second_order_nbs])
        A_3nb_vals.append([sample['A'][j] for j in third_order_nbs])
        
        Y_vals.append(sample['Y'][i])
        Y_1nb_vals.append([sample['Y'][j] for j in neighbors])
        Y_2nb_vals.append([sample['Y'][j] for j in second_order_nbs])

    df = pd.DataFrame({
        'unit_id': unit_ids,
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
    # NETWORK_NAME = "HR_edges" 
    # NETWORK_NAME = "HU_edges" 
    # NETWORK_NAME = "RO_edges"
    # NETWORK_NAME = "deezer_europe_edges"
    NETWORK_NAME = "lastfm_asia_edges"  
    print("--- NETWORK_NAME --- :", NETWORK_NAME)
    
    n_trials = 200 
    np.random.seed(42)
    # sys.setrecursionlimit(4000) # set a new recursion limit
    
    # load the data
    with open(f"./intermediate_data/{NETWORK_NAME}/{NETWORK_NAME}_network.pkl", 'rb') as file:
        network_dict = pickle.load(file)
    BBB_sample = pd.read_csv(f"./intermediate_data/{NETWORK_NAME}/{NETWORK_NAME}_BBB_sample.csv")
    UUU_sample = pd.read_csv(f"./intermediate_data/{NETWORK_NAME}/{NETWORK_NAME}_UUU_sample.csv")
    
    # the Unnamed: 0 column is the index of the unit;
    # make sure that the unit_index in the network is the same as the row number in the sample dataframes.
    BBB_sample.rename(columns={'Unnamed: 0': 'unit_index'}, inplace=True)
    UUU_sample.rename(columns={'Unnamed: 0': 'unit_index'}, inplace=True)
    BBB_sample = BBB_sample.sort_values(by='unit_index').reset_index(drop=True)
    UUU_sample = UUU_sample.sort_values(by='unit_index').reset_index(drop=True)
    BBB_sample.drop(columns=['unit_index'], inplace=True)
    UUU_sample.drop(columns=['unit_index'], inplace=True)
    
    # prepare test dataframes; later we will extract relevant rows from these 
    # dataframes given the units in ind_set
    BBB_test_df_full = prepare_test_df(BBB_sample, network_dict)
    UUU_test_df_full = prepare_test_df(UUU_sample, network_dict)
    
    # prepare lists to store results
    pred_correct_when_L_biedge = []
    pred_correct_when_L_udedge = []
    pred_correct_when_A_biedge = []
    pred_correct_when_A_udedge = []
    pred_correct_when_Y_biedge = []
    pred_correct_when_Y_udedge = []
    
    for i in range(n_trials):
        
        # run 5 times and select the largest independent set
        # largest_ind_set = {}
        # for _ in range(2):
        #     ind_set = maximal_n_hop_independent_set(network_dict, n=5, verbose=False)
        #     if len(ind_set) > len(largest_ind_set):
        #         largest_ind_set = ind_set
        
        # ind_set = largest_ind_set
        
        ind_set = maximal_n_hop_independent_set(network_dict, n=5, verbose=False)
        print("size of ind set: ", len(ind_set))
        
        # extract relevant data for the current likelihood ratio test
        BBB_test_df = BBB_test_df_full[BBB_test_df_full['unit_id'].isin(ind_set)].copy()
        UUU_test_df = UUU_test_df_full[UUU_test_df_full['unit_id'].isin(ind_set)].copy()
        
        # drop the unit_id column because it's unused for the test
        BBB_test_df.drop(columns=['unit_id'], inplace=True)
        UUU_test_df.drop(columns=['unit_id'], inplace=True)
        
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
                                                    Z_set=["L_3nb_sum", "A_2nb_sum"], 
                                                    cond_set=["A_1nb_sum", "L", "L_1nb_sum", "L_2nb_sum"],
                                                    Y_type="binary")
        # p-value of A layer test when the true model have undirected edges (-)
        p_value_A_udedge = likelihood_ratio_test(data=UUU_test_df, 
                                                    Y="A", 
                                                    Z_set=["L_3nb_sum", "A_2nb_sum"], 
                                                    cond_set=["A_1nb_sum", "L", "L_1nb_sum", "L_2nb_sum"],
                                                    Y_type="binary")
        
        # p-value of Y layer test when the true model have bidirected edges (<->)
        p_value_Y_biedge = likelihood_ratio_test(data=BBB_test_df, 
                                                    Y="Y", 
                                                    Z_set=["L_3nb_sum", "A_3nb_sum", "Y_2nb_sum"], 
                                                    cond_set=["L", "L_1nb_sum", "L_2nb_sum", "A", "A_1nb_sum", "A_2nb_sum", "Y_1nb_sum"],
                                                    Y_type="binary")
        # p-value of Y layer test when the true model have undirected edges (-)
        p_value_Y_udedge = likelihood_ratio_test(data=UUU_test_df, 
                                                    Y="Y", 
                                                    Z_set=["L_3nb_sum", "A_3nb_sum", "Y_2nb_sum"], 
                                                    cond_set=["L", "L_1nb_sum", "L_2nb_sum", "A", "A_1nb_sum", "A_2nb_sum", "Y_1nb_sum"],
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
    
    # Compile final results
    result = pd.DataFrame({
        'outcome': ['L', 'A', 'Y'],
        'type_I_error': [type_I_L, type_I_A, type_I_Y],
        'power': [power_L, power_A, power_Y]
    })
    
    # save the results as csv files
    result.to_csv(f"./result/{NETWORK_NAME}/{NETWORK_NAME}_result.csv", index=False)