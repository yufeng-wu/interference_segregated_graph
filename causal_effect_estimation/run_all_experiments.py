'''
Master file to run all experiments in the causal_effect_estimation/ directory.

Experimens are divided into two categories: 

1. those that verify the correctness of our estimation method

2. those that either verifies the correctness of the autog method 
(e.g. UUU_autog.py, which verifies that the autog method can correctly
estimate the network causal effects when the edge types of L, A, and Y layers 
are Undirected, Undirected, and Undirected) or demonstrates the bias of 
the autog method when applied to a graphical model that violates the
the assumptions of the autog method (e.g. BBB_autog.py) 
'''

import os
import subprocess

# get all files in causal_effect_estimation/code/
os.chdir("code")
files_in_directory = os.listdir()

# get all experiments that verifies the correctness of our method
our_method_experiments = [f for f in files_in_directory if f.endswith("_ours.py")]

for experiment in our_method_experiments:
    print(f"Running {experiment}...")
    subprocess.call(['python3', experiment])
    
# get all experiments that either verifies the correctness of the autog method 
# or demonstrates the bias of the autog method when applied to a graphical model
# that violates the assumptions of the autog method.
autog_method_experiments = [f for f in files_in_directory if f.endswith("_autog.py")]

for experiment in autog_method_experiments:
    print(f"Running {experiment}...")
    subprocess.call(['python3', experiment])
