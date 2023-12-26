import numpy as np
import random
from itertools import product

# Unnormalized joint distribution
# i.e. P(A=a, B=b, C=c) = psi_1(x, y) * psi_2(y, z) * psi_3(x) * psi_4(y) * psi_5(z)
# values is a dictionary in the form such as {'A': 1, 'B': 1, 'C': 0}
def joint_dist(values):
    result = 1.0
    assert len(values) == 3
    assert 'A' in values.keys() and 'B' in values.keys() and 'C' in values.keys()
    for clique, factor_func in factors.items():
        clique_var_values = [values[var] for var in clique]
        result *= factor_func(*clique_var_values)
    return result

def calculate_normalizing_constant(variables_domain):
    Z = 0.0
    variable_names = list(variables_domain.keys())
    # Generate all possible combinations of variable values
    for values in product(*[variables_domain[var] for var in variable_names]):
        # map variable names to their values
        values_dict = dict(zip(variable_names, values))
        Z += joint_dist(values_dict)
    return Z

def normalized_joint_dist(values, Z):
    '''
    values: a dictionary in the form such as {'A': 1, 'B': 1, 'C': 0}
    '''
    return joint_dist(values) / Z

def conditional_dist(left, right):
    '''
    Calculate the conditional distribution P(left | right).

    P(X|Y) in UG is factorized as: 
    (the product of all clique potentials involving X) 
        / Sum over x(the product of all clique potentials involving A)

    params:
    - left: a dictionary that maps one variable name to one value
    - right: a dictionary that maps the rest of the variable names to their values
    ''' 
    assert len(left) == 1 and len(left) + len(right) == len(graph)
    
    # get variable name
    left_var = list(left.keys())[0]
    
    numerator = 1.0
    denominator = 1.0
    
    for clique, factor_func in factors.items():
        if left_var in clique:
            # Modify numerator
            clique_var_values = []
            for var in clique:
                if var == left_var:
                    clique_var_values.append(left[var])
                else:
                    clique_var_values.append(right[var])
            numerator *= factor_func(*clique_var_values)
        
    for left_val in variables_domain[left_var]:
        denominator_added = 1.0
        for clique, factor_func in factors.items():
            if left_var in clique:
                clique_var_values = []
                for var in clique:
                    if var == left_var:
                        clique_var_values.append(left_val)
                    else:
                        clique_var_values.append(right[var])

                denominator_added *= factor_func(*clique_var_values)
        denominator += denominator_added

    return numerator / denominator

# UG structure: A -- B -- C
graph = {
    'A': ['B'],
    'B': ['A', 'C'],
    'C': ['B']
}

# Factors (a non-negative function) for each clique
# psi_1() through psi_5()
factors = {
    ('A') : lambda a: np.log(10 + a),
    ('B') : lambda b: 2*(11 + b),
    ('C') : lambda c: 2**(1 + c),
    ('A', 'B'): lambda a, b: np.exp(a + b),
    ('B', 'C'): lambda b, c: np.exp(b - c)
}

# A, B, C are all binary variables
variables_domain = {'A': [1, 0], 'B': [1, 0], 'C': [1, 0]}

Z = calculate_normalizing_constant(variables_domain)
var_values = {'A': 1, 'B': 1, 'C': 1}


#-------------- Gibbs sampler --------------#
# Initialize variables with random values
variables = {'A': random.choice(variables_domain['A']),
             'B': random.choice(variables_domain['B']),
             'C': random.choice(variables_domain['C'])}

num_iterations = 10000
warm_up = 5000

samples = []

for i in range(num_iterations):
    for node in graph.keys():
        right = dict(variables) # create a new copy
        del right[node]
        left = {node : variables[node]}
        conditional_proba = conditional_dist(left, right)
        # the interpretation of conditional proba is P(x=1 | y).
        if left[node] == 0:
            conditional_proba = 1 - conditional_proba
        # sample a new value for the current node based on conditional proba
        variables[node] = np.random.choice(variables_domain[node], size=1, p=np.array([conditional_proba, 1-conditional_proba]))[0]

    if i >= warm_up:
        samples.append(dict(variables))


# Extract the values of variables A, B, and C from the samples
A_values = [sample['A'] for sample in samples]
B_values = [sample['B'] for sample in samples]
C_values = [sample['C'] for sample in samples]

def empirical_cond_proba_A_given_BC(a, b, c):
    count = 0
    total_count = 0
    for i in range(len(samples)):
        if B_values[i] == b and C_values[i] == c:
            total_count += 1
            if A_values[i] == a:
                count += 1
    return count/total_count


# Calculate the odds ratio for (A and C) given B=1
OR_A_and_C_given_B = (empirical_cond_proba_A_given_BC(a=1, b=1, c=1) / \
                      empirical_cond_proba_A_given_BC(a=0, b=1, c=1)) * \
                     (empirical_cond_proba_A_given_BC(a=0, b=1, c=0) / \
                      empirical_cond_proba_A_given_BC(a=1, b=1, c=0))

# Print the results
print(f"Odds of (A and C) given B=1: {OR_A_and_C_given_B}")