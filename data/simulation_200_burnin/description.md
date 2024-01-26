Data in this folder is generated using the following setup:

```python
MIN_NB = 1
MAX_NB = 6
BURN_IN = 200
VERBOSE = True

def generate_edge_types(true_model):
    # Build edge_types based on edge types at each layer
    edge_types = {}

    if true_model[0] == 'U': 
        edge_types['L'] = ['U', {'sample_given_boundary':dg.sample_given_boundary_continuous, 'verbose':VERBOSE, 'burn_in':BURN_IN}]
    else:
        edge_types['L'] = ['B', {'U_dist':dg.U_dist_1, 'f':dg.f_1}]

    if true_model[1] == 'U': 
        edge_types['A'] = ['U', {'sample_given_boundary':dg.sample_given_boundary_continuous, 'verbose':VERBOSE, 'burn_in':BURN_IN}]
    else:
        edge_types['A'] = ['B', {'U_dist':dg.U_dist_1, 'f':dg.f_binary}]

    if true_model[2] == 'U': 
        edge_types['Y'] = ['U', {'sample_given_boundary':dg.sample_given_boundary_continuous, 'verbose':VERBOSE, 'burn_in':BURN_IN}]
    else:
        edge_types['Y'] = ['B', {'U_dist':dg.U_dist_1, 'f':dg.f_1}]

    return edge_types
```