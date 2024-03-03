# Implementing the EM algorithm

import numpy as np 
from scipy.optimize import minimize
from scipy.special import expit

# U -> X
def generate_data(n_samples=1000):
    U = np.random.binomial(1, 0.8, n_samples) # params[0]
    X = 0.5 * U + np.random.normal(0, 1, n_samples) # params[1] params[2]
    return U, X 

def p_u(u, params):
    return params[0] if u == 1 else 1 - params[0]

def p_x1u(x, u, params):
    mu = params[1] * u  # mean
    sigma = params[2]  # std

    # Gaussian PDF for x given u
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def p_u1x(u, x, old_params):
    numer = p_u(u, old_params) * p_x1u(x, u, old_params)
    denom = np.sum([p_u(u, old_params) * p_x1u(x, u, old_params) for u in [0, 1]])
    return numer / denom

def Q(params, x, old_params):
    expected_val = 0
    for u in [0, 1]:
        expected_val += np.log(p_u(u, params)) * p_u1x(u, x, old_params)
    return expected_val

def M_step(X, old_params):
    #def objective(params, X, old_params):
    #    return -np.sum([Q(params, x, old_params) for x in X])
    def objective(params, X, old_params):
        pU1_given_Xoldparams = [p_u1x(1, x, old_params) for x in X]
        U = np.random.binomial(1, pU1_given_Xoldparams)
        return -np.sum([np.log(p_u(u, params)) + np.log(p_x1u(x, u, params)) for u, x in zip(U, X)])
    new_params = minimize(objective, 
                          x0=np.random.uniform(0, 1, 3), 
                          args=(X, old_params), 
                          bounds=[(0, 1)]).x
    print(objective(new_params, X, old_params))
    return new_params

def main():
    U, X = generate_data()
    params = np.random.uniform(0, 1, 3)
    for i in range(100):
        params = M_step(X, params)
        print(params)
    print("True Params: [0.8, 0.5, 1.0]")

main()

# focus on autog implementation now
# send rohit EM code
# thesis outline, chapter by chapter
