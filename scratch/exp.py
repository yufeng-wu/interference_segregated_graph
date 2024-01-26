# experiment how many trees is suitable for random forest in my project
from nonparametric_test_undirected_vs_bidirected import prepare_data, test_edge_type
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import pandas as pd
import random
import os
import pickle

# Global variables
DATA_SOURCE = "../data/simulation/"

def main():
    true_models = ["UBU"]
    effective_sample_sizes = [1000, 3000, 5000]
    num_trees_options = [20, 50, 100, 200, 300, 400, 500, 800, 1000]

    with open(os.path.join(DATA_SOURCE, 'network.pkl'), 'rb') as file:
        network = pickle.load(file)

    # Load the pre-specified 5-independent set
    ind_set_full = pd.read_csv(os.path.join(DATA_SOURCE, '5_ind_set.csv'))['subject'].tolist()
    
    for true_model in true_models:
        GM_sample = pd.read_csv(os.path.join(DATA_SOURCE, f"{true_model}_sample.csv"))

        for sample_size in effective_sample_sizes:
            # Randomly select sample_size elements from the pre-specified 5-ind set
            ind_set = random.sample(ind_set_full, sample_size)

            # Prepare data using 5-independent set for later ML models
            df = prepare_data(GM_sample, ind_set, network)

            predictors = ['L_1nb_sum', 'L_1nb_avg', 'L_2nb_sum', 'L_2nb_avg']
            target = 'L' 

            # Train-test split (70/30)
            X_train, X_test, y_train, y_test = train_test_split(df[predictors], df[target], test_size=0.3, random_state=42)

            for num_trees in num_trees_options:
                # Define the parameter grid including max_depth and potentially other parameters
                param_grid = {
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 10, 20],
                    'min_samples_leaf': [1, 4, 10],
                }

                # Create a GridSearchCV object
                grid_search = GridSearchCV(estimator=RandomForestRegressor(n_estimators=num_trees, random_state=42), 
                                           param_grid=param_grid, 
                                           cv=5,  # 5-fold cross-validation
                                           scoring='neg_mean_squared_error')

                # Fit the grid search to the data
                grid_search.fit(X_train, y_train)

                # Get the best model
                best_rf = grid_search.best_estimator_

                # Predict on the test set using the best model
                y_pred = best_rf.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)

                # Output the results
                print(f"True Model: {true_model}, Sample Size: {sample_size}, Number of Trees: {num_trees}, Best Parameters: {grid_search.best_params_}, Best MSE: {mse}")

            print()

if __name__ == "__main__":
    main()
