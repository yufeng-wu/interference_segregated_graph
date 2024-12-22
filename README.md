## Network Causal Effect Estimation In Graphical Models Of Contagion And Latent Confounding

This repo contains the experimentation code for our paper "Network Causal Effect Estimation In Graphical Models Of Contagion And Latent Confounding."

---

### Repo Organization
`causal_effect_estimation/`: directory that contains code, data, and results
for verifying the correctness of our estimation strategies as well as demonstrating that autog produces biased results when the model is misspecified.
- `run_all_experiments.py`: master file to run all experiments, i.e., all the `{XYZ}_ours.py` and `{XYZ}_autog.py` files, under `causal_effect_estimation/code/`. The effect of executing this code is to produce a set of csv files saved to `causal_effect_estimatio/result/`. 
- `autog_simulation_manager.py`: master file to run all experiments in `code/` which has the format `{XYZ}_autog.py`.  
- `ours_simulation_manager.py`: master file to run all experiments in `code/` which has the format `{XYZ}_ours.py`.  
- `code/`
    - `autog_estimation_methods.py`: methods that implements the Auto-G paper. 
    - `our_estimation_methods.py`: our methods to estimate network causal effects for the following cases: BBB, BUB, BBU, BUU, UBB, UUB, where the three letters corresponds to what type of edge is present in the graphical model. For example, BUB means that the L layer is connected by bidirected edges, A layer by undirected edges, and Y layer by bidirected edges.  
    - `{XYZ}_ours.py`: the set of experiments that verify the correctness of our proposed estimation strategies. In each program, we generate network realizations following the edge types specified by `{XYZ}` and estimate the average causal effect using our method. With increasing sample size, we compare our methods' estimations with the true causal effect evaluated using the true DGP.
    - `{XYZ}_autog.py`: in each program, we generate network realizations following the edge types specified by `{XYZ}` and estimate the average causal effect using the Auto-G method, regardless of whether the graphical model follows the assumption of the Auto-G method. With increasing sample size, we compare the Auto-G estimation results with the true causal effect evaluated using the true DGP. 
    - `setup.py`: the common set up for all of our `{XYZ}_ours.py` and `{XYZ}_autog.py` experiments, including global variables such as burn-in period and the true parameters of the DGP.
    - `visualizer.py`: processes a csv file in `result/raw_output/` and saves the plot in `result/plot/`.
    - `visualizer_side_by_side.py`: create a visualization that compares the causal effect estimation results using our methods vs. using autog method on the same setup. 
- `result/`
    - `raw_output/`: raw outputs in csv format from running `{XYZ}_ours.py` and `{XYZ}_autog.py`.
    - `plot`: visualizations of the csv files in `raw_output/`, produced by `visualizer.py`.

`contagion_vs_latent_confounding_test(synthetic)/`: directory that contains code, data, and results for verifying the correctness of our proposed independence tests that distinguishes contagion (represented by an undirected edge, -) and latent confounding (represented by a bidirected edge, <->) across the L, A, and Y layers of the partially determined segregated graph. Fully synthetic version.
- `data/binary_sample/`: 
    - `network.pkl`: a network in pickle format created by `data_generator_LRT.py`.
    - `5_ind_set.csv`: a maximal 5-hop independent set (we used the term 6-degree separated set in our paper) of the network saved in `network.pkl`. This file is generated from `data_generator_LRT.py`.
    - `BBB_sample.csv`: a dataframe generated using the underlying network in `network.pkl` where all layers (L, A, and Y) are connected by bidirected edges. This file is generated from `data_generator_LRT.py`.
    - `UUU_sample`: a dataframe generated using the underlying network in `network.pkl` where all layers (L, A, and Y) are connected by undirected edges. This file is generated from `data_generator_LRT.py`.
- `data_generator_LRT.py`: code that generates data for our likelihood ratio test. All files generated from this code are then saved to `data/binary_sample/`. 
- `likelihood_ratio_test.py`: code to run our likelihood ratio tests for each layer (L, A, and Y) to determine the presence of contagion (null hypothesis) vs. latent confounding (alternative hypothesis). Outputs are saved in `contagion_vs_latent_confounding_test/result/`.
- `likelihood_ratio_test_layer_only.py`: almost the same as `likelihood_ratio_test.py` but with small modifications of the conditioning set in the tests. This version is what our paper eventually used. Both are correct LRTs to distinguish between undirected and bidirected edges, but this version is easier to present. 
- `visualizer.py`: visualize the csv files in `result/` and save the plot to `result/plot/`.
- `result/`
    - `L_results.csv`: results from `likelihood_ratio_test.py` for the L layer.
    - `A_results.csv`: results from `likelihood_ratio_test.py` for the A layer.
    - `Y_results.csv`: results from `likelihood_ratio_test.py` for the Y layer.
    - `L_results_layer_only.csv`: results from `likelihood_ratio_test_layer_only.py` for the L layer.
    - `A_results_layer_only.csv`: results from `likelihood_ratio_test_layer_only.py` for the A layer.
    - `Y_results_layer_only.csv`: results from `likelihood_ratio_test_layer_only.py` for the Y layer.
    - `plot/`: folder to save the visualizations of the three csv files in `result/`. 

`contagion_vs_latent_confounding_test(semi-synthetic)/`: directory for verifying the likelihood ratio test using semi-synthetic data (i.e., the network are real networks but the data are generated using our own DGP).
- `raw_data/`: raw data downloaded from SNAP.
    - `RO_edges.csv`: Deezer RO
    - `musae_git_edges.csv`: GitHub Social Networkk
    - `artist_edges.csv`: GEMSEC FB Artists
- `intermediate_data/`: processed raw data, organized into different subfolders. Includes a dictionary version of the original network, a 6-degree separated set, a dataset when three layers are all bidirected, and another dataset when three layers are all undirected.
- `result/`: result of the likelihood ratio tests, organized into different subfolders. 

`infrastructure/`: directory that contains the utility code that is used by programs in `causal_effect_estimation/` or `contagion_vs_latent_confounding_test/`.
-   `data_generator.py`: methods for generating network realizations. Used by programs in `causal_effect_estimation/`.
-   `maximal_independence_set.py`: methods for finding n-hop maximal independence sets from a network (note: we called them n+1 degree separated sets in our paper).
-   `network_utils.py`: methods for creating random networks and process these graphs (e.g. finding the kth-order neighborhood of a node). 

`network_six_degree_separation_experiment/`: directory that contains data and experimentation code to test how large of six degree separated sets we can find in various real social networks. 
-   `data/`: folder containing network data downloaded from https://snap.stanford.edu/data/index.html#socnets
-   `experiment.ipynb`: notebook to run this set of experiments. 
