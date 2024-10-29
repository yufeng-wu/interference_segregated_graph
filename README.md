## Causal Inference With Contagion and Latent Homophily Under Full Interference
### Williams College Computer Science Thesis

Authors: Yufeng Wu, Rohit Bhattacharya

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
    - `raw_output_used_in_thesis/`: raw outputs in csv format from running `{XYZ}_ours.py` and `{XYZ}_autog.py` that are eventually used in my thesis. 
    - `plot_used_in_thesis/`: visualizations of the csv files in `raw_output_used_in_thesis/`. 

`contagion_vs_latent_homophily_test/`: directory that contains code, data, and results for verifying the correctness of our proposed independence tests that distinguishes contagion (represented by an undirected edge, -) and latent homophily (represented by a bidirected edge, <->) across the L, A, and Y layers of the graphical model.
- `data/binary_sample/`: 
    - `network.pkl`: a network in pickle format created by `data_generator_LRT.py`.
    - `5_ind_set.csv`: a maximal 5-hop independent set of the network saved in `network.pkl`. This file is generated from `data_generator_LRT.py`.
    - `BBB_sample.csv`: a dataframe generated using the underlying network in `network.pkl` where all layers (L, A, and Y) are connected by bidirected edges. This file is generated from `data_generator_LRT.py`.
    - `UUU_sample`: a dataframe generated using the underlying network in `network.pkl` where all layers (L, A, and Y) are connected by undirected edges. This file is generated from `data_generator_LRT.py`.
- `data_generator_LRT.py`: code that generates data for our likelihood ratio test. All files generated from this code are then saved to `data/binary_sample/` (when I run this code for my thesis, I simply dragged and dropped the outputs into `data/binary_sample/). 
- `likelihood_ratio_test.py`: code to run our likelihood ratio tests for each layer (L, A, and Y) to determine the presence of contagion (null hypothesis) vs. latent homophily (alternative hypothesis). Outputs are saved in `result/`.
- `visualizer.py`: visualize the csv files in `result/` and save the plot to `result/plot/`.
- `result/`
    - `L_results.csv`: results from `likelihood_ratio_test.py` for the L layer.
    - `A_results.csv`: results from `likelihood_ratio_test.py` for the A layer.
    - `Y_results.csv`: results from `likelihood_ratio_test.py` for the Y layer.
    - `plot/`: folder to save the visualizations of the three csv files in `result/`. 

`infrastructure/`: directory that contains the utility code that is shared by code in `causal_effect_estimation/` and `contagion_vs_latent_homophily_test/`.
-   `data_generator.py`: methods for generating network realizations.
-   `maximal_independence_set.py`: methods for finding n-hop maximal independence sets from a network. 
-   `network_utils.py`: methods for creating random networks and process these graphs (e.g. finding the kth-order neighborhood of a node). 

`scratch/`: directory that contains some python code and jupyter notebooks that I used for quick experiments. Code in this directory are not used in simulation studies. 

### Required Packages

TODO

### Replication
Follow the steps below to replicate our results:
1. install required packages
2. run `contagion_vs_latent_homophily_test/contagiondata_generator_LRT.py` and put the outputs in `contagion_vs_latent_homophily_test/data/binary_sample/`. 
3. run `contagion_vs_latent_homophily_test/likleihood_ratio_test.py` to replicate results for our hypothesis tests.
4. run `causal_effect_estimation/run_all_experiments.py` to replicate results for the causal effect estimation part of the simulation studies. 