# Williams College Computer Science Thesis

Authors: Yufeng Wu, Rohit Bhattacharya

### Repo Organization
`causal_effect_estimation/`: directory that contains code, data, and results
for verifying the correctness of our estimation strategies as well as demonstrating that autog produces biased results when the model is misspecified.
- `run_all_experiments.py`: master file to run all experiments, i.e., all the `{XYZ}_ours.py` and `{XYZ}_autog.py` files, under `causal_effect_estimation/code/`. The effect of executing this code is to produce a set of csv files saved to `causal_effect_estimatio/result/`. 
- `code/`
    - `autog_estimation_methods.py`: methods that implements the Auto-G paper. 
    - `our_estimation_methods.py`: our methods to estimate network causal effects for the following cases: BBB, BUB, BBU, BUU, UBB, UUB, where the three letters corresponds to what type of edge is present in the graphical model. For example, BUB means that the L layer is connected by bidirected edges, A layer by undirected edges, and Y layer by bidirected edges.  
    - `{XYZ}_ours.py`: the set of experiments that verify the correctness of our proposed estimation strategies. In each program, we generate network realizations following the edge types specified by `{XYZ}` and estimate the average causal effect using our method. With increasing sample size, we compare our methods' estimations with the true causal effect evaluated using the true DGP.
    - `{XYZ}_autog.py`: in each program, we generate network realizations following the edge types specified by `{XYZ}` and estimate the average causal effect using the Auto-G method, regardless of whether the graphical model follows the assumption of the Auto-G method. With increasing sample size, we compare the Auto-G estimation results with the true causal effect evaluated using the true DGP. 
    - `setup.py`: the common set up for all of our `{XYZ}_ours.py` and `{XYZ}_autog.py` experiments, including global variables such as burn-in period and the true parameters of the DGP.
    - `visualizer.py`: processes a csv file in `result/raw_output/` and saves the plot in `result/plot/`.
- `result/`
    - `raw_output/`: raw outputs in csv format from running `{XYZ}_ours.py` and `{XYZ}_autog.py`.
    - `plot`: visualizations of the csv files in `raw_output/`, produced by `visualizer.py`.

`contagion_vs_latent_homophily_test/`: directory that contains code, data, and results for verifying the correctness of our proposed independence tests that distinguishes contagion (represented by an undirected edge, -) and latent homophily (represented by a bidirected edge, <->) across the L, A, and Y layers of the graphical model.

`infrastructure/`: directory that contains the utility code that is shared by code in `causal_effect_estimation/` and `contagion_vs_latent_homophily_test/`.
-   `data_generator.py`: methods for generating network realizations.
-   `maximal_independence_set.py`: methods for finding n-hop maximal independence sets from a network. 
-   `network_utils.py`: methods for creating random networks and process these graphs (e.g. finding the kth-order neighborhood of a node). 
