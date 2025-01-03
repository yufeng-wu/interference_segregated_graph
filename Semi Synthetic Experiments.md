### Additional semi-synthetic experiments with real-world network topologies

We would like to follow-up on our response to reviewers cYdZ and eR2p. As suggested by reviewer eR2p, we conducted semi-synthetic experiments using five real-world networks from Table 1 in our paper. For each network, we generate synthetic data on $L$, $A$, and $Y$ and perform our likelihood ratio tests on these data. Then, we run the following procedure 500 times: 

1. randomly obtain a maximal 6-degree separated set from the network and record the effective sample size
2. perform our test on a dataset where three layers are all bidirected edges
3. perform our test on a dataset where three layers are all undirected edges

We calculate Type I Error rate and Power of our tests for each layer. Our results are summarized in the following tables (in the paper we will put these in as plots similar to our synthetic experiments, but we use tables here since the reviewing guidelines do not permit sharing of images/links).

#### L Layer Experiments: 
| Network | Average Effective Sample Size  | Type I Error | Power |
| -------- | ------- | ------- | ------- |
| LastFM Asia | 139.4 | 0.056 | 0.416 | 
| Deezer HR | 297.7 | 0.046 | 0.49 |
| Deezer Europe | 451.1  | 0.044 | 0.392 |
| Deezer HU | 464.6 | 0.058 | 0.878 | 
| Deezer RO | 1147.0 | 0.050 | 0.998 |

#### A Layer Experiments:
| Network | Average Effective Sample Size  | Type I Error | Power |
| -------- | ------- | ------- | ------- |
| LastFM Asia | 139.4 | 0.058 | 0.198 |
| Deezer HR | 297.7 | 0.028 | 0.358 |
| Deezer Europe | 451.1 | 0.046 | 0.324 |
| Deezer HU | 464.6 | 0.058 | 0.398 |
| Deezer RO | 1147.0 | 0.030 |  0.916 |

#### Y Layer Experiments: 
| Network | Average Effective Sample Size  | Type I Error | Power |
| -------- | ------- | ------- | ------- |
| LastFM Asia | 139.4 | 0.038 | 0.124 |
| Deezer HR | 297.7 | 0.052 | 0.044 |
| Deezer Europe | 451.1 | 0.056 | 0.416 |
| Deezer HU | 464.6 | 0.058 | 0.652 |
| Deezer RO | 1147.0 | 0.038 | 0.872 |

We make the following observations: 
1. Type I error rate is well controlled at the desired significance level ($\alpha = 0.05$) across all networks.
2. For a given layer, the power of our test is generally higher for networks where we are able to obtain larger effective samples. 
3. For a given network, the power of our test is generally the highest at the $L$ layer. Distinguishing latent confounding from contagion is slightly more challenging for the $A$ layer and the most challenging for the $Y$ layer. 

These results are consistent with the synthetic experiment results in Figure 2 of our paper, showing that our likelihood ratio tests work well on a variety of network toplogies. Our synthetic experiments generate networks where each units has 1 to 6 randomly assigned neighbors. In these real-world networks, however, the maximal degree and average degree are much higher. For example, in Deezer HR, the maximum and average degrees are 420 and 18.26; in LastFM Asia, these numbers are 216 and 7.29.

We thank the reviewers for their suggestion; these new experiments strengthen our work by providing more challenging robustness checks for our proposed tests.