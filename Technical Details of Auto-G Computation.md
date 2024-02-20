# Technical Details of Auto-G Computation

Feb.7th, 2024



### Question:

1.   why introduce the exp() term?
2.   why have the yi terms outside in $\sum_{i \in \mathcal{G}_\mathcal{E}} y_i G_i(y_i; \textbf{a}, \textbf{l}) + \sum_{\{i,j\} \in \mathcal{G}_\mathcal{E}} y_i y_j \theta_{ij}(\textbf{a}, \textbf{l})$ ?
3.   how to get the true causal effects from the DGP? Do i just run the gibbs sampler and set everybody's A to 1?
4.   in the auto-gaussian model, is 2 * mu a typo? shouldn't it be yi - mu squared?
5.   why don't we just directly estimate f(Y | stuff) using ML? why do we have to go through all these?

-   binary case is fine. continuous case is not fine because we nee the whole distribution.



**<u>new questions:</u>**

1.   page 16: why is log(p/p) = a linear comb?
2.   page 16: why do we need the normalizing weight term? is it necessary? isn't it misspecification?



### Set up:

We want to estimate the following probability density function:
$$
f(Y_i = y_i \mid \textbf{Y}_{-i}=\textbf{y}_{-i}, \textbf{a}, \textbf{l}) 
 
 =^{(1)} \frac{exp(U(\textbf{y};\textbf{a}, \textbf{l}))}{\sum_{\textbf{y}':\textbf{y}'_{-i}=\textbf{y}_{-i}}exp(U(\textbf{y};\textbf{a}, \textbf{l}))}
$$
(1): by bayes rule and the factorization $P(B | pa(B))$ in chain graphs. 



This PDF is available to us once we know $U(\textbf{y};\textbf{a}, \textbf{l})$, which is a conditional energy function that can be decomposed into a **sum** (sum, instead of product, because of the exp() term) of terms called conditional clique potentials, with a term for every maximal clique in the graph $\mathcal{G}_\mathcal{E}^a$ (that involves some $Y_i \in \textbf{Y}$, i.e., can't only involve $\textbf{a}, \textbf{l}$ because they are $pa(\textbf{y})$). 

$U(\textbf{y};\textbf{a}, \textbf{l})$ is a function that takes inputs $\textbf{y}, \textbf{a}, \textbf{l}$  and outputs a positive real number. 

Therefore, 
$$
\begin{align*}
U(\textbf{y};\textbf{a}, \textbf{l}) &=^{(2)} \sum_{c \in C(\mathcal{G}_\mathcal{E}^a)}U_c(\textbf{y}; \textbf{a}, \textbf{l}) \\
&=^{(3)} \sum_{i \in \mathcal{G}_\mathcal{E}} y_i G_i(y_i; \textbf{a}, \textbf{l}) + \sum_{\{i,j\} \in \mathcal{G}_\mathcal{E}} y_i y_j \theta_{ij}(\textbf{a}, \textbf{l})

\end{align*}
$$
(2): by definition / factorization of chain graph. $U_c(\textbf{y}; \textbf{a}, \textbf{l})$ denotes a clique potential of the clique $c$.

(3): by the following regularizations: 

1.   only cliques of size 1 or 2 (cliques that contain a single $Y_i \in \textbf{Y}$ or two $Y_i, Y_j \in \textbf{Y}$ ) have non-zero potential function $U_c(Y_i; A_i, L_i, A_s, L_s)$ for all $s \in \mathcal{N}_i$, or $U_c(Y_i, Y_j; A_i, L_i, A_j, L_j, A_s, L_s)$ for all $s \in \mathcal{N}_i \cap \mathcal{N}_j$. 
2.   the conditional probability $f(Y_i = y_i \mid \textbf{Y}_{-i}=\textbf{y}_{-i}, \textbf{a}, \textbf{l}) $ have exponential family form.

notes: G(yi; a, l) = this distribution takes in yi and outputs a probbility measure and the shape of that distribution depends on a, l .

### Parametric assumptions:
$$
\begin{align*}
G_i(y_i; \textbf{a}, \textbf{l}) &= - \left( \frac{1}{2\sigma^2_y} \right) (y_i - 2\mu_{y,i}(\textbf{a},\textbf{1})); \\
\mu_{y,i}(\textbf{a},\textbf{1}) &= \beta_0 + \beta_1 a_i + \beta'_2 l_i + \beta_3 \sum_{j \in \mathcal{N}_i} w_{ij}^a a_j + \beta'_4 \sum_{j \in \mathcal{N}_i} w_{ij}^l l_j; \\
\theta_{ij} &= w_{ij}^{\theta}\theta,
\end{align*}
$$


where $\mu_{y,i}(\textbf{a},\textbf{l}) = E(Y_i|\textbf{a},\textbf{l}, \textbf{Y}_{-i} = 0)$, and $\sigma^2_y = \text{var}(Y_i|\textbf{a},\textbf{l}, \textbf{Y}_{-i} = 0)$. 

Model parameters $\tau_Y = (\beta_0, \beta_1, \beta'_2, \beta_3, \beta'_4, \sigma^2_y, \theta)$ are shared across units in the network.


### Simplified representation of the distribution:

$f(Y_i = y_i \mid \textbf{Y}_{-i}=\textbf{y}_{-i}, \textbf{a}, \textbf{l}) = f(Y_i = y_i \mid O_i, A_i, L_i; \tau_Y)$, where $\tau_Y$ indicates that the parameters of this distribution following the parametric assumptions listed before is $\tau_Y$. To prove that this is true:

$$
\begin{align*}
&f(Y_k = y_k \mid \textbf{Y}_{-k}=\textbf{y}_{-k}, \textbf{a}, \textbf{l}) \\

&= \frac{exp(U(\textbf{y};\textbf{a}, \textbf{l}))}{\sum_{\textbf{y}':\textbf{y}'_{-k}=\textbf{y}_{-k}}exp(U(\textbf{y};\textbf{a}, \textbf{l}))} \\

&= \frac{exp( \sum_{i \in \mathcal{G}_\mathcal{E}} y_i G_i(y_i; \textbf{a}, \textbf{l}) + \sum_{\{i,j\} \in \mathcal{G}_\mathcal{E}} y_i y_j \theta_{ij}(\textbf{a}, \textbf{l}))}{\sum_{\textbf{y}':\textbf{y}'_{-k}=\textbf{y}_{-k}}exp( \sum_{i \in \mathcal{G}_\mathcal{E}} y_i G_i(y_i; \textbf{a}, \textbf{l}) + \sum_{\{i,j\} \in \mathcal{G}_\mathcal{E}} y_i y_j \theta_{ij}(\textbf{a}, \textbf{l}))} \\

&= \frac{
        exp(
            y_k G_k(y_k; \textbf{a}, \textbf{l}) + 

            \sum_{i \in \mathcal{G}_\mathcal{E} \setminus k} y_i G_i(y_i; \textbf{a}, \textbf{l}) + 

            \sum_{\{k,j\} \in \mathcal{G}_\mathcal{E}} y_k y_j \theta_{kj}(\textbf{a}, \textbf{l} + 
    
            \sum_{\{i,j\} \in \mathcal{G}_\mathcal{E} \text{ s.t. } i \not= j \not= k} y_i y_j \theta_{ij}(\textbf{a}, \textbf{l}))
        )
    }
    {
        \sum_{\textbf{y}':\textbf{y}'_{-k}=\textbf{y}_{-k}} 
        
        exp(
            y_k G_k(y_k; \textbf{a}, \textbf{l}) + 

            \sum_{i \in \mathcal{G}_\mathcal{E} \setminus k} y_i G_i(y_i; \textbf{a}, \textbf{l}) + 

            \sum_{\{k,j\} \in \mathcal{G}_\mathcal{E}} y_k y_j \theta_{kj}(\textbf{a}, \textbf{l} + 
    
            \sum_{\{i,j\} \in \mathcal{G}_\mathcal{E} \text{ s.t. } i \not= j \not= k} y_i y_j \theta_{ij}(\textbf{a}, \textbf{l}))
        )
    } \\

&=^{(4)} \frac{
        exp(
            y_k G_k(y_k; \textbf{a}, \textbf{l}) + 

            \sum_{\{k,j\} \in \mathcal{G}_\mathcal{E}} y_k y_j \theta_{kj}(\textbf{a}, \textbf{l})
        )
    }
    {
        \sum_{\textbf{y}':\textbf{y}'_{-k}=\textbf{y}_{-k}} 
        
        exp(
            y_k G_k(y_k; \textbf{a}, \textbf{l}) + 

            \sum_{\{k,j\} \in \mathcal{G}_\mathcal{E}} y_k y_j \theta_{kj}(\textbf{a}, \textbf{l})
        )
    } \\

\end{align*}
$$

(4): all the terms that does not depend on $y_k$ can be moved outside the summation sign in the denominator, therefore they can be cancelled with the same terms in the numerator. 

In addition, in order to be consistent with local Markov conditions, the shape of the function $G_k(y_k; \textbf{a}, \textbf{l})$ can only depend on $\{(a_s, l_s):s \in \mathcal{N}_k\}$ and $\theta_{kj}(\textbf{a}, \textbf{l})$ can depend at most on $\{(a_s, l_s):s \in \mathcal{N}_k \cap \mathcal{N}_j\}$. 

Therefore, the distribution that we're trying to estimate can be simplified as follows: $f(Y_i = y_i \mid \textbf{Y}_{-i}=\textbf{y}_{-i}, \textbf{a}, \textbf{l}) = f(Y_i = y_i \mid O_i, A_i, L_i; \tau_Y)$.



### L layer

$$
f(\textbf{l}) 

=^{(a)} \frac{exp(W(\textbf{l}))}{\sum_{\textbf{l}'}exp(W(\textbf{l}'))}
$$

(a): $W(\textbf{l})$ is an energy function which can be decomposed as as sum over cliques in the induced undirected graph $(\mathcal{G}_\mathcal{E})_{\textbf{L}}$.

$$
f(l_i \mid \textbf{l}_{-i}) 

=^{(b)} \frac{exp(L_i * H_i(L_i) + \sum_{\{i,j\} \in \mathcal{G}_\mathcal{E}} w_{i,j}^L L_i L_j)}{\sum_{\textbf{l}':\textbf{l}'_{-k}=\textbf{l}_{-k}}exp(L_i' * H_i(L_i') + \sum_{\{i,j\} \in \mathcal{G}_\mathcal{E}} w_{i,j}^L L_i' L_j)}
$$

(b): Apply bayes rule, then cancel out the common $\sum_{\textbf{l}'}exp(W(\textbf{l}'))$ which does not depend on $i$. Then, apply regularization similarly to how we specify the form of $f(Y|-)$. The simplification from $W(\textbf{l})$ to $L_i * H_i(L_i) + \sum_{\{i,j\} \in \mathcal{G}_\mathcal{E}} w_{i,j}^L L_i L_j$. **Note: the original paper's form is not exactly this as it considers $L$ to be a vector of confounders. Details are at page 15 of the paper.**




### Estimation Process:

We have specified the **parametric form / functional shape** of the distribution $f(Y_i = y_i \mid \textbf{Y}_{-i}=\textbf{y}_{-i}, \textbf{a}, \textbf{l})$. Therefore, we can estimate the parameters $\tau_Y$ using coding-type maximum likelihood. That is, we are trying to find the parameters $\hat{\tau_Y}$ that maximizes the likelihood of getting $Y_i$ in our sample given $\textbf{Y}_{-i}=\textbf{y}_{-i}, \textbf{a}, \textbf{l}$ (actually, given $O_i, a_i, l_i$).

### How is the estimated distribution then used:

Once $\hat{\tau_Y}$ is obtained, we then have our estimate of the distribution $f(Y_i = y_i \mid \textbf{Y}_{-i}=\textbf{y}_{-i}, \textbf{a}, \textbf{l})$, which is a conditional distribution that we can sample from (in our Gibbs sampler algorithm to obtain network causal effects).



