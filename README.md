# Mixture of Experts Implementation

This repository contains implementations of various approaches to building Mixture of Experts (MoE) models. The implementations are based on PyTorch and include algorithms from the [Shazeer et al. 2017]((https://openreview.net/forum?id=B1ckMDqlg)) on sparse MoE with auxiliary loss, as well as the Load-Balancing Loss (LBL) [(Fedus et al. 2022)]((https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf)) and a notebook for testing.

## Formulas and Explanations

### Classic Mixture of Experts (MoE)

The objective is to construct a mapping:

$$F: \mathcal{X} \rightarrow \mathcal{Y}.$$

The MoE mapping is defined as:

$$F(\mathbf{x}) = \sum_{k=1}^{K} \mathbf{G}(\mathbf{x})_k \cdot F_k(\mathbf{x}), \quad F_k: \mathcal{X} \rightarrow \mathcal{Y}, \quad k=1,\ldots,K$$

where $F_k$ are experts, and $\mathbf{G}(\mathbf{x})$ is the gating function.

### [Noisy Top-K Gating](https://openreview.net/forum?id=B1ckMDqlg)

For balancing experts tokens in training, we use Noisy Top-K Gating:

$$G(\mathbf{x}) = \text{Softmax}(\text{KeepTopK}(H(\mathbf{x}), k))$$

Function $H(\mathbf{x})$ and KeepTopK:

$$
H(\mathbf{x})_i = (\mathbf{W}_g \cdot \mathbf{x})_i + \epsilon_i
$$

$$
\epsilon_i = \text{StandardNormal}() \cdot \text{Softplus}\left((\mathbf{W}_{\text{noise}} \cdot \mathbf{x})_i\right)
$$

$$
\text{KeepTopK}(\mathbf{v}, k)_i =
\begin{cases}
v_i & \text{if } v_i \text{ is in the top } k \text{ elements of } \mathbf{v}. \\
-\infty & \text{otherwise.}
\end{cases}
$$

### [Auxiliary Loss](https://openreview.net/forum?id=B1ckMDqlg)

The main challenge is achieving a uniform token distribution among experts. The Auxiliary Loss is defined as:

$$\text{Auxiliary Loss} = w_{\text{importance}} \cdot \text{CV}(\text{Importance})^2$$

$$
\text{Importance} = \sum_{i} \text{softmax}(\text{token}_i)
$$

where CV is the Coefficient of Variation:

$$\text{Coefficient Variation (CV)} = \frac{\text{standard deviation } (\sigma)}{\text{mean } (\mu)}$$

### [Load-Balancing Loss (LBL)](https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf)
Given $K$ experts indexed by $i$ and a batch $\mathcal{B}$ with $T$ tokens, loss is scaled dot-product $\mathbf{f}$ and $\mathbf{P}$:

$$\text{LBL} = K \sum_{i=1}^{K} f_i \cdot P_i$$

$$
f_i = \dfrac{1}{T} \sum_{\mathbf{x} \in \mathcal{B}} \mathbb{1}\{\arg\max p(\mathbf{x}) = i\}, \quad P_i = \frac{1}{T} \sum_{\mathbf{x} \in \mathcal{B}} p_i(\mathbf{x}),
$$

where $f_i$ is fraction of tokens dispatched to expert $i$, $P_i$ is fraction of probability for expert $i$.

## References

- [Shazeer et al. 2017, Noisy Top-K Gating](https://openreview.net/forum?id=B1ckMDqlg)
- [Fedus et al. 2022, Switch Transformer](https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf)
- [Jiang et al. 2024, Mixtral 8x7B](https://arxiv.org/pdf/2401.04088)
- [Qiu et al. 2025, Global-batch Load-balancing Loss](https://arxiv.org/pdf/2501.11873)
- [Yang et al. 2025, Qwen3](https://arxiv.org/pdf/2505.09388)
