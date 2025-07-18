# Mixture of Experts Implementation

This repository contains implementations of various approaches to building Mixture of Experts (MoE) models. The implementations are based on PyTorch and include algorithms from the Shazeer et al. 2017 on sparse MoE with auxiliary loss, as well as the Load-Balancing Loss (LBL) (Fedus et al. 2022) and a notebook for testing.

## About

In this repository, we have implemented the following:

- Sparse MoE algorithms from the Shazeer et al. 2017 with auxiliary loss.
- Load-Balancing Loss (LBL) for better token distribution among experts.
- A Jupyter notebook for testing the implementations.

The implementations are done using PyTorch.

## Formulas and Explanations

### Classic Mixture of Experts (MoE)

The objective is to construct a mapping:

$$f: \mathcal{X} \rightarrow \mathcal{Y}.$$

The MoE mapping is defined as:

$$f(\mathbf{x}) = \sum_{i=k}^{K} \mathbf{G}(\mathbf{x})_k \cdot f_k(\mathbf{x}), \quad f_k: \mathcal{X} \rightarrow \mathcal{Y}, \quad k=1,\ldots,K$$

where $f_k$ are experts, and $\mathbf{G}(\mathbf{x})$ is the gating function.

### Noisy Top-K Gating

For balancing experts tokens in training, we use Noisy Top-K Gating:

$$G(\mathbf{x}) = \text{Softmax}(\text{KeepTopK}(H(\mathbf{x}), k))$$

where:

$$
H(\mathbf{x})_i = (\mathbf{W}_g \cdot \mathbf{x})_i + \epsilon_i
$$

$$
\epsilon_i = \text{StandardNormal}() \cdot \text{Softplus}\left((\mathbf{W}_{\text{noise}} \cdot \mathbf{x})_i\right)
$$

### Auxiliary Loss

The main challenge is achieving a uniform token distribution among experts. The Auxiliary Loss is defined as:

$$\text{Auxiliary Loss} = w_{\text{importance}} \cdot \text{CV}(\text{Importance})^2$$

where CV is the Coefficient of Variation:

$$\text{Coefficient Variation (CV)} = \frac{\text{standard deviation } (\sigma)}{\text{mean } (\mu)}$$

### Load-Balancing Loss (LBL)

$$\text{LBL} = K \sum_{i=1}^{K} f_i \cdot P_i$$

## References

- [Shazeer et al. 2017, Noisy Top-K Gating](https://openreview.net/forum?id=B1ckMDqlg)
- [Fedus et al. 2022, Switch Transformer](https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf)
- [Jiang et al. 2024, Mixtral 8x7B](https://arxiv.org/pdf/2401.04088)
- [Qiu et al. 2025, Global-batch Load-balancing Loss](https://arxiv.org/pdf/2501.11873)
- [Yang et al. 2025, Qwen3](https://arxiv.org/pdf/2505.09388)
