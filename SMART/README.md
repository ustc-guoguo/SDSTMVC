### Official PyTorch Implementation of SMART

[Liang Peng†, Yixuan Ye†, Cheng Liu\*, Hangjun Che, Fei Wang, Zhiwen Yu, Si Wu, and Hau-San Wong. "SMART: Semantic Matching Contrastive Learning for Partially View-Aligned Clustering". *IEEE TCSVT*](https://ieeexplore.ieee.org/document/11268507). 

#### Abstract

Multi-view clustering has been empirically shown to improve learning performance by leveraging the inherent complementary information across multiple views of data. However, in real-world scenarios, collecting strictly aligned views is challenging, and learning from both aligned and unaligned data becomes a more practical solution. Partially View-aligned Clustering (PVC) aims to learn correspondences between misaligned view samples to better exploit the potential consistency and complementarity across views, including both aligned and unaligned data. However, most existing PVC methods fail to leverage unaligned data to capture the shared semantics among samples from the same cluster. Moreover, the inherent heterogeneity of multi-view data induces distributional shifts in representations, leading to inaccuracies in establishing meaningful correspondences between cross-view latent features and, consequently, impairing learning effectiveness. To address these challenges, we propose a Semantic MAtching contRasTive learning model (SMART) for PVC. The main idea of our approach is to alleviate the influence of cross-view distributional shifts, thereby facilitating semantic matching contrastive learning to fully exploit semantic relationships in both aligned and unaligned data. Specifically, we mitigate view distribution shifts by aligning cross-view covariance matrices, which enables the inference of a semantic graph for all data. Guided by the learned semantic graph, we further exploit semantic consistency across views through semantic matching contrastive learning. After the optimization of the above mechanisms, our model smoothly performs semantic matching for different view embeddings instead of the cumbersome view realignment, which enables the learned representations to enjoy richer category-level semantics and stronger robustness. Extensive experiments on eight benchmark datasets demonstrate that our method consistently outperforms existing approaches on the PVC problem.

#### Requirements

- numpy==1.26.1
- torch==1.12.1+cu116
- tqdm==4.66.1
- logging==0.5.1.2

#### Demo

Train a model with default settings.

```
python run.py
```

