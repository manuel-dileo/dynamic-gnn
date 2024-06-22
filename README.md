# Dynamic-gnn
PyTorch Geometric implementation of a dynamic gnn based on the Roland framework.

The model is designed for solving link prediction tasks on temporal attributed directed graph. 

The repository contains the work behind the paper ["Temporal Graph Learning for Dynamic Link Prediction with Text in Online Social Networks"](https://doi.org/10.1007/s10994-023-06475-x).

## Architecture overview
The architecture is inspired by ["ROLAND: Graph Learning Framework for Dynamic Graphs"](https://dl.acm.org/doi/abs/10.1145/3534678.3539300). To have a general idea of how our model works, you can refer to the paper by You et al. The figure below shows the running architecture of our dynamic gnn model.
![GNN Architecture](GNNArchitecture.drawio.png "Dynamic GNN based on ROLAND framework").

## Embedding update module
You can update the node embeddings along the time snapshosts in different ways. You can use the parameter `update` during the inizialization of the model to decide which kind of embedding update module will act. Below we will refer to $H_{t}^{(l)}$ as the node embeddings at gnn-layer $l$ and time snapshot $t$.
- Setting `update` to `mlp`, node embeddings are updated by a 1-layer MLP (default option); $H_{t}^{(l)} = MLP(CONCAT(H_{t-1}^{(l)},H_{t}^{(l)}))$.
- Setting `update` to `gru`, node embeddings are updated by a GRU Cell having $H_{t-1}^{(l)}$ as hidden state and $H_{t}^{(l)}$ as input.
- Setting `update` to `lwa` (learnable weighted average), node embeddings are updated using a weighted sum $H_{t}^{(l)} = \tau  H_{t-1}^{(l)} + (1-\tau) H_{t}^{(l)}$ where $\tau$ is a learnable parameter. Setting `update` to a number between zero and one results in a constant value for $\tau$.

## Running example
A complete running example with datasets, train and test procedures is available on `BitcoinOTC-Example.ipynb` notebook. Information about the dataset can be found in ["EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs"](https://arxiv.org/pdf/1902.10191.pdf). I ran the ROLAND-based model on all the 138 snapshots using a constant encoder as node features and a GRU Cell as embedding update module. The results along the snapshots and over time are reported in AUPRC as suggested in ["Evaluating Link Prediction methods"](https://arxiv.org/pdf/1505.04094.pdf). This running example serves also to show the scalability of the proposed solution on a dataset with a high number of snapshots.

## Steemit Data
Due to privacy reasons on personal data like username and textual content, we can't release the dataset related to Steemit. To patch this problem, we provide an anonymized version of our data. This version represents the final mathematical objects that are used to feed the models. For data gathering, you can refer to the [Steemit API](https://developers.steem.io/) documentation.

## Experiments
For the experiments presented in ["Temporal Graph Learning for Dynamic Link Prediction with Text in Online Social Networks"](https://doi.org/10.1007/s10994-023-06475-x), you can refer to the `Steemit.ipynb` notebook. For the experiments concerning scalability and generality of the solution, you can refer to the "Running example" section of this repo, as well as this [other work](https://github.com/manuel-dileo/t3gnn)

## Cite
If you use the code of this repository for your project or you find the work interesting, please cite the following work:
Dileo, M., Zignani, M. & Gaito, S. Temporal graph learning for dynamic link prediction with text in online social networks. Mach Learn 113, 2207–2226 (2024). https://doi.org/10.1007/s10994-023-06475-x

```bibtex
@Article{Dileo2024,
author={Dileo, Manuel
and Zignani, Matteo
and Gaito, Sabrina},
title={Temporal graph learning for dynamic link prediction with text in online social networks},
journal={Machine Learning},
year={2024},
month={Apr},
day={01},
volume={113},
number={4},
pages={2207-2226},
abstract={Link prediction in Online Social Networks---OSNs---has been the focus of numerous studies in the machine learning community. A successful machine learning-based solution for this task needs to (i) leverage global and local properties of the graph structure surrounding links; (ii) leverage the content produced by OSN users; and (iii) allow their representations to change over time, as thousands of new links between users and new content like textual posts, comments, images and videos are created/uploaded every month. Current works have successfully leveraged the structural information but only a few have also taken into account the textual content and/or the dynamicity of network structure and node attributes. In this paper, we propose a methodology based on temporal graph neural networks to handle the challenges described above. To understand the impact of textual content on this task, we provide a novel pipeline to include textual information alongside the structural one with the usage of BERT language models, dense preprocessing layers, and an effective post-processing decoder. We conducted the evaluation on a novel dataset gathered from an emerging blockchain-based online social network, using a live-update setting that takes into account the evolving nature of data and models. The dataset serves as a useful testing ground for link prediction evaluation because it provides high-resolution temporal information on link creation and textual content, characteristics hard to find in current benchmark datasets. Our results show that temporal graph learning is a promising solution for dynamic link prediction with text. Indeed, combining textual features and dynamic Graph Neural Networks---GNNs---leads to the best performances over time. On average, the textual content can enhance the performance of a dynamic GNN by 3.1{\%} and, as the collection of documents increases in size over time, help even models that do not consider the structural information of the network.},
issn={1573-0565},
doi={10.1007/s10994-023-06475-x},
url={https://doi.org/10.1007/s10994-023-06475-x}
}
```


