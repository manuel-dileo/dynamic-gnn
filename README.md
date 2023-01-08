# Dynamic-gnn
PyTorch Geometric implementation of a dynamic gnn based on Roland framework.

The model is designed for solving link prediction tasks on temporal attributed directed graph.

This project is work in progress.

## Architecture overview
This work is largely inspired by ["ROLAND: Graph Learning Framework for Dynamic Graphs"](https://dl.acm.org/doi/abs/10.1145/3534678.3539300). To have a general idea of how our model works, you can refer to the paper by You et al.

## Embedding update module
You can update the node embeddings along the time snapshosts in different ways. You can use the parameter `update` during the inizialization of the model to decide which kind of embedding update module will act. Below we will refer to $$H_{t}^{(l)}$$ as the node embeddings at gnn-layer $$l$$ and time snapshot $$t$$.
- Setting `update` to `mlp`, node embeddings are updated by a 1-layer MLP; $$\tilde{H}_{t}^{(l)} = MLP(CONCAT(H_{t-1}^{(l)},H_{t}^{(l)}))$$.
- Setting `update` to `gru`, node embeddings are updated by a GRU Cell having $$H_{t-1}^{(l)}$$ as hidden state and $$H_{t}^{(l)}$$ as input.
- Setting `update` to `learnable`, node embeddings are updated using a weighted sum $$\tilde{H}_{t}^{l} = \tau  H_{t-1}^{(l)} + (1-\tau) H_{t}^{(l)}$$ where $$\tau$$ is a learnable parameter. Setting `update` to a number between zero and one results in a constant value for $$\tau$$.

## Running example
A complete running example with datasets, train and test procedures will be available soon.
