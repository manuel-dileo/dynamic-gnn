# Dynamic-gnn
PyTorch Geometric implementation of a dynamic gnn based on Roland framework.

The model is designed for solving link prediction tasks on temporal attributed directed graph.

This project is work in progress.

## Architecture overview
This work is largely inspired by ["ROLAND: Graph Learning Framework for Dynamic Graphs"](https://dl.acm.org/doi/abs/10.1145/3534678.3539300). To have a general idea of how our model works, you can refer to the paper by You et al.

## Embedding update module
You can update the node embeddings along the time snapshosts in different ways. You can use the parameter `update` during the inizialization of the model to decide which kind of embedding update module will act.
