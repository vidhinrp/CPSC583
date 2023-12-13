from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCN2Conv, GCNConv, Linear,  PNAConv


class GCN(nn.Module):
    def __init__(self,
                 in_features: int,
                 hidden_dims: list[int], dropout: float=0.):
        super(GCN, self).__init__()
        self.dropout = dropout
        dims = [in_features] + hidden_dims
        gcn_layers = []
        for i in range(len(hidden_dims) - 1):
            gcn_layers.append(GCNConv(in_channels=dims[i],
                                      out_channels=dims[i + 1]))

        gcn_layers.append(GCNConv(in_channels=dims[-2], out_channels=dims[-1]))
        self.gcn_layers = nn.ModuleList(gcn_layers)

    def forward(self,
                x: torch.Tensor,
                edge_index: Union[torch.Tensor, list[torch.Tensor]],
                ) -> torch.Tensor:
        layerwise_adjacency = type(edge_index) == list

        for i, layer in enumerate(self.gcn_layers[:-1], start=1):
            edges = edge_index[-i] if layerwise_adjacency else edge_index
            x = torch.relu(layer(x, edges))
            x = F.dropout(x, p=self.dropout, training=self.training)

        edges = edge_index[0] if layerwise_adjacency else edge_index
        node_embeddings = x
        logits = self.gcn_layers[-1](x, edges)
        logits = F.dropout(logits, p=self.dropout, training=self.training)

        # torch.cuda.synchronize()
        memory_alloc = torch.cuda.memory_allocated() / (1024 * 1024)

        return logits, memory_alloc, node_embeddings

