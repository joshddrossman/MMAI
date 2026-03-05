# models/graph_encoder_gcn.py
# Lightweight GCN encoder for OR problems: global pooling (mean + attention-ish option later).
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, global_mean_pool
except Exception as e:
    raise ImportError("Please install torch-geometric to use this encoder.") from e


@dataclass
class GCNConfig:
    in_dim: int
    hidden_dim: int = 128
    out_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.2
    use_edge_weight: bool = False  # set True if you want to feed edge_attr as weights (careful)


class GraphEncoderGCN(nn.Module):
    def __init__(self, cfg: GCNConfig):
        super().__init__()
        self.cfg = cfg

        dims = [cfg.in_dim] + [cfg.hidden_dim] * (cfg.num_layers - 1) + [cfg.out_dim]
        self.convs = nn.ModuleList([GCNConv(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for d in dims[1:]])

    def forward(self, data) -> torch.Tensor:
        """
        data: PyG Batch with fields x, edge_index, (optional edge_attr), batch
        returns: [B, out_dim]
        """
        x = data.x
        edge_index = data.edge_index

        edge_weight = None
        if self.cfg.use_edge_weight and hasattr(data, "edge_attr") and data.edge_attr is not None:
            # edge_attr shape [E,1]; use as edge_weight (must be non-negative for classic GCN)
            # We take abs to be safe; you can do signed variants later.
            edge_weight = data.edge_attr.view(-1).abs()

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.cfg.dropout, training=self.training)

        g = global_mean_pool(x, data.batch)  # [B, out_dim]
        return g