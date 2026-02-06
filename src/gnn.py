# gnn.py
# =========================================================
# Graph Neural Network encoders for formula graphs
# =========================================================

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool


# =========================================================
# FormulaGINE_Sem
# =========================================================
class FormulaGINE_Sem(nn.Module):
    """
    Semantic-aware GINE encoder.

    Node input:
        data.x      : [N, 2]
            col0 = label id
            col1 = node type id

        data.x_sem  : [N, 2*sem_dim]
            concatenation of:
                label embedding
                symbol description embedding

    Edge input:
        data.edge_attr : [E, 2]
            col0 = role id
            col1 = child position index

    Output:
        graph embedding [B, out_dim]
    """

    def __init__(
        self,
        n_labels,
        hidden=256,
        out_dim=256,
        node_type_dim=3,
        edge_role_dim=8,
        max_child_idx=200,
        node_sem_dim=768,
        num_layers=2,
        gate=False
    ):
        super().__init__()

        self.hidden = hidden
        self.gate = gate

        # -------------------------------------------------
        # Node embeddings
        # -------------------------------------------------
        self.label_emb = nn.Embedding(n_labels, hidden)
        self.type_emb  = nn.Embedding(node_type_dim, hidden)

        # semantic projection (label + desc)
        self.sem_proj = nn.Linear(node_sem_dim * 2, hidden)

        # -------------------------------------------------
        # Edge embeddings
        # -------------------------------------------------
        self.edge_role_emb = nn.Embedding(edge_role_dim, hidden)
        self.child_pos_emb = nn.Embedding(max_child_idx, hidden)

        # -------------------------------------------------
        # GINE layers
        # -------------------------------------------------
        self.convs = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden)
            )
            self.convs.append(GINEConv(mlp))

        # -------------------------------------------------
        # Output projection
        # -------------------------------------------------
        self.proj = nn.Linear(hidden, out_dim)

        # optional semantic gate
        if gate:
            self.gate_proj = nn.Linear(node_sem_dim * 2, hidden)

    # =====================================================
    # forward
    # =====================================================
    def forward(self, data):
        """
        Returns graph embeddings [batch, out_dim]
        """

        # -------------------------
        # node features
        # -------------------------
        x_label = self.label_emb(data.x[:, 0])
        x_type  = self.type_emb(data.x[:, 1])
        x_sem   = self.sem_proj(data.x_sem)

        if self.gate:
            g = torch.sigmoid(self.gate_proj(data.x_sem))
            x_sem = x_sem * g

        x = x_label + x_type + x_sem

        # -------------------------
        # edge features
        # -------------------------
        role = self.edge_role_emb(data.edge_attr[:, 0].long())

        child_idx = torch.clamp(
            data.edge_attr[:, 1].long(),
            max=self.child_pos_emb.num_embeddings - 1
        )
        child = self.child_pos_emb(child_idx)

        edge_attr = role + child

        # -------------------------
        # GINE message passing
        # -------------------------
        for conv in self.convs:
            x = conv(x, data.edge_index, edge_attr)

        # -------------------------
        # graph pooling
        # -------------------------
        batch = getattr(
            data,
            "batch",
            torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        )

        x = global_mean_pool(x, batch)

        return self.proj(x)
