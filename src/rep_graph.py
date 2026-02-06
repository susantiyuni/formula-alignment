import torch
from torch_geometric.data import Data

from .semantic import get_symbol_description


NODE_TYPE_TO_ID = {"sym": 0, "op": 1, "func": 2}

EDGE_ROLE_TO_ID = {
    "child": 0,
    "base": 1,
    "exp": 2,
    "num": 3,
    "den": 4,
    "sub": 5,
    "sup": 6,
    "arg": 7
}


# =========================================================
# OPT → PyG graph with semantic-aware nodes
# =========================================================
def opt_to_pyg_graph_sem(
    opt,
    label_to_id,
    idx,
    label_embeddings,
    row,
    sem_model,
    device="cpu"
):
    """
    Converts OPT → torch_geometric.data.Data

    Node features:
        - label id
        - node type id
        - semantic embedding (label emb + description emb)
    """

    nodes, node_types, node_sem = [], [], []
    edges, edge_types = [], []

    emb_dim = label_embeddings.size(1)

    def build(node, parent_id=None, edge_role="child", child_idx=0):
        node_id = len(nodes)

        label = node["value"]
        label_id = label_to_id[label]

        nodes.append(label)
        node_types.append(NODE_TYPE_TO_ID.get(node["type"], 2))

        # -------------------
        # semantic embedding
        # -------------------
        sym_desc = get_symbol_description(row, label)

        if sym_desc:
            desc_emb = sem_model.encode(
                sym_desc,
                convert_to_tensor=True,
                device=device
            )
        else:
            desc_emb = torch.zeros(emb_dim, device=device)

        sem_node = torch.cat(
            [label_embeddings[label_id], desc_emb],
            dim=0
        )
        node_sem.append(sem_node)

        # -------------------
        # edges
        # -------------------
        if parent_id is not None:
            edges.append([parent_id, node_id])
            role_id = EDGE_ROLE_TO_ID.get(edge_role, EDGE_ROLE_TO_ID["arg"])
            edge_types.append([role_id, child_idx])

        # -------------------
        # child roles
        # -------------------
        if label == "power":
            roles = ["base", "exp"]
        elif label == "frac":
            roles = ["num", "den"]
        elif label == "subscript":
            roles = ["sub", "arg"]
        elif label == "subsup":
            roles = ["sub", "sup", "arg"]
        else:
            roles = ["arg"] * len(node["children"])

        for i, c in enumerate(node["children"]):
            role = roles[i] if i < len(roles) else "arg"
            build(c, node_id, role, i)

    build(opt)

    # -------------------------------------------------
    # Build PyG tensors
    # -------------------------------------------------
    x_label = torch.tensor([[label_to_id[n]] for n in nodes], dtype=torch.long)
    x_type  = torch.tensor([[t] for t in node_types], dtype=torch.long)
    x = torch.cat([x_label, x_type], dim=1)

    x_sem = torch.stack(node_sem, dim=0)

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_types, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 2), dtype=torch.long)

    g = Data(x=x, x_sem=x_sem, edge_index=edge_index, edge_attr=edge_attr)
    g.idx = torch.tensor([idx], dtype=torch.long)

    return g
