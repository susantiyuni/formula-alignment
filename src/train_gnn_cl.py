# train_gnn.py
# ======================================================
# Graph–Semantic Contrastive Training 
# ======================================================

import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer
from torch_geometric.loader import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score


# =========================
# Project imports
# =========================
from utils import set_seed
from utils import get_or_cache_embeddings
from utils import retrieval_metrics

from mathml-parser import parse_mathml_tree, mathml_to_opt

from rep_semantics import build_semantic_texts
from rep_graph import opt_to_pyg_graph_sem

from gnn import FormulaGINE_Sem
from losses import contrastive_loss, intra_modal_loss


# ======================================================
# Config
# ======================================================

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_FILE = "formula-eg-grouped.jsonl"
FOLDS_DIR = "data-cat"

BATCH_SIZE = 128
EPOCHS = 200

GNN_HIDDEN = 768
GNN_OUT = 512
PROJ_DIM = 256

TEMP_START = 0.1
TEMP_END = 0.05


# ======================================================
# Helpers
# ======================================================

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def get_temp(epoch):
    return TEMP_START - (TEMP_START - TEMP_END) * epoch / EPOCHS


def embed_structural(loader, gnn_model, struct_proj):
    gnn_model.eval()
    Z = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            z = gnn_model(batch)
            z = struct_proj(z)
            z = F.normalize(z, dim=1)
            Z.append(z.cpu())

    return torch.cat(Z, dim=0)


def embed_semantic(X, sem_proj):
    with torch.no_grad():
        X = X.to(DEVICE)
        z = sem_proj(X)
        z = F.normalize(z, dim=1)

    return z.cpu()


def fuse_embeddings(Z_struct, Z_sem, alpha=0.5):
    Z = alpha * Z_sem + (1 - alpha) * Z_struct
    return Z / Z.norm(dim=1, keepdim=True)


# ======================================================
# Build dataset once
# ======================================================

def build_graph_dataset(data, sem_model):
    """
    Parse MathML + build graphs once for entire dataset.
    """

    print("Building graphs...")

    opts = []
    labels = set()

    for row in data:
        root = parse_mathml_tree(row["formula"])
        opt = mathml_to_opt(root)
        opts.append(opt)

        def collect(o):
            labels.add(o["value"])
            for c in o["children"]:
                collect(c)
        collect(opt)

    label_to_id = {l: i for i, l in enumerate(sorted(labels))}

    label_texts = list(label_to_id.keys())
    label_embs = sem_model.encode(label_texts, normalize_embeddings=True)
    label_embs = torch.tensor(label_embs, device=DEVICE)

    graphs = [
        opt_to_pyg_graph_sem(
            opt,
            label_to_id,
            idx=i,
            label_embeddings=label_embs,
            row=data[i],
            sem_model=sem_model,
            device=DEVICE
        )
        for i, opt in enumerate(opts)
    ]

    return graphs, label_to_id


# ======================================================
# Training loop
# ======================================================

def train_fold(k, data, graphs, X_sem_vec):

    print(f"\n================ Fold {k} ================\n")

    train_rows = load_jsonl(f"{FOLDS_DIR}/fold_{k}_train.jsonl")
    test_rows  = load_jsonl(f"{FOLDS_DIR}/fold_{k}_val.jsonl")

    item_to_index = {r["item"]: i for i, r in enumerate(data)}

    train_idx = [item_to_index[r["item"]] for r in train_rows]
    test_idx  = [item_to_index[r["item"]] for r in test_rows]

    graphs_train = [graphs[i] for i in train_idx]
    graphs_test  = [graphs[i] for i in test_idx]

    X_sem_train = X_sem_vec[train_idx]
    X_sem_test  = X_sem_vec[test_idx]

    train_loader = DataLoader(graphs_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(graphs_test, batch_size=1, shuffle=False)

    # =========================
    # Model
    # =========================

    gnn_model = FormulaGINE_Sem(
        n_labels=len(label_to_id),
        hidden=GNN_HIDDEN,
        out_dim=GNN_OUT
    ).to(DEVICE)

    struct_proj = nn.Sequential(
        nn.Linear(GNN_OUT, PROJ_DIM),
        nn.ReLU(),
        nn.Linear(PROJ_DIM, PROJ_DIM)
    ).to(DEVICE)

    sem_proj = nn.Sequential(
        nn.Linear(X_sem_vec.shape[1], PROJ_DIM),
        nn.ReLU(),
        nn.Linear(PROJ_DIM, PROJ_DIM)
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(gnn_model.parameters()) +
        list(struct_proj.parameters()) +
        list(sem_proj.parameters()),
        lr=1e-4
    )

    best_score = -1
    best_Z_struct = None
    best_Z_sem = None

    # =========================
    # Training
    # =========================

    for epoch in range(EPOCHS):

        gnn_model.train()
        total_loss = 0

        for batch in train_loader:

            batch = batch.to(DEVICE)
            optimizer.zero_grad()

            z_struct = struct_proj(gnn_model(batch))

            batch_ids = batch.idx.squeeze()
            z_sem = sem_proj(X_sem_vec[batch_ids].to(DEVICE))

            z_struct = F.normalize(z_struct, dim=1)
            z_sem = F.normalize(z_sem, dim=1)

            loss_cross = contrastive_loss(z_struct, z_sem)

            if epoch < 5:
                loss = loss_cross
            else:
                loss = (
                    loss_cross +
                    0.05 * intra_modal_loss(z_struct) +
                    0.05 * intra_modal_loss(z_sem)
                )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1:03d}  Loss={total_loss:.4f}")

        # =========================
        # Validation
        # =========================

        Z_struct = embed_structural(test_loader, gnn_model, struct_proj)
        Z_sem = embed_semantic(X_sem_test, sem_proj)

        metrics = retrieval_metrics(Z_struct, Z_sem)
        score = metrics["recall"][10]

        if score > best_score:
            best_score = score
            best_Z_struct = Z_struct
            best_Z_sem = Z_sem
            print("★ New best:", metrics)

    # =========================
    # Clustering
    # =========================

    cats = [r["category"] for r in test_rows]
    cat_to_id = {c: i for i, c in enumerate(sorted(set(cats)))}
    y_test = np.array([cat_to_id[c] for c in cats])

    Z_joint = fuse_embeddings(best_Z_struct, best_Z_sem)
    kmeans = KMeans(n_clusters=len(cat_to_id), random_state=SEED).fit(Z_joint)

    nmi = normalized_mutual_info_score(y_test, kmeans.labels_)

    print("Fold NMI:", nmi)

    return metrics, nmi


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    set_seed(SEED)

    data = load_jsonl(DATA_FILE)

    # ---------- semantic texts ----------
    sem_texts = build_semantic_texts(data)

    sem_encoder = SentenceTransformer("all-mpnet-base-v2").to(DEVICE)

    X_sem_vec = get_or_cache_embeddings(
        sem_texts,
        sem_encoder,
        "cache_global_sem.npy",
        DEVICE
    )

    # ---------- graphs ----------
    graphs, label_to_id = build_graph_dataset(data, sem_encoder)

    all_metrics = []
    all_nmi = []

    for k in ["1", "2", "3", "4", "5"]:
        metrics, nmi = train_fold(k, data, graphs, X_sem_vec)
        all_metrics.append(metrics)
        all_nmi.append(nmi)

    print("\n========== FINAL ==========")
    print("Mean NMI:", np.mean(all_nmi), "±", np.std(all_nmi))
