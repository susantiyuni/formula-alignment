# dual_encoder.py
# ======================================================
# Dual Transformer (SBERT) contrastive baseline
# ======================================================

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from sentence_transformers import SentenceTransformer, models
from torch.utils.data import TensorDataset, DataLoader

# =========================
# Project imports 
# =========================
from seed import set_seed
from cache import get_or_cache_embeddings
from metrics import retrieval_metrics

from mathml_parser import parse_mathml_tree, mathml_to_opt
from rep_semantics import build_semantic_texts
from losses import contrastive_loss, intra_modal_loss


# ======================================================
# Config
# ======================================================

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_FILE = "formula-eg-grouped.jsonl"
FOLDS_DIR = "data-cat"

MODEL_NAME = "allenai/scibert_scivocab_uncased"

PROJ_DIM = 256
EPOCHS = 200
BATCH_SIZE = 128


# ======================================================
# Helpers
# ======================================================

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


# ------------------------------------------------------
# Structural linearization (same as BM25 baseline)
# ------------------------------------------------------
def opt_to_sequence(opt):
    seq = [opt["value"]]
    for c in opt["children"]:
        seq += opt_to_sequence(c)
    return seq


def build_structural_texts(data):
    opts = []
    for row in data:
        root = parse_mathml_tree(row["formula"])
        opts.append(mathml_to_opt(root))
    return [" ".join(opt_to_sequence(o)) for o in opts]


# ------------------------------------------------------
# SBERT wrapper
# ------------------------------------------------------
def to_sbert(model_name):
    word_model = models.Transformer(model_name)
    pooling = models.Pooling(word_model.get_word_embedding_dimension())
    return SentenceTransformer(modules=[word_model, pooling])


# ======================================================
# Training per fold
# ======================================================

def train_fold(k, data, struct_texts, semantic_texts):

    print(f"\n========== Fold {k} ==========")

    train_rows = load_jsonl(f"{FOLDS_DIR}/fold_{k}_train.jsonl")
    test_rows  = load_jsonl(f"{FOLDS_DIR}/fold_{k}_val.jsonl")

    item_to_index = {r["item"]: i for i, r in enumerate(data)}

    train_idx = [item_to_index[r["item"]] for r in train_rows]
    test_idx  = [item_to_index[r["item"]] for r in test_rows]

    struct_train = [struct_texts[i] for i in train_idx]
    struct_test  = [struct_texts[i] for i in test_idx]

    sem_train = [semantic_texts[i] for i in train_idx]
    sem_test  = [semantic_texts[i] for i in test_idx]

    # -------------------------
    # Encoders
    # -------------------------
    encoder = to_sbert(MODEL_NAME).to(DEVICE)

    X_struct_train = get_or_cache_embeddings(
        struct_train, encoder, f"cache_struct_train{k}.npy", DEVICE
    )
    X_struct_test = get_or_cache_embeddings(
        struct_test, encoder, f"cache_struct_test{k}.npy", DEVICE
    )

    X_sem_train = get_or_cache_embeddings(
        sem_train, encoder, f"cache_sem_train{k}.npy", DEVICE
    )
    X_sem_test = get_or_cache_embeddings(
        sem_test, encoder, f"cache_sem_test{k}.npy", DEVICE
    )

    # -------------------------
    # Projection heads
    # -------------------------
    proj_struct = nn.Sequential(
        nn.Linear(X_struct_train.size(1), PROJ_DIM),
        nn.ReLU(),
        nn.Linear(PROJ_DIM, PROJ_DIM)
    ).to(DEVICE)

    proj_sem = nn.Sequential(
        nn.Linear(X_sem_train.size(1), PROJ_DIM),
        nn.ReLU(),
        nn.Linear(PROJ_DIM, PROJ_DIM)
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(proj_struct.parameters()) + list(proj_sem.parameters()),
        lr=1e-4
    )

    train_dataset = TensorDataset(X_struct_train, X_sem_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    # -------------------------
    # Training
    # -------------------------
    for epoch in range(EPOCHS):

        proj_struct.train()
        proj_sem.train()

        total_loss = 0

        for xb_s, xb_t in train_loader:

            xb_s = xb_s.to(DEVICE)
            xb_t = xb_t.to(DEVICE)

            optimizer.zero_grad()

            z_s = F.normalize(proj_struct(xb_s), dim=1)
            z_t = F.normalize(proj_sem(xb_t), dim=1)

            loss_cross = contrastive_loss(z_s, z_t)

            if epoch < 5:
                loss = loss_cross
            else:
                loss = (
                    loss_cross +
                    0.05 * intra_modal_loss(z_s) +
                    0.05 * intra_modal_loss(z_t)
                )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d}  Loss={total_loss:.4f}")

    # -------------------------
    # Evaluation
    # -------------------------
    with torch.no_grad():
        Zs = F.normalize(proj_struct(X_struct_test.to(DEVICE)), dim=1)
        Zt = F.normalize(proj_sem(X_sem_test.to(DEVICE)), dim=1)

    metrics_s2t = retrieval_metrics(Zs, Zt)
    metrics_t2s = retrieval_metrics(Zt, Zs)

    print("\nStructural→Semantic:", metrics_s2t)
    print("Semantic→Structural:", metrics_t2s)

    return metrics_s2t, metrics_t2s


# ======================================================
# Main
# ======================================================

if __name__ == "__main__":

    set_seed(SEED)

    data = load_jsonl(DATA_FILE)

    print("Building texts...")
    struct_texts = build_structural_texts(data)
    semantic_texts = build_semantic_texts(data)

    all_s2t, all_t2s = [], []

    for k in ["1", "2", "3", "4", "5"]:
        s2t, t2s = train_fold(k, data, struct_texts, semantic_texts)
        all_s2t.append(s2t)
        all_t2s.append(t2s)

    print("\n========== FINAL ==========")

    def mean(vals): return sum(vals) / len(vals)

    print("Mean MRR S→T:", mean([m["MRR"] for m in all_s2t]))
    print("Mean MRR T→S:", mean([m["MRR"] for m in all_t2s]))
