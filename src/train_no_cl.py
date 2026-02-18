import os
import math
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lxml import etree
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from sentence_transformers import SentenceTransformer, models
import argparse


# =========================================================
# Setup
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =========================================================
# Argument parsing
# =========================================================

parser = argparse.ArgumentParser(description="Run baseline alignment experiments.")

parser.add_argument(
    "--baseline_type",
    type=str,
    default="dense_noalign",
    choices=["dense_noalign", "dual_encoder", "simcse"],
    help="Baseline type to run."
)
parser.add_argument(
    "--encoder_type",
    type=str,
    default="sbert",
    choices=["sbert", "mathbert"],
    help="Encoder backbone to use."
)

args = parser.parse_args()
BASELINE_TYPE = args.baseline_type
print(f"\nUsing baseline: {BASELINE_TYPE}")
# BASELINE_TYPE = "dense_noalign"  # dual_encoder | simcse | dense_noalign
ENCODER_TYPE = args.encoder_type
if ENCODER_TYPE == "sbert":
    MODEL_NAME = "all-mpnet-base-v2"
elif ENCODER_TYPE == "mathbert":
    MODEL_NAME = "tbs17/MathBERT"

print(f"Using encoder: {ENCODER_TYPE} ({MODEL_NAME})")

DATA_FILE = "data/formula-eg.jsonl"
FOLDS_DIR = "data"
EMB_DIM = 256
EPOCHS = 200
BATCH_SIZE = 128
TEMP = 0.07

# =========================================================
# Data
# =========================================================

with open(DATA_FILE) as f:
    data = [json.loads(line) for line in f]


# =========================================================
# MathML → OPT
# =========================================================

def parse_mathml_tree(mathml_str):
    try:
        root = etree.fromstring(mathml_str.encode("utf8"))
        return root
    except Exception:
        parser = etree.HTMLParser()
        tree = etree.fromstring(mathml_str.encode("utf8"), parser=parser)
        node = tree.find(".//math")
        if node is None:
            raise ValueError("No <math> node found")
        return node


def mathml_to_opt(node):
    tag = etree.QName(node.tag).localname
    children = [c for c in node if isinstance(c.tag, str)]

    if tag in ("mi", "mn", "mtext"):
        return {"type": "sym", "value": (node.text or "").strip(), "children": []}
    if tag == "mo":
        return {"type": "op", "value": (node.text or "").strip(), "children": []}

    return {
        "type": "func",
        "value": tag,
        "children": [mathml_to_opt(c) for c in children],
    }


# =========================================================
# Text building
# =========================================================

def opt_to_sequence(opt):
    seq = [opt["value"]]
    for c in opt["children"]:
        seq += opt_to_sequence(c)
    return seq


def get_descriptions(row):
    texts = []

    for lbl, desc in zip(row.get("conceptLabels", []),
                         row.get("conceptDescriptions", [])):
        if lbl:
            texts.append(lbl.strip())
        if desc:
            texts.append(desc.strip())

    if row.get("itemDescription"):
        texts.append(row["itemDescription"].strip())
    if row.get("itemLabel"):
        texts.append(row["itemLabel"].strip())

    return list(set(texts))


# =========================================================
# Metrics
# =========================================================

def retrieval_metrics(Z_query, Z_cand, topk=(1, 5, 10)):
    S = torch.matmul(Z_query, Z_cand.T)
    ranks = torch.argsort(-S, dim=1)

    recall = {
        k: sum(int(i in ranks[i, :k]) for i in range(len(ranks))) / len(ranks)
        for k in topk
    }

    rr = [
        1.0 / ((ranks[i] == i).nonzero(as_tuple=True)[0].item() + 1)
        for i in range(len(ranks))
    ]

    return {"recall": recall, "MRR": sum(rr) / len(rr)}


def mean_std_metrics(metrics_list):
    out = {}

    vals = [m["MRR"] for m in metrics_list]
    mean = sum(vals) / len(vals)
    std = math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals))
    out["MRR"] = {"mean": mean, "std": std}

    out["recall"] = {}
    for k in metrics_list[0]["recall"].keys():
        vals = [m["recall"][k] for m in metrics_list]
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals))
        out["recall"][k] = {"mean": mean, "std": std}

    return out


# =========================================================
# Losses
# =========================================================

def contrastive_loss(z1, z2, temp=TEMP):
    sim = torch.matmul(z1, z2.T) / temp
    targets = torch.arange(z1.size(0), device=z1.device)
    return (
        F.cross_entropy(sim, targets) +
        F.cross_entropy(sim.T, targets)
    ) / 2


# =========================================================
# Encoder helpers
# =========================================================

def to_sbert(model_name):
    word = models.Transformer(model_name)
    pool = models.Pooling(word.get_word_embedding_dimension())
    return SentenceTransformer(modules=[word, pool]).to(device)


def encode_and_cache(texts, encoder, cache_file, proj=None):
    if os.path.exists(cache_file):
        return torch.from_numpy(np.load(cache_file)).to(device)

    X = encoder.encode(texts, convert_to_tensor=True, device=device)

    if proj is not None:
        X = proj(X)

    X = F.normalize(X, dim=1)
    np.save(cache_file, X.cpu().numpy())
    return X


# =========================================================
# Build texts once
# =========================================================

print("Building representations...")

opts = []
for row in tqdm(data):
    root = parse_mathml_tree(row["formula"])
    opts.append(mathml_to_opt(root))

struct_texts = [" ".join(opt_to_sequence(o)) for o in opts]
semantic_texts = [" ".join(get_descriptions(r)) for r in data]


# =========================================================
# Training / Evaluation
# =========================================================

all_s2t, all_t2s = [], []

for fold in ["1", "2", "3", "4", "5"]:

    print(f"\n===== Fold {fold} =====")
    
    with open(f"{FOLDS_DIR}/fold_{fold}_train.jsonl") as f:
        train_rows = [json.loads(l) for l in f]

    with open(f"{FOLDS_DIR}/fold_{fold}_val.jsonl") as f:
        test_rows = [json.loads(l) for l in f]

    item_to_idx = {r["item"]: i for i, r in enumerate(data)}
    train_idx = [item_to_idx[r["item"]] for r in train_rows]
    test_idx = [item_to_idx[r["item"]] for r in test_rows]

    struct_train = [struct_texts[i] for i in train_idx]
    struct_test = [struct_texts[i] for i in test_idx]
    sem_train = [semantic_texts[i] for i in train_idx]
    sem_test = [semantic_texts[i] for i in test_idx]

    # =====================================================
    # Baselines
    # =====================================================

    if BASELINE_TYPE == "dense_noalign":

        # encoder = to_sbert("tbs17/MathBERT")
        encoder = to_sbert(MODEL_NAME)

        Zs_test = encode_and_cache(struct_test, encoder, f"s_test_{fold}.npy")
        Zt_test = encode_and_cache(sem_test, encoder, f"t_test_{fold}.npy")

    elif BASELINE_TYPE == "dual_encoder":

        # enc_s = to_sbert("all-mpnet-base-v2")
        # enc_t = to_sbert("all-mpnet-base-v2")
        enc_s = to_sbert("all-mpnet-base-v2")
        enc_t = to_sbert("all-mpnet-base-v2")
        hidden_dim = enc_s.get_sentence_embedding_dimension()
        
        proj_s = nn.Linear(hidden_dim, EMB_DIM).to(device)
        proj_t = nn.Linear(hidden_dim, EMB_DIM).to(device)

        # proj_s = nn.Linear(768, EMB_DIM).to(device)
        # proj_t = nn.Linear(768, EMB_DIM).to(device)

        Xs = encode_and_cache(struct_train, enc_s, f"s_train_{fold}.npy")
        Xt = encode_and_cache(sem_train, enc_t, f"t_train_{fold}.npy")

        dataset = TensorDataset(Xs, Xt)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        optim = torch.optim.Adam(
            list(proj_s.parameters()) + list(proj_t.parameters()),
            lr=1e-4
        )

        for _ in range(EPOCHS):
            for xs, xt in loader:
                z1 = F.normalize(proj_s(xs), dim=1)
                z2 = F.normalize(proj_t(xt), dim=1)
                loss = contrastive_loss(z1, z2)
                optim.zero_grad()
                loss.backward()
                optim.step()

        Zs_test = F.normalize(proj_s(
            encode_and_cache(struct_test, enc_s, f"s_test_{fold}.npy")), dim=1)
        Zt_test = F.normalize(proj_t(
            encode_and_cache(sem_test, enc_t, f"t_test_{fold}.npy")), dim=1)

    # =====================================================
    # Evaluate
    # =====================================================

    m_s2t = retrieval_metrics(Zs_test, Zt_test)
    m_t2s = retrieval_metrics(Zt_test, Zs_test)

    print("S→T:", m_s2t)
    print("T→S:", m_t2s)

    all_s2t.append(m_s2t)
    all_t2s.append(m_t2s)


# =========================================================
# Final results
# =========================================================

print("\n===== FINAL RESULTS =====")

avg_s2t = mean_std_metrics(all_s2t)
avg_t2s = mean_std_metrics(all_t2s)

print("\nStructural → Semantic", avg_s2t)
print("\nSemantic → Structural", avg_t2s)
