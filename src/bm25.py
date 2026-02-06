# baseline_bm25.py
# ======================================================
# BM25 Structural–Semantic Retrieval Baseline
# ======================================================

import json
import math
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi

from utils.seed import set_seed
from utils.metrics import retrieval_metrics

from mathml_parser import parse_mathml_tree, mathml_to_opt
from rep_semantics import build_semantic_texts


# ======================================================
# Config
# ======================================================

SEED = 42
DATA_FILE = "formula-eg-grouped.jsonl"
FOLDS_DIR = "data-cat"


# ======================================================
# Helpers
# ======================================================

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


# ------------------------------------------------------
# Structural linearizations
# ------------------------------------------------------
def opt_to_sequence(opt):
    """Value-only preorder traversal (best BM25 baseline)."""
    seq = [opt["value"]]
    for c in opt["children"]:
        seq += opt_to_sequence(c)
    return seq


def linearize_opts(opts):
    return [" ".join(opt_to_sequence(o)) for o in opts]


def tokenize(texts):
    return [t.split() for t in texts]


# ------------------------------------------------------
# Mean/std across folds
# ------------------------------------------------------
def mean_std_metrics(metrics_list):
    out = {}

    # MRR
    vals = [m["MRR"] for m in metrics_list]
    mean = np.mean(vals)
    std = np.std(vals)
    out["MRR"] = {"mean": mean, "std": std}

    # Recall
    out["recall"] = {}
    ks = metrics_list[0]["recall"].keys()

    for k in ks:
        vals = [m["recall"][k] for m in metrics_list]
        out["recall"][k] = {
            "mean": np.mean(vals),
            "std": np.std(vals)
        }

    return out


# ======================================================
# Build representations once
# ======================================================

def build_structural_texts(data):

    print("Parsing MathML → OPT ...")

    opts = []

    for row in data:
        root = parse_mathml_tree(row["formula"])
        opts.append(mathml_to_opt(root))

    struct_texts = linearize_opts(opts)

    return struct_texts


# ======================================================
# BM25 evaluation
# ======================================================

def run_bm25(struct_texts, semantic_texts, test_idx):

    struct_test = [struct_texts[i] for i in test_idx]
    sem_test = [semantic_texts[i] for i in test_idx]

    struct_tok = tokenize(struct_test)
    sem_tok = tokenize(sem_test)

    bm25_struct = BM25Okapi(struct_tok)
    bm25_sem = BM25Okapi(sem_tok)

    # -------------------------
    # Structural → Semantic
    # -------------------------
    scores_S2T = np.zeros((len(struct_tok), len(sem_tok)))

    for i, q in enumerate(tqdm(struct_tok, leave=False)):
        scores_S2T[i] = bm25_sem.get_scores(q)

    metrics_s2t = retrieval_metrics(scores_S2T, scores_S2T.argmax(axis=1))

    # -------------------------
    # Semantic → Structural
    # -------------------------
    scores_T2S = np.zeros((len(sem_tok), len(struct_tok)))

    for i, q in enumerate(tqdm(sem_tok, leave=False)):
        scores_T2S[i] = bm25_struct.get_scores(q)

    metrics_t2s = retrieval_metrics(scores_T2S, scores_T2S.argmax(axis=1))

    return metrics_s2t, metrics_t2s


# ======================================================
# Main
# ======================================================

if __name__ == "__main__":

    set_seed(SEED)

    print("Loading dataset...")
    data = load_jsonl(DATA_FILE)

    # ----------------------------------------
    # Build shared representations once
    # ----------------------------------------
    struct_texts = build_structural_texts(data)
    semantic_texts = build_semantic_texts(data)

    all_s2t = []
    all_t2s = []

    for k in ["1", "2", "3", "4", "5"]:

        print(f"\n========== Fold {k} ==========")

        train_rows = load_jsonl(f"{FOLDS_DIR}/fold_{k}_train.jsonl")
        test_rows  = load_jsonl(f"{FOLDS_DIR}/fold_{k}_val.jsonl")

        item_to_index = {r["item"]: i for i, r in enumerate(data)}
        test_idx = [item_to_index[r["item"]] for r in test_rows]

        s2t, t2s = run_bm25(struct_texts, semantic_texts, test_idx)

        print("\nStructural → Semantic:", s2t)
        print("Semantic → Structural:", t2s)

        all_s2t.append(s2t)
        all_t2s.append(t2s)

    # ----------------------------------------
    # Final averages
    # ----------------------------------------
    avg_s2t = mean_std_metrics(all_s2t)
    avg_t2s = mean_std_metrics(all_t2s)

    print("\n=========== FINAL ===========")

    print("\nStructural → Semantic")
    print("MRR:", avg_s2t["MRR"])
    for k, v in avg_s2t["recall"].items():
        print(f"R@{k}: {v}")

    print("\nSemantic → Structural")
    print("MRR:", avg_t2s["MRR"])
    for k, v in avg_t2s["recall"].items():
        print(f"R@{k}: {v}")
