import numpy as np
import random, os, sys, math
import json
from sentence_transformers import SentenceTransformer, models
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from scipy.linalg import orthogonal_procrustes

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

print (f"Loading formulae representations...")
X_struct = np.load("data/X_struct.npy")
X_sem    = np.load("data/X_sem.npy")

print("X_struct shape:", X_struct.shape)
print("X_sem shape:", X_sem.shape)

X_struct = X_struct / np.linalg.norm(X_struct, axis=1, keepdims=True)
X_sem    = X_sem / np.linalg.norm(X_sem, axis=1, keepdims=True)

assert X_struct.shape[0] == X_sem.shape[0]
assert not np.isnan(X_struct).any()
assert not np.isnan(X_sem).any()

# 1. Instance-level cosine alignment (per formula)
# Whether each formula’s structural embedding aligns with its own semantic embedding.
# Higher values indicate stronger per-instance structure–semantics agreement.

Xs_pca = PCA(n_components=128).fit_transform(X_struct)
Xm_pca = PCA(n_components=128).fit_transform(X_sem)

Xs_n = Xs_pca / np.linalg.norm(Xs_pca, axis=1, keepdims=True)
Xm_n = Xm_pca / np.linalg.norm(Xm_pca, axis=1, keepdims=True)

cos_self = np.sum(Xs_n * Xm_n, axis=1)

print("Mean cosine:", cos_self.mean())
print("Median cosine:", np.median(cos_self))
print("Std cosine:", cos_self.std())

# 2. Representational Similarity Analysis (RSA)
# Whether the two embedding spaces induce similar global similarity structures.
# A high Spearman ρ indicates that structure and semantics organize formulas similarly.

S_struct = cosine_similarity(X_struct)
S_sem = cosine_similarity(X_sem)

def upper_tri(M):
    return M[np.triu_indices_from(M, k=1)]

rho, p = spearmanr(upper_tri(S_struct), upper_tri(S_sem))
print("RSA Spearman ρ:", rho, "p:", p)

# 3. Canonical Correlation Analysis (CCA)
# Whether there exist shared latent directions where structure and meaning correlate.
# High canonical correlations imply shared latent structure across modalities.

Xs = StandardScaler().fit_transform(X_struct)
Xm = StandardScaler().fit_transform(X_sem)

k = min(50, Xs.shape[1], Xm.shape[1])
cca = CCA(n_components=k)

Xs_c, Xm_c = cca.fit_transform(Xs, Xm)
corrs = [np.corrcoef(Xs_c[:, i], Xm_c[:, i])[0, 1] for i in range(k)]

print("Mean CCA correlation:", np.mean(corrs))
print("Top-10 CCA correlations:", corrs[:10])

# 4. Orthogonal Procrustes alignment
# How well one space can be aligned to the other using a rigid transformation.
# Lower error means stronger geometric compatibility.

Xs0 = Xs_pca - Xs_pca.mean(axis=0)
Xm0 = Xm_pca - Xm_pca.mean(axis=0)

R, _ = orthogonal_procrustes(Xs0, Xm0)
aligned = Xs0 @ R

error = np.mean(np.linalg.norm(aligned - Xm0, axis=1))
print("Procrustes alignment error:", error)

# 5. Cross-modal retrieval (structure → semantics)
# Whether the correct semantic embedding is retrieved given a structural query.

S = cosine_similarity(Xs_n, Xm_n)
ranks = np.argsort(-S, axis=1)
def topk(ranks, k):
    return np.mean([i in ranks[i, :k] for i in range(len(ranks))])

print("Top-1:", topk(ranks, 1))
print("Top-5:", topk(ranks, 5))
print("Top-10:", topk(ranks, 10))

# 6. Centered Kernel Alignment (CKA)
# Global representational similarity, invariant to scaling and rotation.
# CKA provides a robust summary score for representation alignment.

def linear_CKA(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    return np.linalg.norm(X.T @ Y, 'fro')**2 / (
        np.linalg.norm(X.T @ X, 'fro') * np.linalg.norm(Y.T @ Y, 'fro')
    )

cka = linear_CKA(Xs_pca, Xm_pca)
print("Linear CKA:", cka)

cka_o = linear_CKA(X_struct, X_sem)
print("Linear CKA (orig):", cka_o)

# 7. Permutation test (statistical significance)
# Shows if alignment is significantly above chance.

def perm_test(Xs, Xm, n_perm=1000):
    real = np.mean(np.sum(Xs * Xm, axis=1))
    perms = []
    for _ in range(n_perm):
        idx = np.random.permutation(len(Xs))
        perms.append(np.mean(np.sum(Xs * Xm[idx], axis=1)))
    return real, np.mean(perms), np.std(perms)

real, mean_null, std_null = perm_test(Xs_n, Xm_n)
print("Real:", real, "Null mean:", mean_null, "Null std:", std_null)
