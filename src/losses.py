# losses.py
# =========================================================
# Contrastive / alignment losses
# =========================================================

import torch
import torch.nn.functional as F


# =========================================================
# InfoNCE (symmetric)
# =========================================================
def contrastive_loss(z1, z2, temp=0.07):
    """
    Standard symmetric InfoNCE.

    z1, z2 : [B, D]
    """
    sim = torch.matmul(z1, z2.T) / temp
    targets = torch.arange(z1.size(0), device=z1.device)

    loss = (
        F.cross_entropy(sim, targets) +
        F.cross_entropy(sim.T, targets)
    ) / 2

    return loss


# =========================================================
# Hard negative emphasis (optional)
# =========================================================
def contrastive_loss_neg(z1, z2, temp=0.07, hard_neg_weight=0.5):
    """
    InfoNCE with harder negatives emphasized.

    Helpful when many easy negatives exist.
    """

    sim = torch.matmul(z1, z2.T) / temp
    B = sim.size(0)

    mask = torch.eye(B, device=sim.device)
    hard_mask = 1 - mask

    hard_scale = 1 + hard_neg_weight * (sim * hard_mask)
    sim = sim * (mask + hard_scale)

    targets = torch.arange(B, device=sim.device)

    loss = (
        F.cross_entropy(sim, targets) +
        F.cross_entropy(sim.T, targets)
    ) / 2

    return loss


# =========================================================
# Intra-modal consistency
# =========================================================
def intra_modal_loss(z, temp=0.07):
    """
    Self-contrastive regularization.

    Encourages modality consistency.
    """
    return contrastive_loss(z, z)
