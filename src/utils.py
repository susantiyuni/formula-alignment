import random
import os
import torch
import numpy as np
from sklearn.metrics import normalized_mutual_info_score

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_or_cache_embeddings(texts, encoder, cache_file, device):
    if os.path.exists(cache_file):
        return torch.from_numpy(np.load(cache_file)).to(device)

    embs = encoder.encode(texts, convert_to_tensor=True, device=device)
    np.save(cache_file, embs.cpu().numpy())
    return embs

def retrieval_metrics(Z_query, Z_cand, topk=(1,5,10)):
    S = torch.matmul(Z_query, Z_cand.T)
    ranks = torch.argsort(-S, dim=1)

    recall = {
        k: sum(int(i in ranks[i,:k]) for i in range(len(ranks))) / len(ranks)
        for k in topk
    }

    rr = [(1.0 / ((ranks[i]==i).nonzero()[0].item()+1)) for i in range(len(ranks))]

    return {"recall": recall, "MRR": sum(rr)/len(rr)}
