import json
import numpy as np
from sklearn.model_selection import KFold

# Load data
with open('formula-eg-grouped.jsonl') as f:
    data = [json.loads(line) for line in f]

N = len(data)
indices = np.arange(N)

k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
    print(f"Saving Fold {fold+1}/{k}")

    # Select examples
    train_data = [data[i] for i in train_idx]
    val_data   = [data[i] for i in val_idx]

    # Save JSONL
    with open(f"fold_{fold+1}_train.jsonl", "w") as f_train:
        for item in train_data:
            f_train.write(json.dumps(item) + "\n")

    with open(f"fold_{fold+1}_val.jsonl", "w") as f_val:
        for item in val_data:
            f_val.write(json.dumps(item) + "\n")
