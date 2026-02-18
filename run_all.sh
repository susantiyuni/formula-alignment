#!/bin/bash

set -e  # stop on error

echo "Running BM25..."
python bm25.py

echo "Running dense_noalign..."
python train.py --baseline_type dense_noalign --encoder_type mathbert
python train.py --baseline_type dense_noalign --encoder_type sbert

echo "Running dual_encoder..."
python train.py --baseline_type dual_encoder --encoder_type mathbert
python train.py --baseline_type dual_encoder --encoder_type sbert

echo "Running simcse..."
python train.py --baseline_type simcse

echo "Running GNN-CL..."
python train_gnn_cl.py

echo "All experiments completed."
