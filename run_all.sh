#!/bin/bash

set -e  # stop on error

echo "Running BM25..."
python bm25.py

echo "Running dual_encoder no CL..."
python train.py --baseline_type dual_encoder --encoder_type mathbert
python train.py --baseline_type dual_encoder --encoder_type sbert

echo "Running dense_noalign no CL..."
python train.py --baseline_type dense_noalign --encoder_type mathbert
python train.py --baseline_type dense_noalign --encoder_type sbert

echo "Running dual_encoder CL..."
python dual_encoder.py --model_name all-mpnet-base-v2
python dual_encoder.py --model_name tbs17/MathBERT

echo "Running GNN-CL..."
python train_gnn_cl.py

echo "All experiments completed."
