#!/bin/bash

set -e  # stop on error

echo "Running dense_noalign..."
python train.py --baseline_type dense_noalign

echo "Running dual_encoder..."
python train.py --baseline_type dual_encoder

echo "Running simcse..."
python train.py --baseline_type simcse

echo "Running GNN-CL..."
python train_gnn_cl.py

echo "All experiments completed."
