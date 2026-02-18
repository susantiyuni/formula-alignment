#!/bin/bash

set -e  # stop on error

echo "Running dense_noalign baseline..."
python train.py --baseline_type dense_noalign

echo "Running dual_encoder baseline..."
python train.py --baseline_type dual_encoder

echo "Running simcse baseline..."
python train.py --baseline_type simcse

echo "All experiments completed."
