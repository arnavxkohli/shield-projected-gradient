#!/usr/bin/env bash

set -euo pipefail
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

datasets=("url" "faulty-steel-plates" "news" "lcld" "botnet")

for dataset in "${datasets[@]}"; do
    echo "Running gradient analysis for dataset=$dataset"
    if [ "$dataset" == "botnet" ]; then
        python3 src/run_grad_analysis.py --data-dir "$dataset" --numpy-data
    else
        python3 src/run_grad_analysis.py --data-dir "$dataset"
    fi
done
