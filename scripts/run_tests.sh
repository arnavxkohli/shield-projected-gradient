#!/usr/bin/env bash

set -euo pipefail
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

datasets=("url" "faulty-steel-plates" "news" "lcld")
architectures=("shallow" "deep")
mask_options=("" "--mask-method")

for dataset in "${datasets[@]}"; do
    for arch in "${architectures[@]}"; do
        for mask in "${mask_options[@]}"; do
            if [ "$dataset" = "lcld" ] && [ "$mask" = "--mask-method" ]; then
                echo "Skipping masked method for dataset=$dataset"
                continue
            fi
            cmd=(python3 src/run_rmse.py --data-dir "$dataset" --base-arch "$arch")
            if [ -n "$mask" ]; then
                cmd+=("$mask")
            fi
            if [ "$dataset" = "lcld" ]; then
                cmd+=(--train-fraction 0.1)
            fi
            echo "Running: ${cmd[*]}"
            "${cmd[@]}"
        done
    done
done

python3 src/analyze_rmse.py
