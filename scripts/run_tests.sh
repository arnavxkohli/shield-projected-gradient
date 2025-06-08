#!/usr/bin/env bash

set -euo pipefail
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

set -o allexport
source .env
set +o allexport

function send_notification {
    status=$?
    if [ $status -eq 0 ]; then
        subject="✅ Experiment Succeeded"
    else
        subject="❌ Experiment Failed"
    fi
    echo "Experiment completed with status $status at $(date '+%Y-%m-%d %H:%M:%S')." | mail -s "$subject" "$EMAIL_NOTIFY"
}

trap send_notification EXIT

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
