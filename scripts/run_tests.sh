#!/bin/bash

datasets=("url" "faulty-steel-plates" "lcld" "news")
architectures=("shallow" "deep")

for dataset in "${datasets[@]}"; do
    for arch in "${architectures[@]}"; do
        cmd="python3 model_run.py --data-dir $dataset --base-arch $arch"
        if [ "$dataset" = "lcld" ]; then
            cmd="$cmd --train-fraction 0.1"
        fi
        echo "Running: $cmd"
        eval $cmd
    done
done
