#!/bin/bash

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
            cmd="python3 model_run.py --data-dir $dataset --base-arch $arch $mask"
            if [ "$dataset" = "lcld" ]; then
                cmd="$cmd --train-fraction 0.1"
            fi
            echo "Running: $cmd"
            eval $cmd
        done
    done
done
