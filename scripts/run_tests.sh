#!/bin/bash
python3 model_run.py --data-dir url --optimizer sgd
python3 model_run.py --data-dir faulty-steel-plates
python3 model_run.py --data-dir lcld --train-fraction 0.1
python3 model_run.py --data-dir news
