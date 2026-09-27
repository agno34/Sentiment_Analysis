#!/usr/bin/bash

python3 prepare_data.py 2>/dev/null

python3 train_model.py 2>/dev/null

python3 evaluate_model.py 2>/dev/null