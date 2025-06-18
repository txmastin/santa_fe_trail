#!/bin/bash

alphas=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

for alpha in "${alphas[@]}"; do
    echo "Launching alpha=$alpha"
    ALPHA=$alpha ~/.venv/bin/python main.py 
done

wait
echo "All runs completed."

