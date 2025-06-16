#!/bin/bash

alphas=(0.1, 0.3, 0.5, 0.7)

for alpha in "${alphas[@]}"; do
    echo "Launching alpha=$alpha"
    ALPHA=$alpha ~/.venv/bin/python main.py &
done

wait
echo "All runs completed."

