#!/bin/bash

# Define the names of the Python scripts to be terminated
declare -a script_names=(
    "claw_detect.py"
    "claw_pickup.py"
    "ur_executor.py"
)

# Loop through the script names and terminate each
for script_name in "${script_names[@]}"; do
    echo "Shutting down $script_name..."
    pkill -f $script_name
done

echo "All specified Python scripts have been shut down."
