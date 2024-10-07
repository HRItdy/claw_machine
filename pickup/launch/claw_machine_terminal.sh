#!/bin/bash

# Source ROS setup
source /opt/ros/noetic/setup.bash

# Source your workspace setup
source /home/lab_cheem/claw_machine/devel/setup.bash

# Define the commands and titles for each tab
declare -a commands=(
    "mamba activate claw_machine; python /home/lab_cheem/claw_machine/src/pickup/scripts/claw_detect.py; exec bash"
    "mamba activate claw_machine; python /home/lab_cheem/claw_machine/src/pickup/scripts/claw_pickup.py; exec bash"
    "mamba activate claw_machine; python /home/lab_cheem/claw_machine/src/pickup/scripts/ur_executor.py; exec bash"
)

declare -a titles=(
    "Detection"
    "Manipulation service"
    "Robot initialization"
)

# Open a new gnome-terminal window with multiple tabs
for i in "${!commands[@]}"; do
    if [ $i -eq 0 ]; then
        gnome-terminal --tab --title="${titles[i]}" -- bash -c "${commands[i]}"
    else
        gnome-terminal --tab --title="${titles[i]}" -- bash -c "${commands[i]}" &
    fi
done

echo "All scripts are running in separate gnome-terminal tabs."
