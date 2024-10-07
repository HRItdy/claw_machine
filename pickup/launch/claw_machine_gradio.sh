#!/bin/bash
source /home/lab_cheem/miniforge3/etc/profile.d/mamba.sh

# Source ROS setup
source /opt/ros/noetic/setup.bash

# Source your workspace setup
source /home/lab_cheem/claw_machine/devel/setup.bash

# Activate the mamba environment
mamba activate claw_machine

# Run the commands in sequence
python /home/lab_cheem/claw_machine/src/pickup/scripts/pc_calibration.py &
python /home/lab_cheem/claw_machine/src/pickup/scripts/claw_detect.py &
python /home/lab_cheem/claw_machine/src/pickup/scripts/claw_depth.py &
python /home/lab_cheem/claw_machine/src/pickup/scripts/ur_executor.py &
python /home/lab_cheem/claw_machine/src/pickup/scripts/claw_pickup.py &
wait  # Wait for all background processes to finish

echo "All scripts have finished execution."
