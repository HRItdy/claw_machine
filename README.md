# claw_machine 

## This repo is a project to control the UR5 robot to grasp object. 
**Model used:** 
GPT, GroundingDINO, Segment-Anything\
Implementation of GroundingDINO:\
https://github.com/IDEA-Research/GroundingDINO/tree/main

Implementaion of SAM: \
https://github.com/facebookresearch/segment-anything

In SAM, the checkpoint used is vit-h:\
https://github.com/facebookresearch/segment-anything#model-checkpoints

A script to download the pretrained groundingDINO and sam models:\
https://github.com/z-x-yang/Segment-and-Track-Anything/blob/main/script/download_ckpt.sh

**Features:** 
1. The **GroundingDINO** and **Segment-Anything** are integrated to detect the object.
2. The functions are wrapped as individual services. 

**TODO:**
- Enhance the groundingdino with GPT-4.
- Now the detection is only executed once. If want to track the object, can use XMEM or recent SAM-v2.
- Write the launch to run the nodes simultaneously.
- Form a document.
- Use the graspposMap to do the grasp.


## Sturcture:

**LargestCluster:** A nodelet to select the largest cluster from the clusters generated by EuclideanCluster nodelet.

**PickUp:** The main package including detection, depth and manipulation services.
- pc_calibration.py: The script to map 2D image coordinates to 3D position on tabletop (essentially a plane-to-plane homography projection).
- models.py: Logic to load GroundingDINO and Segment-Anything. **TODO:** change the path to relative path.
- claw_detect.py: Receive the instruction, detect the target and feedback the mask.
- claw_depth.py: Receive the bottom point of the mask, project it to the tabletop coordinate, 
- ur_executor.py: Connect to the robot and initialize the actionlib.
- claw_pickup.py: Initialize the pickup service.
- **external**: Folder containing the call_service script can be called in other workpackages (called in PromptChat project).


## How to use:
### LAUNCH CONNECTION WITH ROBOT AND ROSCORE
**Terminal 1:**
```python
cd ~/Sources/png_ws/
source setup.bash
roslaunch lab_launch sys_lux.launch
```

### RUN ALL SCRIPTS WITH ONE COMMAND 
**Terminal 2:**
```python
cd ~/claw_machine/src/pickup/launch/    
./claw_machine.sh
```
This shell will:
```python
source ~/claw_machine/devel/setup.bash
mamba activate claw_machine
```
Then run:\
**pc_calibration.py**: The script to map 2D image coordinates to 3D position on tabletop (essentially a plane-to-plane homography projection, detailed usecase please refer to calibration session). \
**claw_detect.py**: Receive the instruction, load the pretrained model as specified in models.py, detect the target and feedback the mask. \
**claw_depth.py**: Receive the bottom point of the mask, project it to the tabletop, estimate the centroid location of the target. \
**ur_executor.py**: Connect to the robot and initialize the actionlib. \
**claw_pickup.py**: Start the manipulation service.


### RUN EACH SCRIPT IN INDIVIDUAL TERMINAL
**Terminal 2A: Calibration**
```python
source ~/claw_machine/devel/setup.bash
mamba activate claw_machine
python pc_calibration.py
```
The calibration result will be stored in `calibration_data.json` file. If there is already a file, the program will load the parameters in default. If the home position is moved, please trigger a new calibration procedure by deleting the calibration json file.

Usecase:\
After running `pc_calibration.py`, when the color image is received, there will be a window. In this window select four points in counterclockwise order. The order of the selected points is indicated by number.

Run `rostopic echo /clicked_point`, select `Publish point` function in rviz, click the bottom of each ball, the coordinates will be published into `/clicked_point` topic. Input them into the terminal. Make sure the fixed frame of rviz is `realsense_wrist_link`. And the order of inputs should match the order of 2D image points.

After calibration, the selected 2D pixel coordinates will be stored in parameter `/calibration/points_2d`, the corresponindg 3D pointcloud coordinates will be stored in parameter `/calibration/points_3d`, and the calculated homography matrix is stored in `/calibration/H`. All these info will be stored in `calibration_data.json` file.

**Terminal 2B:  Detection**
```python
source ~/claw_machine/devel/setup.bash
mamba activate claw_machine
python claw_detect.py
```

#NOTE: Change the model path in models.py if the host machine is changed. In future will change this path to relative path.
This script starts the detection service, which can be called by `call_detect_service.py`.

**Terminal 2C: 3D Position Estimate**
```python
source ~/claw_machine/devel/setup.bash
mamba activate claw_machine
python claw_depth.py
```

This script starts the depth service, which can be called by `call_depth_service.py`.

**Terminal 2D: Initialize Robot**
```python
source ~/claw_machine/devel/setup.bash
mamba activate claw_machine
python ur_executor.py
```

This script will initialize the robot and the actionlib.

**Terminal 2E: Manipulation**     
```python
source ~/claw_machine/devel/setup.bash
mamba activate claw_machine
python claw_pickup.py
```
This script starts the manipulation service, which can be called by `call_pickup_service.py`.

### USE REAL-TIME OWL-ViT

https://github.com/NVIDIA-AI-IOT/nanoowl
