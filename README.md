# claw_machine

## This repo is a project to control the UR5 robot to grasp object. 
**Model used:** 
GPT, GroundingDINO, Segment-Anything

**Features:** 
1. The groundingdino and segment-anything are integrated to detect the object.
2. The detection is wrapped as one service. 
3. In this service the mask of the object is generated and published to `/mask` topic.

**TODO:**
- The transformation from original pointcloud to a planar pointcloud is good, but the inverse transformation is not accurate.
- The claw_detection is stuck after adding find_bottom function. The easiest way to solve this is putting the find_bottom in pc_segment.py
- Enhance the groundingdino with GPT-4.
- Now the detection is only executed once. If want to track the object, can use XMEM or recent SAM-v2.

- Store the calibration parameter into local file. Consider the load logic to get the local file, or redo the calibration.
- Solve the deadlock in claw_detect.py.
- Write the launch to run the nodes simultaneously.


## Sturcture:

**LargestCluster:** A nodelet to select the largest cluster from the clusters generated by EuclideanCluster nodelet.

**PickUp:** The main function. Now contains the detection and grasping. Overlay is to be done.

**call_service_global.py:** The client to call the detection service. Can be placed in any other ROS package or projects. But remember to place the `srv` file into the project.

## How to use:
### Run sys_lux.launch
### Run pc_transform.py

### Run pc_calibration.py
 - For image point input, run `python pc_calibration.py`, select four points in counterclockwise order. The order of the points is indicated by number. Remember to input the pointcloud coordinates accordingly.
 - For point cloud points input, run `rostopic echo /clicked_point`, select `Publish point` in rviz, click the bottom of each ball, the coordinates will be published into `/clicked_point` topic. Input them into the terminal.
### Run claw_detection.py and call_detect_service.py

### Run pc_segment.py
The center point is published to `..... ` TODO: Add the pipeline diagram and the architecture of the topics in readme.
