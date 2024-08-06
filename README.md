# claw_machine

## This repo is a project to control the UR5 robot to grasp object. 
**Model used:** 
GPT, GroundingDINO, Segment-Anything

**Features:** 
1. The groundingdino and segment-anything are integrated to detect the object.
2. The detection is wrapped as one service. 
3. In this service the mask of the object is generated and published to `/mask` topic.

**TODO:**
- The projection from 2D pixel coordinates to 3D pointcloud coordinates is not accurate.
- Enhance the groundingdino with GPT-4.
- Now the detection is only executed once. If want to track the object, can use XMEM or recent SAM-v2.
