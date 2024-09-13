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
2. Add **nanoowl** and **FastSAM** to reduce time consumption.
3. The functions are wrapped as individual services. 

**TODO:**
- &#10004; Test the cuda-version groundingDINO and SAM
- &#10004; Test the nanowrl with ["red balls", "purple balls"]
- ~~Install nanosam~~
- &#10004; Write the gradio app, complish the reassignment of object
- &#10004; Test with multiple balls
  - &#10004; GroundingDINO+FastSAM could find the ball and segment it quickly.
  - &#10004; OWL-ViT could not detect balls when there are lots of balls.
- &#10004; Use FastSAM,
  - &#10004; GroundingDINO+FastSAM
  - &#10004; OwlVit+FastSAM
  - &#10004; FastSAM with point
- Overlap retest (convert the whole rgb image into pointcloud, and select several points, decide the correspondence between rgb image and pointcloud)--find project_2d_to_3d.py in /src
- &#10004; Add real-time interaction, service launch and redetect into the gradio app.
- &#10004; Add Azure speech service.
  
- ~~Enhance the groundingdino with GPT-4.~~
- &#10004; Enhance the real-time owl with GPT-4.
- Merge the services into one file.
- ~~Now the detection is only executed once. If want to track the object, can use XMEM or recent SAM-v2.(Resource required)~~
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

If use a real-time version of OWL-ViT: https://github.com/NVIDIA-AI-IOT/nanoowl or use cuda to accelerate GroundingDINO. You need to install cuda toolkit.

**1. Let's start with installation of cuda.**

Reference: https://blog.csdn.net/leo0308/article/details/136414444
The difference between the cuda version shown in `nvidia-smi` and `cuda-toolkit`: https://forums.developer.nvidia.com/t/nvdia-smi-show-cuda-version-but-nvcc-not-found/67311

1. Assume you already installed cuda (nvidia-smi can be called normally). Then need to install cuda-toolkit. Go to https://developer.nvidia.com/cuda-toolkit-archive to find the version desired.

2. You can install multiple versions of cuda-toolkit, and indicate the version to be used in `~/.bashrc`. Take `cuda-11.8` as one example. 
   ```python
   export PATH=/usr/local/cuda-11.8/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
   export CUDA_HOME=/usr/local/cuda-11.8 # GroundingDINO requires to set such environment variable
   ```
   To verify the installation, run
   ```python
   nvcc -V
   ```
   or check whether the file structure is the same as described in: https://saturncloud.io/blog/how-to-locate-cuda-installation-on-linux/

**2. Install tensorRT.**

Need to install `tensorRT` first, because `torch2trt` is depended on `tensorRT`.
Reference: https://medium.com/kgxperience/how-to-install-tensorrt-a-comprehensive-guide-99557c0e9d6 (start from `Downloading cuDNN for Linux`)
https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html

1. Check the version of cuDNN to use: https://docs.nvidia.com/deeplearning/cudnn/latest/reference/support-matrix.html
   Download cuDNN: https://developer.nvidia.com/cudnn

   Previously I installed `cudn-cuda-12`, so it raised the error:
   ```
   sudo apt-get -y install cudnn-cuda-11
   Reading package lists... Done
   Building dependency tree       
   Reading state information... Done
   Note, selecting 'cudnn9-cuda-11' instead of 'cudnn-cuda-11'
   Some packages could not be installed. This may mean that you have
   requested an impossible situation or if you are using the unstable
   distribution that some required packages have not yet been created
   or been moved out of Incoming.
   The following information may help to resolve the situation:
   
   The following packages have unmet dependencies:
    cudnn9-cuda-11 : Depends: cudnn9-cuda-11-8 (>= 9.3.0.75) but it is not going to be installed
   E: Unable to correct problems, you have held broken packages.
   ```
   If so, run
   ```
   sudo apt-get remove --purge cudnn-cuda-12
   ```
   or
   ```
   sudo apt-get install cudnn9-cuda-11-8
   ```
   Then retry.

   NOTE: Follow the instructions after the selection of cuDNN version is enough. Don't need to go back and follow the instructions in https://medium.com/kgxperience/how-to-install-tensorrt-a-comprehensive-guide-99557c0e9d6 for cuDNN.

2. Install tensorRT
   Installation guidance: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html
   You can only follow `3.1. Python Package Index Installation` session.

   If you encountered:
   ```
   WARNING: Error parsing dependencies of torch: [Errno 2] No such file or directory: '/home/lab_cheem/miniforge3/envs/claw_machine/lib/python3.10/site-packages/torch-2.3.1.dist-info/METADATA'
   ```
   when run `python3 -m pip install --upgrade pip`, and
   ```
   ERROR: Could not install packages due to an OSError: [Errno 2] No such file or directory: '/home/lab_cheem/miniforge3/envs/claw_machine/lib/python3.10/site-packages/torch-2.3.1.dist-info/METADATA'
   ```
   when run `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`,
   run `rm -rf /home/lab_cheem/miniforge3/envs/claw_machine/lib/python3.10/site-packages/torch*`, then install torch torchvision torchaudio again. Reference (asked GPT): https://chatgpt.com/share/4d0d57b9-18b4-4343-a615-8e1b94288a69

   Then follow the installation guidance and verify the installation.
   If encountered
   ```
   [TRT] [W] CUDA lazy loading is not enabled. Enabling it can significantly reduce device memory usage and speed up TensorRT initialization. See "Lazy Loading" section of CUDA documentation https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#lazy-loading
   ```
   Try `export CUDA_MODULE_LOADING=LAZY` or `echo 'export CUDA_MODULE_LOADING=LAZY' >> ~/.bashrc` and `source ~/.bashrc` for permanent configuration.

   NOTE: This is only the installation of python support!!! You need to install the local tensorRT using debian or other manners. I used debian. Refer to `3.2.1. Debian Installation` session.
   When I run `sudo apt-get install tensorrt`, don't know why it cannot be installed.
   The error is like:
   ```
   N: Skipping acquire of configured file 'main/binary-i386/Packages' as repository 'https://brave-browser-apt-release.s3.brave.com stable InRelease' doesn't support architecture 'i386'
   W: GPG error: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  InRelease: The following signatures couldn't be verified because the public key is not available: NO_PUBKEY A4B469963BF863CC
   E: The repository 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  InRelease' is not signed.
   N: Updating from such a repository can't be done securely, and is therefore disabled by default.
   ```

   The response from GPT:

   ```
   The errors you're encountering suggest two different issues:

    Skipping i386 Architecture: This is not critical and only indicates that the i386 architecture is being skipped in certain repositories, which is fine if you're running a 64-bit system.

    GPG Key Error for NVIDIA Repository: The more pressing issue is that the public key for the NVIDIA repository is missing, which prevents the repository from being used securely.
   ```
   Then I run:

   ```
   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
   ```
   and
   ```
   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
   ```

   Then run `sudo apt-get update`, this error `E: The repository 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  InRelease' is not signed.` disappeared. And then I run `sudo apt-get install tensorrt` it worked.

**3. Install torch2trt.**

```
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python setup.py install
```

**4. Install transformer.**
```
python3 -m pip install transformers
```

### USE REAL-TIME SAM

If use real time version of SAM: https://github.com/NVIDIA-AI-IOT/nanosam

Due to some operation incompatibility, there are some errors:

```
[E] Error[4]: ITensor::getDimensions: Error Code 4: Internal Error (/OneHot: an IIOneHotLayer cannot be used to compute a shape tensor)
[09/02/2024-17:35:22] [E] [TRT] ModelImporter.cpp:949: While parsing node number 146 [Tile -> "/Tile_output_0"]:
[09/02/2024-17:35:22] [E] [TRT] ModelImporter.cpp:950: --- Begin node ---
input: "/Unsqueeze_3_output_0"
input: "/Reshape_2_output_0"
output: "/Tile_output_0"
name: "/Tile"
op_type: "Tile"

[09/02/2024-17:35:22] [E] [TRT] ModelImporter.cpp:951: --- End node ---
[09/02/2024-17:35:22] [E] [TRT] ModelImporter.cpp:954: ERROR: ModelImporter.cpp:195 In function parseNode:
[6] Invalid Node - /Tile
ITensor::getDimensions: Error Code 4: Internal Error (/OneHot: an IIOneHotLayer cannot be used to compute a shape tensor)
[09/02/2024-17:35:22] [E] Failed to parse onnx file
[09/02/2024-17:35:22] [I] Finished parsing network model. Parse time: 0.0483808
[09/02/2024-17:35:22] [E] Parsing model failed
[09/02/2024-17:35:22] [E] Failed to create engine from model or file.
[09/02/2024-17:35:22] [E] Engine set up failed
&&&& FAILED TensorRT.trtexec [TensorRT v100300] # trtexec --onnx=data/mobile_sam_mask_decoder.onnx --saveEngine=data/mobile_sam_mask_decoder.engine --minShapes=point_coords:1x1x2,point_labels:1x1 --optShapes=point_coords:1x1x2,point_labels:1x1 --maxShapes=point_coords:1x10x2,point_labels:1x10
```

