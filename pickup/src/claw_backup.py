#!/usr/bin/env python3
# main.py

import rospy
import cv2
import numpy as np
from models import runGroundingDino, GroundedDetection, DetPromptedSegmentation, draw_candidate_boxes
from geometry_msgs.msg import Pose, Point32
from std_msgs.msg import Bool, Header
from sensor_msgs.msg import Image, PointCloud, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import argparse
from PIL import Image as Img
import os
import copy
import actionlib
from pickup.msg import pickupAction, pickupGoal 
from message_filters import ApproximateTimeSynchronizer, Subscriber
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 


parser = argparse.ArgumentParser("GroundingDINO", add_help=True)
parser.add_argument("--debug", action="store_true", help="using debug mode")
parser.add_argument("--share", action="store_true", help="share the app")
parser.add_argument("--box_threshold", type=float, default=0.25, help="box threshold")
parser.add_argument("--text_threshold", type=float, default=0.2, help="text threshold")
parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold")
parser.add_argument("--visualize", default=False, help="visualize intermediate data mode")
parser.add_argument("--device", type=str, default="cpu", help="run on: 'cuda:0' for GPU 0 or 'cpu' for CPU. Default GPU 0.")
cfg = parser.parse_args()    

class Claw:
    def __init__(self, instruct):
        rospy.init_node('claw', anonymous=True)
        self.bridge = CvBridge()
        self.pc_pub = rospy.Publisher('/masked_point_cloud', PointCloud, queue_size=1)
        # Initialize components
        self.detector = GroundedDetection(cfg)
        self.segmenter = DetPromptedSegmentation(cfg)
        self.bridge = CvBridge()
        
        # Get the instruction input
        # self.instruct = input('Please enter your instruction')
        self.instruct = instruct
        

        # Subscribers for the topics
        self.info_sub = Subscriber('/realsense_wrist/aligned_depth_to_color/camera_info', CameraInfo)
        self.depth_sub = Subscriber('/realsense_wrist/aligned_depth_to_color/image_raw', Image)
        self.color_sub = Subscriber('/realsense_wrist/color/image_raw', Image)
        
        # # Initiate the realsense camera
        # self.init_camera()
        
        # Synchronize the topics
        self.ats = ApproximateTimeSynchronizer([self.depth_sub, self.color_sub, self.info_sub], queue_size=5, slop=0.1)
        self.ats.registerCallback(self.callback)
        
        # self.action_client = actionlib.SimpleActionClient('grasp_action', pickupAction)
        # self.action_client.wait_for_server()
        rospy.loginfo("Action server is ready")
        
        self.rate = rospy.Rate(10)
        rospy.spin()
        
    #     self.intrinsics = self.color_frame.profile.as_video_stream_profile().intrinsics
    def callback(self, depth_msg, color_msg, info_msg):
        # Extract the intrinsic matrix from CameraInfo
        self.intrinsic_matrix = np.array(info_msg.K).reshape(3, 3)

        # Convert ROS Image messages to OpenCV images
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        color_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        
        # Process the color image
        image = copy.deepcopy(color_image)
        image_pil = Img.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mask = self.process_image(image_pil, self.instruct)
        indices = np.argwhere(mask)[2:,:].transpose(0,1)
        
        points_3d = self.get_3d_points(indices, depth_image)
        self.cluster_pub(points_3d)
        # del self

    def get_3d_points(self, pixel_coords, depth_image):
        points_3d = []
        for [u, v] in pixel_coords:
            depth = depth_image[v, u] / 1000.0  # Assuming depth is in millimeters
            x = (u - self.intrinsic_matrix[0, 2]) * depth / self.intrinsic_matrix[0, 0]
            y = (v - self.intrinsic_matrix[1, 2]) * depth / self.intrinsic_matrix[1, 1]
            z = depth
            points_3d.append((x, y, z))
        return points_3d
    
    def cluster_pub(self, point_cloud):
        header = Header()
        header.stamp = rospy.Time.now()
        # header.frame_id = 'realsense_wrist_aligned_depth_to_color_frame' # correct frame?
        header.frame_id = 'realsense_wrist_aligned_depth_to_color_frame'
        pc_msg = PointCloud()
        pc_msg.header = header
        pc_msg.points = [Point32(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in point_cloud]
        self.pc_pub.publish(pc_msg)
        
        centroid = np.mean(point_cloud, axis=0)
        # Create and send goal to the action server
        goal = pickupGoal()
        goal.name == 'pickup'
        goal.pos.extend(centroid)
        # self.action_client.send_goal(goal)
        # self.action_client.wait_for_result()
        # result = self.action_client.get_result()
        # rospy.loginfo(f"Action result: {result}")
        # Shutdown after executing
        # rospy.signal_shutdown("Action executed.")

    def process_image(self, image, user_request):
        if image is None:
            rospy.logwarn("No image received yet.")
            return

        # Use GroundingDINO to find the ball
        output_dir = os.path.join("outputs/" , user_request)
        os.makedirs(output_dir,exist_ok=True)

        results = self.detector.inference(image, user_request, cfg.box_threshold, cfg.text_threshold, cfg.iou_threshold)
        # results: tensor[n, 4], list[n]
        dino_pil = draw_candidate_boxes(image, results, output_dir, stepstr='nouns', save=True)
        #image_pil.show()
        # only save the first element and ignore the others
        results_ = []
        results_.append(results[0][0].unsqueeze(0))
        results_.append([results[1][0]])
        sin_pil = draw_candidate_boxes(image, results_, output_dir, stepstr='sing', save=True)
        mask = self.segmenter.inference(image, results_[0], results_[1], output_dir, save_json=True)
        return mask

if __name__ == '__main__':
    point_cloud_publisher = Claw('pick up the left red ball')