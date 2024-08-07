#!/usr/bin/env python3
# main.py

import rospy
import numpy as np
from models import runGroundingDino, GroundedDetection, DetPromptedSegmentation, draw_candidate_boxes
from pickup.srv import GroundingDINO, GroundingDINOResponse  # Adjust the import based on your package name
from geometry_msgs.msg import Point32
from std_msgs.msg import Header, Bool, Int32MultiArray, MultiArrayDimension
from sensor_msgs.msg import Image, PointCloud, CameraInfo
import argparse
from PIL import Image as Img, ImageOps, ImageDraw
import os
import copy
from message_filters import ApproximateTimeSynchronizer, Subscriber
import pyrealsense2 as rs2

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
        self.pc_pub = rospy.Publisher('/masked_point_cloud', PointCloud, queue_size=1)
        # Initialize components
        self.detector = GroundedDetection(cfg)
        self.segmenter = DetPromptedSegmentation(cfg)
        self.color_image = None
        self.depth_image = None

        # Service server
        self.service = rospy.Service('grounding_dino', GroundingDINO, self.handle_service)

        # Get the instruction input
        self.instruct = instruct
        self.processed_point_cloud = None

        # Subscribers for the topics
        # self.color_info_sub = Subscriber('/realsense_wrist/color/camera_info', CameraInfo)
        # self.depth_info_sub = Subscriber('/realsense_wrist/depth/camera_info', CameraInfo)

        self.info_sub = Subscriber('/realsense_wrist/aligned_depth_to_color/camera_info', CameraInfo)
        self.depth_sub = Subscriber('/realsense_wrist/aligned_depth_to_color/image_raw', Image)
        #self.depth_sub = Subscriber('/realsense_wrist/aligned_depth_to_color/image_raw', Image)
        self.color_sub = Subscriber('/realsense_wrist/color/image_raw', Image)
        self.mask_pub = rospy.Publisher('/mask', Int32MultiArray, queue_size=10)

        # Synchronize the topics
        self.ats = ApproximateTimeSynchronizer([self.depth_sub, self.color_sub, self.info_sub], queue_size=5, slop=0.1)
        self.ats.registerCallback(self.callback)

        self.rate = rospy.Rate(10)  # 10 Hz

        rospy.spin()

    def callback(self, depth_msg, color_msg, info_msg):

        # self.color_info = color_info_msg
        # self.depth_info = depth_info_msg
        self.intrinsic_matrix = np.array(info_msg.K).reshape(3, 3)

        # Convert ROS Image messages to OpenCV images manually
        self.depth_image = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(depth_msg.height, depth_msg.width)
        self.color_image = np.frombuffer(color_msg.data, dtype=np.uint8).reshape(color_msg.height, color_msg.width, -1)
        self.cammera_info = info_msg

        # # Create a RealSense frame reference to work with the projection function
        # self.depth_frame = rs2.frame_data(self.depth_image)

        #rospy.loginfo("Received depth and color images")

        # Check if the images are empty
        if self.depth_image is None or self.color_image is None:
            rospy.logwarn("Empty images received.")
            return

        # # Unregister subscribers after receiving the first set of data
        # self.info_sub.unregister()
        # self.depth_sub.unregister()
        # self.color_sub.unregister()

    def handle_service(self, req):
        if self.color_image is None:
            rospy.logwarn("No image received yet.")
            return GroundingDINOResponse(cX=-1, cY=-1)

        image_pil = Img.fromarray(self.color_image)
        mask = self.process_image(image_pil, req.instruction)
        # Publish the mask
        mask_msg = Int32MultiArray()
        mask_msg.data = mask.flatten().tolist()
        mask_msg.layout.dim = [
            MultiArrayDimension(label='height', size=mask.shape[0], stride=mask.shape[0] * mask.shape[1]),
            MultiArrayDimension(label='width', size=mask.shape[1], stride=mask.shape[1])
        ]
        self.mask_pub.publish(mask_msg)
        print('Image has been processed.')

        indices = np.argwhere(mask)[2:, :].transpose(0, 1)

        points_3d = self.get_3d_points(indices, self.cammera_info)
        self.processed_point_cloud = points_3d
        self.publish_point_cloud()
        
        return GroundingDINOResponse(cX=self.cX, cY=self.cY)

    def get_3d_points(self, pixel_coords, camera_info):
        points_3d = []
        for [u, v] in pixel_coords:
            # depth = depth_image[v, u] / 1000
            x, y, z = self.convert_depth_to_phys_coord_using_realsense(u, v, camera_info)
            points_3d.append((x, y, z))
        return points_3d

    def convert_depth_to_phys_coord_using_realsense(self, x, y, cameraInfo):  

        # Extract the intrinsic matrix from CameraInfo
        # Get depth scale
        # depth_scale = 0.001  # Assuming depth scale is 0.001 (you might need to check this value)

        # # Get camera intrinsics
        # depth_intrin = rs2.intrinsics()
        # depth_intrin.width = self.depth_info.width
        # depth_intrin.height = self.depth_info.height
        # depth_intrin.ppx = self.depth_info.K[2]
        # depth_intrin.ppy = self.depth_info.K[5]
        # depth_intrin.fx = self.depth_info.K[0]
        # depth_intrin.fy = self.depth_info.K[4]
        # depth_intrin.model = rs2.distortion.brown_conrady
        # depth_intrin.coeffs = self.depth_info.D

        # color_intrin = rs2.intrinsics()
        # color_intrin.width = self.color_info.width
        # color_intrin.height = self.color_info.height
        # color_intrin.ppx = self.color_info.K[2]
        # color_intrin.ppy = self.color_info.K[5]
        # color_intrin.fx = self.color_info.K[0]
        # color_intrin.fy = self.color_info.K[4]
        # color_intrin.model = rs2.distortion.brown_conrady
        # color_intrin.coeffs = self.color_info.D

        # # Get extrinsics (assuming identity matrices if not available)
        # depth_to_color_extrin = rs2.extrinsics()
        # color_to_depth_extrin = rs2.extrinsics()


        _intrinsics = rs2.intrinsics()
        _intrinsics.width = cameraInfo.width
        _intrinsics.height = cameraInfo.height
        _intrinsics.ppx = cameraInfo.K[2]
        _intrinsics.ppy = cameraInfo.K[5]
        _intrinsics.fx = cameraInfo.K[0]
        _intrinsics.fy = cameraInfo.K[4]
        if cameraInfo.distortion_model == 'plumb_bob':
            _intrinsics.model = rs2.distortion.brown_conrady
        elif cameraInfo.distortion_model == 'equidistant':
            _intrinsics.model = rs2.distortion.kannala_brandt4
        rs2.coeffs = [i for i in cameraInfo.D]  

        # depth = rs2.rs2_project_color_pixel_to_depth_pixel(
        #     self.depth_frame, depth_scale,
        #     0.11, 2.0,  # depth_min, depth_max
        #     depth_intrin, color_intrin, depth_to_color_extrin, color_to_depth_extrin, [x, y])
        
        depth = self.depth_image[x, y]

        result = rs2.rs2_deproject_pixel_to_point(_intrinsics, [x, y], depth)  #result[0]: right, result[1]: down, result[2]: forward
        return result[2], -result[0], -result[1]

    def cluster_pub(self):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'realsense_wrist_link'
        pc_msg = PointCloud()
        pc_msg.header = header
        pc_msg.points = [Point32(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in self.processed_point_cloud]
        self.pc_pub.publish(pc_msg)

    def process_image(self, image, user_request):
        if image is None:
            rospy.logwarn("No image received yet.")
            return

        # Use GroundingDINO to find the ball
        output_dir = os.path.join("outputs/", user_request)
        os.makedirs(output_dir, exist_ok=True)

        results = self.detector.inference(image, user_request, cfg.box_threshold, cfg.text_threshold, cfg.iou_threshold)
        dino_pil = draw_candidate_boxes(image, results, output_dir, stepstr='nouns', save=False)

        results_ = []
        results_.append(results[0][0].unsqueeze(0))
        results_.append([results[1][0]])
        sin_pil = draw_candidate_boxes(image, results_, output_dir, stepstr='sing', save=False)
        mask = self.segmenter.inference(image, results_[0], results_[1], output_dir, save_json=True)

        # Save the original image
        original_image_path = os.path.join(output_dir, "original_image.png")
        image.save(original_image_path)

        # Convert mask to numpy array if not already
        mask = np.array(mask)
        mask = np.squeeze(mask)

        indices = np.argwhere(mask == 1)
        centroid = indices.mean(axis=0)
        self.cY, self.cX = centroid.astype(int)

        # Create a copy of the original image to draw on
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)

        # Define the radius of the circle
        radius = 1

        # Draw circles on the image where the mask is 1
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if mask[y, x] > 0:  # Adjust this condition based on mask value
                    upper_left = (x - radius, y - radius)
                    lower_right = (x + radius, y + radius)
                    draw.ellipse([upper_left, lower_right], outline="red", width=2)

        upper_left = (self.cX - 2, self.cY - 2)
        lower_right = (self.cX + 2, self.cY + 2)
        draw.ellipse([upper_left, lower_right], outline="yellow", width=2)

        # # Convert mask to PIL image
        # mask_pil = Img.fromarray(mask, mode='L')

        # # Ensure mask is the same size as the original image
        # mask_pil = mask_pil.resize(image.size, resample=Img.NEAREST)

        # # Create an overlay by combining the original image and the mask
        # mask_colored = ImageOps.colorize(mask_pil, (0, 0, 0), (255, 0, 0))
        # overlay_image = Img.composite(image, mask_colored, mask_pil)

        # Save the overlay image
        overlay_image_path = os.path.join(output_dir, "overlay_image.png")
        draw_image.save(overlay_image_path)
        return mask

    def publish_point_cloud(self):
        while not rospy.is_shutdown():
            self.cluster_pub()
            self.rate.sleep()

if __name__ == '__main__':
    point_cloud_publisher = Claw('pick up the left red ball')
