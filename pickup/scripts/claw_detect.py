#!/usr/bin/env python3
# This script is only responsable for the detection. It will get the mask using GroundingDINO and SAM based on the input instruction, 
# and the generated mask will be stored in one service.

import rospy
import numpy as np
from models import runGroundingDino, GroundedDetection, DetPromptedSegmentation, draw_candidate_boxes
from pickup.srv import GroundingDINO, GroundingDINOResponse, StoreMask
from geometry_msgs.msg import Point32
from std_msgs.msg import Header, Bool, Int32MultiArray, MultiArrayDimension
from sensor_msgs.msg import Image, PointCloud, CameraInfo
import argparse
from PIL import Image as Img, ImageOps, ImageDraw
from store_mask_service import store_mask_client
import os
import copy

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

class ClawDetect:
    def __init__(self, instruct):
        rospy.init_node('claw_detect', anonymous=True)
        # Initialize components
        self.detector = GroundedDetection(cfg)
        self.segmenter = DetPromptedSegmentation(cfg)
        self.color_image = None
        # Get the instruction input
        self.instruct = instruct
        self.color_sub = rospy.Subscriber('/realsense_wrist/color/image_raw', Image, self.callback)
        # Service server
        self.service = rospy.Service('grounding_dino', GroundingDINO, self.handle_service)
        self.rate = rospy.Rate(10)  # 10 Hz
        rospy.spin()

    def callback(self, color_msg):
        self.color_image = np.frombuffer(color_msg.data, dtype=np.uint8).reshape(color_msg.height, color_msg.width, -1)
        # Check if the images are empty
        if self.color_image is None:
            rospy.logwarn("Empty images received.")
            return
   
    def handle_service(self, req):
        if self.color_image is None:
            rospy.logwarn("No image received yet.")
            return GroundingDINOResponse(cX=-1, cY=-1)
        image_pil = Img.fromarray(self.color_image)
        mask = self.process_image(image_pil, req.instruction)
        # Call the store_mask_service to store the mask
        stored_mask, success = store_mask_client(mask, store=True)
        print('Image has been processed.')
        return GroundingDINOResponse(cX=self.cX, cY=self.cY)

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

if __name__ == '__main__':
    point_cloud_publisher = ClawDetect('pick up the left red ball')
