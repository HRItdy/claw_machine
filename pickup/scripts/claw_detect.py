#!/usr/bin/env python3
# This script is only responsable for the detection. It will get the mask using GroundingDINO and SAM based on the input instruction, 
# and the generated mask will be stored in one service.

import rospy
import numpy as np
import torch
from colorama import init as colorama_init
from colorama import Fore, Style
from models import runGroundingDino, GroundedDetection, DetPromptedSegmentation, draw_candidate_boxes, OpenOWLDetection, GPT4Reasoning, FastSAMSegment
from pickup.srv import GroundingDINO, GroundingDINOResponse, SamPoint, SamPointResponse, OwlGpt, OwlGptResponse, Get3DPosition, Get3DPositionResponse 
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import Header, Bool, Int32MultiArray, MultiArrayDimension
from sensor_msgs.msg import Image, PointCloud, CameraInfo, PointCloud2
from sensor_msgs import point_cloud2
import argparse
from message_filters import ApproximateTimeSynchronizer, Subscriber
from PIL import Image as Img, ImageOps, ImageDraw
import tf2_ros
import tf.transformations 
import os
import copy
import cv2
from cv_bridge import CvBridge
import pyrealsense2 as rs2


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
colorama_init()
parser = argparse.ArgumentParser("GroundingDINO", add_help=True)
parser.add_argument("--debug", action="store_true", help="using debug mode")
parser.add_argument("--share", action="store_true", help="share the app")
parser.add_argument("--box_threshold", type=float, default=0.25, help="box threshold")
parser.add_argument("--text_threshold", type=float, default=0.2, help="text threshold")
parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold")
parser.add_argument("--visualize", default=False, help="visualize intermediate data mode")
parser.add_argument("--device", type=str, default="cuda:0", help="run on: 'cuda:0' for GPU 0 or 'cpu' for CPU. Default GPU 0.")
cfg = parser.parse_args()

class ClawDetect:
    def __init__(self, instruct):
        rospy.init_node('claw_detect', anonymous=True)
        # Initialize components
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.bridge = CvBridge()
        self.detector = GroundedDetection(cfg)
        self.segmenter = DetPromptedSegmentation(cfg)
        self.owlvit = OpenOWLDetection()
        self.gpt = GPT4Reasoning()
        self.fastsam = FastSAMSegment()
        print(f'{Fore.YELLOW}All models are initialized!{Style.RESET_ALL}')
        # Register the camera info
        cameraInfo = rospy.wait_for_message('/realsense_wrist/color/camera_info', CameraInfo, timeout=5)
        print(cameraInfo)
        self.color_intrinsic = self.camera_register(cameraInfo)
        print(f'{Fore.YELLOW}Camera has been registered successfully.{Style.RESET_ALL}')
        self.color_image = None
        # Get the instruction input
        self.instruct = instruct
        self.color_sub = Subscriber('/realsense_wrist/color/image_raw', Image)
        self.depth_sub = Subscriber('/realsense_wrist/aligned_depth_to_color/image_raw', Image)
        # Synchronize the topics
        self.ats = ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=5, slop=0.1)
        self.ats.registerCallback(self.callback)
        self.image_pub = rospy.Publisher('/masked_image', Image, queue_size=10) 
        self.center_pub = rospy.Publisher('/3d_centroid', PointStamped, queue_size=1)
        # Service server
        self.gr_service = rospy.Service('grounding_dino', GroundingDINO, self.handle_gr_service)
        self.sam_service = rospy.Service('sam_point', SamPoint, self.handle_sam_service)
        self.owl_service = rospy.Service('owl_gpt', OwlGpt, self.handle_owl_service)
        # Service to get 3D grasping position
        self.td_service = rospy.Service('get_3d_position', Get3DPosition, self.handle_td_service)
        self.td_pub = rospy.Publisher('/crop_cloud', PointCloud2, queue_size=10) 
        self.rate = rospy.Rate(10) 
        rospy.spin()

    def callback(self, color_msg, depth_msg):
        self.color_image = np.frombuffer(color_msg.data, dtype=np.uint8).reshape(color_msg.height, color_msg.width, -1)
        self.depth_image = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(depth_msg.height, depth_msg.width)
        # Check if the images are empty
        if self.color_image is None:
            rospy.logwarn("Empty images received.")
            return
        
        # Check if the view png exists, if not, save the current image
        ppng = os.path.join(os.path.expanduser("~"), "claw_machine/src/pickup/scripts/cache/view.png")
        if not os.path.exists(ppng):
            # Convert the image to PIL format and save it as 'a.png'
            img = Img.fromarray(self.color_image)
            img.save(ppng)
        else:
            pass 
        
    def handle_gr_service(self, req):
        if self.color_image is None:
            rospy.logwarn("No image received yet.")
            return GroundingDINOResponse(cX=-1, cY=-1)
        image_pil = Img.fromarray(self.color_image)
        mask = self.process_image(image_pil, req)
        rospy.set_param('/pc_transform/image_mask', mask.tolist())
        masked_img = self.segmenter.get_image(image_pil, mask)
        # Convert the processed image to a ROS Image message and publish it
        masked_img = np.array(masked_img)
        ros_image = self.bridge.cv2_to_imgmsg(masked_img, encoding="rgb8")
        self.image_pub.publish(ros_image)
        print('Image has been processed.')
        return GroundingDINOResponse(cX=self.cX, cY=self.cY)
    
    def handle_sam_service(self, req):
        if self.color_image is None:
            rospy.logwarn("No image received yet.")
            return SamPointResponse()
        image_pil = Img.fromarray(self.color_image)
        point_coord = np.array([[req.cX, req.cY]])
        point_label = np.array([1])
        if req.fast_sam:
            ann = self.fastsam.predict_point(image_pil, point_coord, point_label)
            mask = np.squeeze(ann)
            rospy.set_param('/pc_transform/image_mask', mask.tolist())
            masked_img = self.fastsam.get_image(image_pil, mask)
        else:
            mask = self.segmenter.inference_point(image_pil, point_coord, point_label)
            mask = np.squeeze(mask)
            rospy.set_param('/pc_transform/image_mask', mask.tolist())
            masked_img = self.segmenter.get_image(image_pil, mask)
        # Convert the processed image to a ROS Image message and publish it
        masked_img = np.array(masked_img)
        ros_image = self.bridge.cv2_to_imgmsg(masked_img, encoding="rgb8")
        self.image_pub.publish(ros_image)
        print('Image has been processed.')
        return SamPointResponse()
    
    def handle_owl_service(self, req):
        # req.enhance = False: detect all the balls and visualize them.
        # req.enhance = True: gpt will detect the only one target, and then trigger SAM to generate mask.
        if self.color_image is None:
            rospy.logwarn("No image received yet.")
            return OwlGptResponse()
        image_pil = Img.fromarray(self.color_image)
        prompt = req.instruction
        prompt = prompt.split(',')
        output = self.owlvit.inference(image_pil, prompt[0:2])
        boxes, labels, scores = self.owlvit.save_mask_json("/home/lab_cheem/claw_machine/src/pickup/scripts/cache", output, prompt)
        masked_img = self.owlvit.draw_boxes(image_pil, boxes, labels)
        # publish the image with all objects marked
        # masked_img = np.array(masked_img)
        # ros_image = self.bridge.cv2_to_imgmsg(masked_img, encoding="rgb8")
        # self.image_pub.publish(ros_image)
        if not req.enhance:
            print('Image has been processed.')
            return OwlGptResponse()
        else:
            # enhance with gpt
            infer = self.gpt.GroundedSAM_json_asPrompt(prompt[-1])
            boxes, labels, scores, obj_captions = self.gpt.parse_output(infer)
            # get the most confidiential target
            # box = torch.as_tensor([boxes[0]])
            box = [boxes[0]]
            obj = [obj_captions[0]]
            if req.fast_sam:
                ann = self.fastsam.predict_box(image_pil, box)
                mask = np.squeeze(ann)
                masked_img = self.fastsam.get_image(image_pil, mask)
            else:
                mask = self.segmenter.inference(image_pil, box, obj, None, save_json=False)
                mask = np.array(mask)
                mask = np.squeeze(mask)
                masked_img = self.segmenter.get_image(image_pil, mask)
            # Convert the processed image to a ROS Image message and publish it
            masked_img = np.array(masked_img)
            ros_image = self.bridge.cv2_to_imgmsg(masked_img, encoding="rgb8")
            self.image_pub.publish(ros_image)
            # bottom = self.find_bottom_point(mask)
            rospy.set_param('/pc_transform/image_mask', mask.tolist())
            # rospy.set_param('/pc_transform/bottom', bottom.tolist())
            print('Image has been processed.')
            return OwlGptResponse()
        
    def handle_td_service(self, req):
        # Get the mask from parameter server
        if rospy.has_param('/pc_transform/image_mask'):
            mask = np.array(rospy.get_param('/pc_transform/image_mask'))
            indices = np.argwhere(mask)[2:, :] 
            indices = indices[:, [1, 0]] # Convert [a,b] to [b,a]
            #np.save('indices', indices)
        else:
            rospy.logerr("No mask generated yet. Please call detection service first")
            return Get3DPositionResponse(Point(-1, -1, -1))
        # Transform te 2d mask to the 3d points under `realsense_wrist_color_optical_link` frame
        # points_2d = np.array([[226, 208],
        #                       [404, 425]])
        # mask_cloud = self.point_to_point_cloud(points_2d, self.depth_image, self.color_intrinsic)
        mask_cloud = self.point_to_point_cloud(indices, self.depth_image, self.color_intrinsic)
        # Transform the 3d points from frame `realsense_wrist_color_optical_link` to `realsense_wrist_link`
        converted_cloud = self.transform_point_cloud(mask_cloud, source_frame='realsense_wrist_color_optical_frame', target_frame='realsense_wrist_link')
        # Convert the depth image to point cloud to verify
        # depth_cloud = self.depth_to_point_cloud(self.depth_image, self.color_intrinsic)
        # converted_depth_cloud = self.transform_point_cloud(depth_cloud, source_frame='realsense_wrist_color_optical_frame', target_frame='realsense_wrist_link')
        # self.publish_point_cloud(converted_cloud, self.td_pub)
        print(f'{Fore.YELLOW}Pointcloud has been cropped.{Style.RESET_ALL}')
        # Estimate the grasping position
        centroid = np.mean(converted_cloud, axis=0)
        centroid = self.adjust(centroid, 0.005, 0.005, -0.003)
        self.publish_centroid_point(centroid, self.center_pub)
        return Get3DPositionResponse(Point(*centroid))

    def adjust(self, point, dx, dy, dz):
        point[0] = point[0] + dx
        point[1] = point[1] + dy
        point[2] = point[2] + dz
        return point        
        
    def publish_point_cloud(self, points, pub):
        """Publish the point cloud as sensor_msgs/PointCloud2"""
        header = rospy.Header()
        header.frame_id = 'realsense_wrist_link'
        header.stamp = rospy.Time.now()
        # Convert numpy array to PointCloud2 message
        pointcloud_msg = point_cloud2.create_cloud_xyz32(header, points.tolist())
        # Publish the message
        pub.publish(pointcloud_msg)

    def publish_centroid_point(self, centroid, pub):
        """Publish the point cloud as PointStamped"""    
        # Create the PointStamped message
        point_stamped = PointStamped()
        point_stamped.header.stamp = rospy.Time.now()
        point_stamped.header.frame_id = 'realsense_wrist_link'
        point_stamped.point.x = centroid[0]
        point_stamped.point.y = centroid[1]
        point_stamped.point.z = centroid[2]
        # Publish the transformed point
        pub.publish(point_stamped)
    
    def depth_to_point_cloud(self, depth_image, intrinsics):
        """
        Projects a 2D depth image into 3D space.

        :param depth_image: 2D depth image (numpy array).
        :param camera_intrinsics: Camera intrinsic matrix (3x3 numpy array).
        :return: 3D points (Nx3 numpy array).
        """
        # Get the depth image dimensions
        height, width = depth_image.shape
        # Generate a grid of (u, v) coordinates corresponding to the pixels
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        # Flatten the u, v arrays
        u_flat = u.flatten()
        v_flat = v.flatten()
        # Convert depth units from millimeters to meters (if needed)
        depth_in_meters = depth_image.flatten() * 0.001
        # Intrinsic matrix parameters
        fx, fy = intrinsics.fx, intrinsics.fy
        cx, cy = intrinsics.ppx, intrinsics.ppy
        # Calculate the 3D points
        z = depth_in_meters
        x = (u_flat - cx) * z / fx
        y = (v_flat - cy) * z / fy
        # Stack the x, y, z arrays into a single (N, 3) array
        points_3d = np.vstack((x, y, z)).T
        return points_3d

    def point_to_point_cloud(self, points_2d, depth_image, intrinsics):
        """
        Projects 2D points into 3D space.

        :param depth_image: 2D depth image (numpy array).
        :param camera_intrinsics: Camera intrinsic matrix (3x3 numpy array).
        :return: 3D points (Nx3 numpy array).
        """
        # Convert depth units from millimeters to meters (if needed)
        # Intrinsic matrix parameters
        fx, fy = intrinsics.fx, intrinsics.fy
        cx, cy = intrinsics.ppx, intrinsics.ppy
        points_3d = []
        # Calculate the 3D points
        for point in points_2d:
            z = depth_image[point[1], point[0]] * 0.001
            x = (point[0] - cx) * z / fx
            y = (point[1] - cy) * z / fy
            points_3d.append([x,y,z])
        # Stack the x, y, z arrays into a single (N, 3) array
        points = np.array(points_3d)
        return points
    
    def transform_point_cloud(self, target_cloud, source_frame, target_frame):
        try:
            # Get the transform from 'realsense_wrist_depth_optical_frame' to 'realsense_wrist_link'
            # transform = self.tf_buffer.lookup_transform('realsense_wrist_link', 
            #                                             'realsense_wrist_depth_optical_frame', 
            #                                             rospy.Time(0), 
            #                                             rospy.Duration(1.0))
            transform = self.tf_buffer.lookup_transform(target_frame, 
                                                        source_frame, 
                                                        rospy.Time(0), 
                                                        rospy.Duration(1.0))
            # Convert the TransformStamped message to a 4x4 transformation matrix
            transform_matrix = self.transform_to_matrix(transform)
            
            # Apply the transformation to the point cloud (n, 3)
            n_points = target_cloud.shape[0]
            ones_column = np.ones((n_points, 1))
            homogenous_points = np.hstack((target_cloud, ones_column))  # Convert to homogeneous coordinates (n, 4)
            transformed_points = (transform_matrix @ homogenous_points.T).T  # Apply transformation
            transformed_points = transformed_points[:, :3]  # Back to (n, 3) by removing homogeneous component

            # # Publish the transformed point cloud
            # self.publish_point_cloud(transformed_points)
            return transformed_points
        
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logerr("Failed to get transform")

    def transform_to_matrix(self, transform):
        """Convert a TransformStamped message to a 4x4 transformation matrix"""
        translation = np.array([transform.transform.translation.x,
                                transform.transform.translation.y,
                                transform.transform.translation.z])
        
        rotation = np.array([transform.transform.rotation.x,
                             transform.transform.rotation.y,
                             transform.transform.rotation.z,
                             transform.transform.rotation.w])
        
        # Create 4x4 transformation matrix
        transform_matrix = tf.transformations.quaternion_matrix(rotation)
        transform_matrix[0:3, 3] = translation
        return transform_matrix

    def camera_register(self, cameraInfo):
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
        _intrinsics.coeffs = [i for i in cameraInfo.D]  
        return _intrinsics
    
    def process_image(self, image, request):
        if image is None:
            rospy.logwarn("No image received yet.")
            return
        # Extract request info
        user_request = request.instruction
        fast_sam = request.fast_sam
        # Use GroundingDINO to find the ball
        output_dir = os.path.join("outputs/", user_request)
        os.makedirs(output_dir, exist_ok=True)
        results = self.detector.inference(image, user_request, cfg.box_threshold, cfg.text_threshold, cfg.iou_threshold)
        dino_pil = draw_candidate_boxes(image, results, output_dir, stepstr='nouns', save=False)
        results_ = []
        results_.append(results[0][0].unsqueeze(0))
        results_.append([results[1][0]])
        sin_pil = draw_candidate_boxes(image, results_, output_dir, stepstr='sing', save=False)
        if fast_sam:
            mask = self.fastsam.predict_prompt(image, user_request)
        else:
            mask = self.segmenter.inference(image, results_[0], results_[1], output_dir, save_json=True)
        # Save the original image
        original_image_path = os.path.join(output_dir, "original_image.png")
        image.save(original_image_path)
        # Convert mask to numpy array if not already
        mask = np.array(mask)
        mask = np.squeeze(mask)
        # bottom = self.find_bottom_point(mask)
        rospy.set_param('/pc_transform/image_mask', mask.tolist())
        # rospy.set_param('/pc_transform/bottom', bottom.tolist())
        # rospy.loginfo('Store bottom point to /pc_transform/bottom')
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
        overlay_image_path = os.path.join(output_dir, "overlay_image.png")
        draw_image.save(overlay_image_path)
        return mask
    
    def find_bottom_point(self, mask):
        # Find the indices where the mask is True
        y_indices, x_indices = np.where(mask)
        if len(y_indices) == 0:
            return None  # No points found in the mask
        # Find the maximum y-index, which corresponds to the bottom point
        max_y_index = np.argmax(y_indices)
        # Return the (x, y) coordinate of the bottom point
        bottom_point = (x_indices[max_y_index], y_indices[max_y_index])
        return np.array(bottom_point)
    
    # def find_bottom_point(self, mask):
    #     # Find contours
    #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     if len(contours) == 0:
    #         return None
    #     # Get the largest contour (assuming the circle is the largest object in the mask)
    #     contour = max(contours, key=cv2.contourArea)
    #     # Find the bottommost point
    #     bottom_point = np.array(contour[contour[:, :, 1].argmax()][0])
    #     return bottom_point
    
    def find_bottom_point_of_fitted_circle(mask):  # can be used if the circle is blocked by other circles
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        # Get the largest contour (assuming the circle is the largest object in the mask)
        contour = max(contours, key=cv2.contourArea)
        # Fit a circle to the contour points
        (x, y), radius = cv2.minEnclosingCircle(contour)
        # Calculate the bottommost point of the fitted circle
        bottom_point = (int(x), int(y + radius))
        return bottom_point, (int(x), int(y)), int(radius)
    
if __name__ == '__main__':
    point_cloud_publisher = ClawDetect('pick up the left red ball')
