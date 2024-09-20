#!/usr/bin/env python3
# 1-- Calculate the point cloud points corresponding to the mask. It will first convert the whole color frame into 3D pointcloud,
#     then match this pointcloud with the one projected from '/realsense_wrist/depth/image_rect_raw', get the transformation matrix (This should be done in the __init__).
# 2-- Call the service to get the mask, and calculate the corresponding pointcloud, then publish it or ?

import rospy
import numpy as np
from pickup.srv import Centroid, CentroidResponse
from geometry_msgs.msg import Point32
from std_msgs.msg import Header, Float32MultiArray
from sensor_msgs.msg import Image, PointCloud, CameraInfo, PointCloud2
import argparse
from PIL import Image as Img, ImageOps, ImageDraw
import os
import copy
from message_filters import ApproximateTimeSynchronizer, Subscriber
import pyrealsense2 as rs2
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from store_mask_service import store_mask_client
from sensor_msgs import point_cloud2
import tf2_ros
import tf.transformations 
from cv_bridge import CvBridge

class ClawDepth:
    def __init__(self):
        rospy.init_node('claw_depth', anonymous=True)
        self.bridge = CvBridge()
        self.color_image = None
        self.depth_image = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.depth_sub = Subscriber('/realsense_wrist/depth/image_rect_raw', Image)  #TODO: project the 2D to 3D pointcloud, publish it and campare with the origin
        self.color_sub = Subscriber('/realsense_wrist/color/image_raw', Image)
        self.depth_info_sub = Subscriber('/realsense_wrist/depth/camera_info', CameraInfo)
        self.color_info_sub = Subscriber('/realsense_wrist/color/camera_info', CameraInfo)
        self.pc_pub_depth = rospy.Publisher('/depth_point_cloud', PointCloud2, queue_size=1)
        self.pc_pub_color = rospy.Publisher('/color_point_cloud', PointCloud, queue_size=1)
        self.image_pub = rospy.Publisher('image_with_points', Image, queue_size=1)

        # Synchronize the topics
        self.ats = ApproximateTimeSynchronizer([self.depth_sub, self.color_sub, self.depth_info_sub, self.color_info_sub], queue_size=5, slop=0.1)
        self.ats.registerCallback(self.callback)
        # Service server  TODO
        self.service = rospy.Service('get_depth', Centroid, self.handle_service)
        self.rate = rospy.Rate(10)  # 10 Hz
        rospy.spin()

    def callback(self, depth_msg, color_msg, depth_info, color_info):
        # Convert ROS Image messages to OpenCV images manually
        self.depth_image = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(depth_msg.height, depth_msg.width)
        self.color_image = np.frombuffer(color_msg.data, dtype=np.uint8).reshape(color_msg.height, color_msg.width, -1)
        self.depth_intrinsics = self.camera_register(depth_info)
        self.color_intrinsics = self.camera_register(color_info)

        # Check if the images are empty
        if self.depth_image is None or self.color_image is None:
            rospy.logwarn("Empty images received.")
            return
        
        # If use self.color_instrinsics in self.color_to_point_cloud, the pointcloud will be zoomed campared with the pointcloud using depth_intrinsic
        # First get the point cloud from depth frame: target point cloud
        target_cloud = self.depth_to_point_cloud(self.depth_image, self.depth_intrinsics)
        # Transform the generated pointcloud under 'realsnse_wrist_depth_optical_frame' to 'realsense_wrist_link' frame
        converted_cloud = self.transform_point_cloud(target_cloud, source_frame='realsense_wrist_depth_optical_frame', target_frame='realsense_wrist_link')
        # 3D coordinates you want to project back to 2D image.
        points_3d = np.array([[0.5887, 0.1332, -0.0715],
                              [0.5839, -0.1712, -0.0875],
                              [0.5284, -0.1678, 0.1503]])
        # Now under realsense_wrist_link frame, convert to realsense_wrist_depth_optical_frame
        converted_point_3d = self.transform_point_cloud(points_3d, source_frame='realsense_wrist_link', target_frame='realsense_wrist_depth_optical_frame')
        points_2d = self.project_3d_to_2d(converted_point_3d, self.depth_intrinsics)
        print(points_2d)
        pil_image = Img.fromarray(self.color_image)
        pil_image = self.draw_points_on_image(pil_image, points_2d)
        self.publish_image(pil_image)

    def project_3d_to_2d(self, points_3d, intrinsics):
        """
        Projects 3D points into 2D image coordinates.
        """
        x = points_3d[:, 0]
        y = points_3d[:, 1]
        z = points_3d[:, 2] / 0.001

        fx, fy = intrinsics.fx, intrinsics.fy
        cx, cy = intrinsics.ppx, intrinsics.ppy

        u = (x * fx / z) + cx
        v = (y * fy / z) + cy

        points_2d = np.vstack((u, v)).T
        return points_2d

    def draw_points_on_image(self, pil_img, points_2d):
        """
        Draws circles on the image at the projected 2D points.
        """
        draw = ImageDraw.Draw(pil_img)
        for point in points_2d:
            u, v = int(point[0]), int(point[1])
            radius = 5
            draw.ellipse((u - radius, v - radius, u + radius, v + radius), outline="green", width=2)
        return pil_img

    def publish_image(self, pil_image):
        """
        Publishes the processed image.
        """
        # Convert the PIL image back to a ROS Image message and publish
        np_image = np.array(pil_image)
        image_msg = self.bridge.cv2_to_imgmsg(np_image, "rgb8")
        self.image_pub.publish(image_msg)

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

            # Publish the transformed point cloud
            self.publish_point_cloud(transformed_points)
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

    def publish_point_cloud(self, points):
        """Publish the point cloud as sensor_msgs/PointCloud2"""
        header = rospy.Header()
        header.frame_id = 'realsense_wrist_link'
        header.stamp = rospy.Time.now()

        # Convert numpy array to PointCloud2 message
        pointcloud_msg = point_cloud2.create_cloud_xyz32(header, points.tolist())

        # Publish the message
        self.pc_pub_depth.publish(pointcloud_msg)
        
    def align_pointclouds(self, source_cloud, target_cloud):
        # Convert numpy arrays to Open3D point clouds
        source_pcd = o3d.geometry.PointCloud()
        target_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source_cloud)
        target_pcd.points = o3d.utility.Vector3dVector(target_cloud)

        # Perform ICP alignment
        threshold = 0.02  # distance threshold
        trans_matrix = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, threshold, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()).transformation
        return trans_matrix

    # The service to get the depth
    def handle_service(self, req):
        # get the mask from the service
        mask, success = store_mask_client(store=False)
        indices = np.argwhere(mask)[2:, :].transpose(0, 1) # TODO: Check the coordinates after the transpose!
        # Get the coordinate in the pointcloud from color frame
        pc_color = self.color_to_point_cloud(self.depth_image, self.depth_intrinsics, indices=indices, color_image=None)
        pc_color = np.hstack([pc_color, np.ones((pc_color.shape[0], 1))])
        # Apply the transformation matrix
        # pc_depth = (self.trans_matrix @ pc_color.T).T
        # # Convert back to 3D coordinates by removing the homogeneous component
        # pc_depth = pc_depth[:, :3]
        self.cluster_pub(pc_color) # Result verifying here.
        centroid = np.mean(pc_color, axis=0).reshape(1, 3)
        res_array = Float32MultiArray()
        res_array.data = centroid
        return CentroidResponse(array = res_array)
        
    def color_to_point_cloud(self, depth_image, intrinsics, indices = None, color_image = None):
        if color_image is not None:
            height, width, _ = color_image.shape
            # Convert the whole color frame to 3D point cloud
            u_list = [u for u in range(width)]
            v_list = [v for v in range(height)]
        else:
            u_list = [u[0] for u in indices]
            v_list = [v[0] for v in indices]
        point_cloud = []
        for v in v_list:
            for u in u_list:
                if depth_image[v, u] > 0:
                    point = rs2.rs2_deproject_pixel_to_point(intrinsics, [u, v], float(depth_image[v, u]) * 0.001)
                    point_cloud.append([point[0], point[1], point[2]])
        return np.array(point_cloud)

    def cluster_pub(self, pc, pub):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'realsense_wrist_link'
        pc_msg = PointCloud()
        pc_msg.header = header
        pc_msg.points = [Point32(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in pc]
        pub.publish(pc_msg)

    def pub_pc(self, pc, pub):
        while not rospy.is_shutdown():
            self.cluster_pub(pc, pub)
            self.rate.sleep()

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

    # def convert_to_numpy(self, cloud_msg):
    #     # Convert PointCloud to numpy array
    #     points = np.array([[p.x, p.y, p.z] for p in cloud_msg.points])
    #     return points

if __name__ == '__main__':
    ClawDepth()
