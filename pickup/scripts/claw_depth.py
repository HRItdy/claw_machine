#!/usr/bin/env python3
# 1-- Calculate the point cloud points corresponding to the mask. It will first convert the whole color frame into 3D pointcloud,
#     then match this pointcloud with the one projected from '/realsense_wrist/depth/image_rect_raw', get the transformation matrix (This should be done in the __init__).
# 2-- Call the service to get the mask, and calculate the corresponding pointcloud, then publish it or ?

import rospy
import numpy as np
from pickup.srv import Centroid, CentroidResponse
from geometry_msgs.msg import Point32
from std_msgs.msg import Header, Float32MultiArray
from sensor_msgs.msg import Image, PointCloud, CameraInfo
import argparse
from PIL import Image as Img, ImageOps, ImageDraw
import os
import copy
from message_filters import ApproximateTimeSynchronizer, Subscriber
import pyrealsense2 as rs2
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from store_mask_service import store_mask_client


class ClawDepth:
    def __init__(self):
        rospy.init_node('claw_depth', anonymous=True)
        self.color_image = None
        self.depth_image = None
        self.depth_sub = Subscriber('/realsense_wrist/depth/image_rect_raw', Image)  #TODO: project the 2D to 3D pointcloud, publish it and campare with the origin
        self.color_sub = Subscriber('/realsense_wrist/color/image_raw', Image)
        self.depth_info_sub = Subscriber('/realsense_wrist/depth/camera_info', CameraInfo)
        self.color_info_sub = Subscriber('/realsense_wrist/color/camera_info', CameraInfo)
        self.pc_pub = rospy.Publisher('/masked_point_cloud', PointCloud, queue_size=1)
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
        
        # First get the point cloud from depth frame: target point cloud
        target_cloud = self.depth_to_point_cloud(self.depth_image, self.depth_intrinsics)
        # Then get the point cloud from color frame: source point cloud
        source_cloud = self.color_to_point_cloud(self.depth_image, self.depth_intrinsics, self.color_image)
        # Finally calculate the transformation matrix between this two point cloud
        self.trans_matrix = self.align_pointclouds(source_cloud, target_cloud)
    
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
        # Convert the mask to numpy.array
        # Determine the number of rows from the stored matrix layout
        rows = mask.layout.dim[0].size
        # Calculate the number of columns based on the total data length divided by the number of rows
        cols = int(len(mask.data) / rows)
        # Convert the flat data list to a NumPy array and reshape it to the desired matrix shape
        mask = np.array(mask.data).reshape(rows, cols)
        
        indices = np.argwhere(mask)[2:, :].transpose(0, 1) # TODO: Check the coordinates after the transpose!
        # Get the coordinate in the pointcloud from color frame
        pc_color = self.color_to_point_cloud(indices, self.depth_image, self.depth_intrinsics)
        pc_color = np.hstack([pc_color, np.ones((pc_color.shape[0], 1))])
        # Apply the transformation matrix
        pc_depth = (self.trans_matrix @ pc_color.T).T
        # Convert back to 3D coordinates by removing the homogeneous component
        pc_depth = pc_depth[:, :3]
        self.pub_pc(pc_depth) # Will block the program. Only for result verifying here.
        centroid = np.mean(pc_depth, axis=0).reshape(1, 3)
        res_array = Float32MultiArray()
        res_array.data = centroid
        return CentroidResponse(array = res_array)
        
    def color_to_point_cloud(self, indices, depth_image, intrinsics, color_image = None):
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

    def cluster_pub(self, pc):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'realsense_wrist_link'
        pc_msg = PointCloud()
        pc_msg.header = header
        pc_msg.points = [Point32(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in pc]
        self.pc_pub.publish(pc_msg)

    def pub_pc(self, pc):
        while not rospy.is_shutdown():
            self.cluster_pub(pc)
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
        depth_in_meters = depth_image.flatten()
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
