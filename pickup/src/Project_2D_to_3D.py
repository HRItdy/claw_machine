#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge, CvBridgeError
import sensor_msgs.point_cloud2 as pc2
import message_filters
from pickup.src.store_mask_service import store_mask_client

class DepthToColorRegistration:
    def __init__(self):
        # Initialize node
        rospy.init_node('depth_to_color_register', anonymous=True)

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Get parameters
        self.rgb_camera_info_topic = rospy.get_param("~rgb_camera_info_topic", "/realsense_wrist/color/camera_info")
        self.depth_camera_info_topic = rospy.get_param("~depth_camera_info_topic", "/realsense_wrist/depth/camera_info")
        self.depth_image_topic = rospy.get_param("~depth_image_topic", "/realsense_wrist/depth/image_rect_raw")
        self.rgb_image_topic = rospy.get_param("~rgb_image_topic", "/realsense_wrist/color/image_raw")
        self.pointcloud_topic = rospy.get_param("~pointcloud_topic", "/camera/depth_registered/segment_points")

        # Subscribers
        self.rgb_camera_info_sub = message_filters.Subscriber(self.rgb_camera_info_topic, CameraInfo)
        self.depth_camera_info_sub = message_filters.Subscriber(self.depth_camera_info_topic, CameraInfo)
        self.depth_image_sub = message_filters.Subscriber(self.depth_image_topic, Image)
        self.rgb_image_sub = message_filters.Subscriber(self.rgb_image_topic, Image)

        # Synchronize the topics
        self.sync = message_filters.ApproximateTimeSynchronizer([self.rgb_camera_info_sub, self.depth_camera_info_sub, self.depth_image_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.callback)

        # Publisher
        self.pointcloud_pub = rospy.Publisher(self.pointcloud_topic, PointCloud2, queue_size=1)

        mask, success = store_mask_client(store=False)
        indices = np.argwhere(mask)[2:, :].transpose(0, 1) # TODO: Check the coordinates after the transpose!
        self.u_v_list = [[item[0], item[1]] for item in indices]
  
    def callback(self, rgb_info_msg, depth_info_msg, depth_msg):
        try:
            # Convert depth image to OpenCV format
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

        # Camera intrinsics
        K_rgb = np.array(rgb_info_msg.K).reshape((3, 3))

        # Camera extrinsics
        R = np.array(depth_info_msg.R).reshape((3, 3))
        T = np.array(depth_info_msg.P).reshape((3, 4))[:, 3]

        # Create point cloud from depth image and RGB camera parameters using the u_v_list
        point_cloud = self.create_point_cloud(self.u_v_list, depth_image, K_rgb, R, T)

        # Publish the point cloud
        self.pointcloud_pub.publish(point_cloud)

    def create_point_cloud(self, u_v_list, depth_image, K_rgb, R, T):
        points = []

        for (u, v) in u_v_list:
            u = int(u)
            v = int(v)

            # Ensure pixel is within the image bounds
            if 0 <= u < depth_image.shape[1] and 0 <= v < depth_image.shape[0]:
                depth_value = depth_image[v, u] / 1000.0  # Convert depth to meters

                if depth_value > 0.01:  # Filter out invalid depth values
                    # Convert 2D pixel to 3D point
                    point_3d = self.pixel_to_3d_point(u, v, depth_value, K_rgb, R, T)
                    points.append(point_3d)

        # Create PointCloud2 message
        point_cloud_msg = self.create_pointcloud2_msg(points)

        return point_cloud_msg

    def pixel_to_3d_point(self, u, v, depth_value, K_rgb, R, T):
        # Step 1: Back-project to 3D in the RGB camera frame
        x_rgb = (u - K_rgb[0, 2]) * depth_value / K_rgb[0, 0]
        y_rgb = (v - K_rgb[1, 2]) * depth_value / K_rgb[1, 1]
        z_rgb = depth_value

        # Step 2: Construct the point in the RGB camera frame
        point_rgb_frame = np.array([x_rgb, y_rgb, z_rgb])

        # Step 3: Apply the inverse transformation to get the point in the depth camera frame
        point_depth_frame = np.dot(np.linalg.inv(R), (point_rgb_frame - T))

        return point_depth_frame

    def create_pointcloud2_msg(self, points):
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "realsense_wrist_color_optical_frame"  # Use the correct frame ID

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]

        # Create the PointCloud2 message
        point_cloud_msg = pc2.create_cloud(header, fields, points)
        return point_cloud_msg

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        depth_to_color_registration = DepthToColorRegistration()
        depth_to_color_registration.run()
    except rospy.ROSInterruptException:
        pass