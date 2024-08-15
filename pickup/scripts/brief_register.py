#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge, CvBridgeError
import sensor_msgs.point_cloud2 as pc2
import message_filters

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
        self.registered_depth_image_topic = rospy.get_param("~registered_depth_image_topic", "/camera/depth_registered/image_rect")
        self.pointcloud_topic = rospy.get_param("~pointcloud_topic", "/camera/depth_registered/cutom_points")

        # Subscribers
        self.rgb_camera_info_sub = message_filters.Subscriber(self.rgb_camera_info_topic, CameraInfo)
        self.depth_camera_info_sub = message_filters.Subscriber(self.depth_camera_info_topic, CameraInfo)
        self.depth_image_sub = message_filters.Subscriber(self.depth_image_topic, Image)
        self.rgb_image_sub = message_filters.Subscriber(self.rgb_image_topic, Image)

        # Synchronize the topics
        self.sync = message_filters.ApproximateTimeSynchronizer([self.rgb_camera_info_sub, self.depth_camera_info_sub, self.depth_image_sub, self.rgb_image_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.callback)

        # Publishers
        self.registered_depth_pub = rospy.Publisher(self.registered_depth_image_topic, Image, queue_size=1)
        self.pointcloud_pub = rospy.Publisher(self.pointcloud_topic, PointCloud2, queue_size=1)

    def callback(self, rgb_info_msg, depth_info_msg, depth_msg, rgb_msg):
        try:
            # Convert depth image to OpenCV format
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

        # Camera intrinsics
        K_rgb = np.array(rgb_info_msg.K).reshape((3, 3))
        K_depth = np.array(depth_info_msg.K).reshape((3, 3))

        # Camera extrinsics
        R = np.array(depth_info_msg.R).reshape((3, 3))
        T = np.array(depth_info_msg.P).reshape((3, 4))[:, 3]

        # Convert depth image to point cloud in depth camera frame
        height, width = depth_image.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        z = depth_image / 1000.0  # Assuming depth is in millimeters, convert to meters

        # Filter out invalid depth values
        valid_mask = z > 0.01  # Exclude very small depth values
        x = np.where(valid_mask, (u - K_depth[0, 2]) * z / K_depth[0, 0], np.nan)
        y = np.where(valid_mask, (v - K_depth[1, 2]) * z / K_depth[1, 1], np.nan)

        points_3d = np.stack((x, y, z), axis=-1)

        # Transform points to RGB camera frame
        points_3d_rgb_frame = np.dot(R, points_3d.reshape(-1, 3).T).T + T
        points_3d_rgb_frame = points_3d_rgb_frame.reshape(height, width, 3)

        # Project points to RGB image plane
        u_rgb = (points_3d_rgb_frame[..., 0] * K_rgb[0, 0] / points_3d_rgb_frame[..., 2]) + K_rgb[0, 2]
        v_rgb = (points_3d_rgb_frame[..., 1] * K_rgb[1, 1] / points_3d_rgb_frame[..., 2]) + K_rgb[1, 2]

        # Generate registered depth image
        registered_depth_image = np.zeros_like(depth_image)

        for i in range(height):
            for j in range(width):
                if np.isnan(u_rgb[i, j]) or np.isnan(v_rgb[i, j]):
                    continue  # Skip invalid projections

                u_r = int(round(u_rgb[i, j]))
                v_r = int(round(v_rgb[i, j]))

                if 0 <= u_r < width and 0 <= v_r < height:
                    registered_depth_image[v_r, u_r] = depth_image[i, j]

        try:
            # Convert registered depth image to ROS Image message
            registered_depth_msg = self.bridge.cv2_to_imgmsg(registered_depth_image, encoding="passthrough")
            # Publish the registered depth image
            self.registered_depth_pub.publish(registered_depth_msg)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

        # Generate and publish PointCloud2 message
        self.publish_point_cloud(registered_depth_image, points_3d_rgb_frame)



    def publish_point_cloud(self, registered_depth_image, points_3d_rgb_frame):
        # Flatten the arrays to loop over each point
        points = points_3d_rgb_frame.reshape(-1, 3)
        depth = registered_depth_image.flatten()

        # Filter out points with invalid depth
        valid_points = points[depth > 0]

        # Create the point cloud message
        cloud_msg = PointCloud2()
        cloud_msg.header.stamp = rospy.Time.now()
        cloud_msg.header.frame_id = "realsense_wrist_color_optical_frame"  # Adjust this to match your TF frame

        # Define the fields of the point cloud
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]

        # Create the point cloud message
        cloud_msg = pc2.create_cloud(cloud_msg.header, fields, valid_points)

        # Publish the point cloud message
        self.pointcloud_pub.publish(cloud_msg)

    def run(self):
        rospy.spin()

    def pixel_to_3d_point(u, v, depth_value, K_rgb, R, T):
        """
        Convert 2D pixel coordinates (u, v) and depth value to 3D point in the camera frame.

        Parameters:
        - u: pixel x-coordinate in the RGB image
        - v: pixel y-coordinate in the RGB image
        - depth_value: depth value at (u, v) in meters
        - K_rgb: intrinsic camera matrix of the RGB camera (3x3)
        - R: rotation matrix (3x3) from depth camera to RGB camera
        - T: translation vector (3x1) from depth camera to RGB camera

        Returns:
        - 3D point in the depth camera frame as a numpy array [x, y, z]
        """

        # Step 1: Back-project to 3D in the RGB camera frame
        x_rgb = (u - K_rgb[0, 2]) * depth_value / K_rgb[0, 0]
        y_rgb = (v - K_rgb[1, 2]) * depth_value / K_rgb[1, 1]
        z_rgb = depth_value

        # Step 2: Construct the point in the RGB camera frame
        point_rgb_frame = np.array([x_rgb, y_rgb, z_rgb])

        # Step 3: Apply the inverse transformation to get the point in the depth camera frame
        point_depth_frame = np.dot(np.linalg.inv(R), (point_rgb_frame - T))

        return point_depth_frame

# Example usage
u = 320  # example pixel x-coordinate
v = 240  # example pixel y-coordinate
depth_value = 1.5  # example depth value in meters

# Assuming we have these matrices from the camera info
K_rgb = np.array([[600, 0, 320],
                  [0, 600, 240],
                  [0, 0, 1]])

R = np.eye(3)  # example rotation matrix (identity for simplicity)
T = np.array([0, 0, 0])  # example translation vector (zero for simplicity)

# Compute the 3D point in the depth camera frame
point_3d = pixel_to_3d_point(u, v, depth_value, K_rgb, R, T)
print("3D point in depth camera frame:", point_3d)


if __name__ == '__main__':
    try:
        depth_to_color_registration = DepthToColorRegistration()
        depth_to_color_registration.run()
    except rospy.ROSInterruptException:
        pass
