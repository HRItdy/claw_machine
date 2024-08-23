import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud
import numpy as np
import cv2
import pyrealsense2 as rs
from cv_bridge import CvBridge
from geometry_msgs.msg import Point32

class PointCloudFromPixel:
    def __init__(self):
        self.bridge = CvBridge()
        self.depth_sub = rospy.Subscriber('/realsense_wrist/depth/image_rect_raw', Image, self.depth_callback)
        self.camera_info_sub = rospy.Subscriber('/realsense_wrist/depth/camera_info', CameraInfo, self.camera_info_callback)
        self.point_cloud_pub = rospy.Publisher('/processed_point_cloud', PointCloud, queue_size=10)
        self.intrinsics = None

    def camera_info_callback(self, msg):
        self.intrinsics = rs.intrinsics()
        self.intrinsics.width = msg.width
        self.intrinsics.height = msg.height
        self.intrinsics.ppx = msg.K[2]
        self.intrinsics.ppy = msg.K[5]
        self.intrinsics.fx = msg.K[0]
        self.intrinsics.fy = msg.K[4]
        self.intrinsics.model = rs.distortion.none
        self.intrinsics.coeffs = [i for i in msg.D]

    def depth_callback(self, msg):
        if self.intrinsics is None:
            return
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        point_cloud = self.get_point_cloud(depth_image)
        point_cloud_msg = self.convert_to_pointcloud(point_cloud, msg.header)
        self.point_cloud_pub.publish(point_cloud_msg)

    def get_point_cloud(self, depth_image):
        height, width = depth_image.shape
        point_cloud = []
        for v in range(height):
            for u in range(width):
                if depth_image[v, u] > 0:
                    point = rs.rs2_deproject_pixel_to_point(self.intrinsics, [u, v], float(depth_image[v, u]) * 0.001)
                    point_cloud.append(Point32(point[0], point[1], point[2]))
        return point_cloud

    def convert_to_pointcloud(self, point_cloud, header):
        point_cloud_msg = PointCloud()
        point_cloud_msg.header = header
        point_cloud_msg.points = point_cloud
        return point_cloud_msg

if __name__ == '__main__':
    rospy.init_node('point_cloud_from_pixel')
    pc_from_pixel = PointCloudFromPixel()
    rospy.spin()
