#!/usr/bin/env python3
# 1-- Calculate the point cloud points corresponding to the mask. It will first convert the whole color frame into 3D pointcloud,
#     then match this pointcloud with the one projected from '/realsense_wrist/depth/image_rect_raw', get the transformation matrix (This should be done in the __init__).
# 2-- Call the service to get the mask, and calculate the corresponding pointcloud, then publish it or ?

import rospy
import numpy as np
from pickup.srv import Centroid, CentroidResponse
from geometry_msgs.msg import Point32
from std_msgs.msg import Header, Float32MultiArray
from sensor_msgs.msg import PointCloud2, CameraInfo, Image
from sensor_msgs import point_cloud2
from store_mask_service import store_mask_client
from image_geometry import PinholeCameraModel
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge
import pyrealsense2 as rs2


class ClawDepth:
    def __init__(self):
        rospy.init_node('claw_depth', anonymous=True)
        self.depth_image = None
        self.info_sub = Subscriber('/realsense_wrist/depth/camera_info', CameraInfo)
        self.depth_sub = Subscriber('/aligned_depth_image', Image)
        self.depth_pub = rospy.Publisher('/depth_point_cloud', PointCloud2, queue_size=1)
        self.color_pub = rospy.Publisher('/color_point_cloud', PointCloud2, queue_size=1)
        self.seg_pub = rospy.Publisher('/segment_point_cloud', PointCloud2, queue_size=1)
        # Synchronize the topics
        self.ats = ApproximateTimeSynchronizer([self.depth_sub, self.info_sub], queue_size=5, slop=0.1)
        self.ats.registerCallback(self.callback)
        # Service server  TODO
        self.service = rospy.Service('get_depth', Centroid, self.handle_service)
        self.rate = rospy.Rate(10)  # 10 Hz
        rospy.spin()

    def callback(self, depth_image, camera_info):
        self.depth_image = np.frombuffer(depth_image.data, dtype=np.uint16).reshape(depth_image.height, depth_image.width)
        self.intrinsics = self.camera_register(camera_info)
        # get the mask
        mask, success = store_mask_client(store=False)
        indices = np.argwhere(mask)[2:, :].transpose(0, 1) # TODO: Check the coordinates after the transpose!
        self.seg_pc = self.points_to_point_cloud(indices, self.depth_image, self.intrinsics)
        self.pub_pc(self.seg_pc, self.seg_pub)
        print('done')

    # The service to get the depth
    def handle_service(self, req):
        # get the mask from the service
        mask, success = store_mask_client(store=False)
        indices = np.argwhere(mask)[2:, :].transpose(0, 1) # TODO: Check the coordinates after the transpose!
        
        # Convert depth image message to numpy array
        bridge = CvBridge()
        depth_image = bridge.imgmsg_to_cv2(self.depth_image, desired_encoding='passthrough')

        # Get 3D coordinates from the depth image
        points = []
        for indice in indices:
            x, y, z = self.pixel_to_3d(indice, depth_image, self.align_intrinsics)
            #point_3d = self.get_point_from_pointcloud(self.point_cloud, x, y, z)
            points.append([x, y, z])

        # Create a PointCloud2 message from the points
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.point_cloud.header.frame_id
        point_cloud_msg = point_cloud2.create_cloud_xyz32(header, points)
        # Publish the new point cloud
        self.seg_pc.publish(point_cloud_msg)
        rospy.loginfo("Published masked point cloud with {} points".format(len(points)))
        # Prepare the response
        response_array = Float32MultiArray()
        points_flat = [coord for point in points for coord in point]
        response_array.data = points_flat
        
        return CentroidResponse(array=response_array)
    
    def pixel_to_3d(self, pixel, depth_image, intrinsics):
        """
        Convert 2D pixel coordinates to 3D point coordinates in the camera frame.
        
        :param pixel: (u, v) tuple, 2D pixel coordinates
        :param depth_image: Depth image (numpy array)
        :param camera_info: CameraInfo message with camera intrinsic parameters
        :return: (x, y, z) tuple, 3D coordinates in the camera frame
        """
        u, v = pixel
        result = rs2.rs2_deproject_pixel_to_point(intrinsics, [u, v], float(depth_image[v, u]) * 0.001)  #result[0]: right, result[1]: down, result[2]: forward
        #return result[2], -result[0], -result[1]
        return result[0], result[1], result[2]
       
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
    
    def points_to_point_cloud(self, indices, depth_image, intrinsics):
        point_cloud = []
        for indice in indices:
            x, y, z = self.pixel_to_3d(indice, depth_image, intrinsics)
            point_cloud.append([x, y, z])
        return np.array(point_cloud)
    
    def cluster_pub(self, pc, pub):
        # Create a PointCloud2 message from the points
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'realsense_wrist_link'
        point_cloud_msg = point_cloud2.create_cloud_xyz32(header, pc)
        pub.publish(point_cloud_msg)

    def pub_pc(self, pc, pub):
        while not rospy.is_shutdown():
            self.cluster_pub(pc, pub)
            self.rate.sleep()

if __name__ == '__main__':
    ClawDepth()
