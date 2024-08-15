import rospy
from sensor_msgs.msg import Image as msg_Image
from sensor_msgs.msg import CameraInfo, PointCloud2
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
import sys
import os
import numpy as np
import pyrealsense2 as rs2
from store_mask_service import store_mask_client
from sensor_msgs import point_cloud2

if (not hasattr(rs2, 'intrinsics')):
    import pyrealsense2.pyrealsense2 as rs2

class ImageListener:
    def __init__(self, depth_image_topic, depth_info_topic):
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber(depth_image_topic, msg_Image, self.imageDepthCallback)
        self.sub_info = rospy.Subscriber(depth_info_topic, CameraInfo, self.imageDepthInfoCallback)
        confidence_topic = depth_image_topic.replace('depth', 'confidence')
        self.sub_conf = rospy.Subscriber(confidence_topic, msg_Image, self.confidenceCallback)
        self.pub = rospy.Publisher('output_pointcloud', PointCloud2, queue_size=10)
        self.intrinsics = None
        self.pix = None
        self.pix_grade = None
        self.rate = rospy.Rate(10)  # 10 Hz

    def imageDepthCallback(self, data):
        depth_image = np.frombuffer(data.data, dtype=np.uint16).reshape(data.height, data.width)
        # cv_image = self.bridge.imgmsg_to_cv2(data, str(data.encoding))
        # pick one pixel among all the pixels with the closest range:
        # indices = np.array(np.where(cv_image == cv_image[cv_image > 0].min()))[:,0]
        mask, success = store_mask_client(store=False)
        indices = np.argwhere(mask)[2:, :].transpose(0, 1) 
        points = []
        for indice in indices:
            pix = (indice[1], indice[0])
            self.pix = pix
            if self.intrinsics:
                depth = depth_image[pix[1], pix[0]] * 0.001
                result = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [pix[0], pix[1]], depth)
                points.append(result)
        pc = np.array(points)
        self.pub_pc(pc, self.pub)
        
    def confidenceCallback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, str(data.encoding))
            grades = np.bitwise_and(cv_image >> 4, 0x0f)
            if (self.pix):
                self.pix_grade = grades[self.pix[1], self.pix[0]]
        except CvBridgeError as e:
            print(e)
            return

    def imageDepthInfoCallback(self, cameraInfo):
        try:
            if self.intrinsics:
                return
            self.intrinsics = rs2.intrinsics()
            self.intrinsics.width = cameraInfo.width
            self.intrinsics.height = cameraInfo.height
            self.intrinsics.ppx = cameraInfo.K[2]
            self.intrinsics.ppy = cameraInfo.K[5]
            self.intrinsics.fx = cameraInfo.K[0]
            self.intrinsics.fy = cameraInfo.K[4]
            if cameraInfo.distortion_model == 'plumb_bob':
                self.intrinsics.model = rs2.distortion.brown_conrady
            elif cameraInfo.distortion_model == 'equidistant':
                self.intrinsics.model = rs2.distortion.kannala_brandt4
            self.intrinsics.coeffs = [i for i in cameraInfo.D]
        except CvBridgeError as e:
            print(e)
            return
            
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

def main():
    depth_image_topic = '/realsense_wrist/aligned_depth_to_color/image_raw'
    depth_info_topic = '/realsense_wrist/depth/camera_info'

    print ('')
    print ('show_center_depth.py')
    print ('--------------------')
    print ('App to demontrate the usage of the /camera/depth topics.')
    print ('')
    print ('Application subscribes to %s and %s topics.' % (depth_image_topic, depth_info_topic))
    print ('Application then calculates and print the range to the closest object.')
    print ('If intrinsics data is available, it also prints the 3D location of the object')
    print ('If a confedence map is also available in the topic %s, it also prints the confidence grade.' % depth_image_topic.replace('depth', 'confidence'))
    print ('')
    
    listener = ImageListener(depth_image_topic, depth_info_topic)
    rospy.spin()

if __name__ == '__main__':
    node_name = os.path.basename(sys.argv[0]).split('.')[0]
    rospy.init_node(node_name)
    main()