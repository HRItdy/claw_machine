#!/usr/bin/env python
import rospy
import numpy as np
from threading import Thread
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ImageProcessorNode:
    def __init__(self):
        rospy.init_node('image_processor_node', anonymous=True)

        # ROS publishers and subscribers
        self.color_sub = rospy.Subscriber('/realsense_wrist/color/image_raw', Image, self.color_callback)
        self.depth_sub = rospy.Subscriber('/realsense_wrist/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        self.flip_color_pub = rospy.Publisher('/flipped_color_image', Image, queue_size=5)
        self.err_pub = rospy.Publisher('/error_masked_image', Image, queue_size=5)

        # Image variables
        self.color_image = None
        self.depth_image = None

        # Bridge for converting between ROS and OpenCV images
        self.bridge = CvBridge()

        # Start the ROS node
        rospy.loginfo("Image Processor Node initialized.")
        rospy.spin()

    def color_callback(self, color_msg):
        """Callback for color image messages."""
        self.color_image = np.frombuffer(color_msg.data, dtype=np.uint8).reshape(color_msg.height, color_msg.width, -1)

    def depth_callback(self, depth_msg):
        """Callback for depth image messages."""
        self.depth_image = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(depth_msg.height, depth_msg.width)
        if self.color_image is not None:
            # Offload processing to a separate thread to reduce latency
            err_masks = rospy.get_param('/error_balls', [])
            """Process the image and publish the results."""
            # Flip the image
            flipped_image = np.flipud(np.fliplr(self.color_image))
            flip_ros_image = self.bridge.cv2_to_imgmsg(flipped_image, encoding="rgb8")
            self.flip_color_pub.publish(flip_ros_image)
            
            # Apply error masks
            mask_array = np.array(err_masks, dtype=np.uint8).T
            masked_image = self.color_image.copy()
            masked_image[mask_array > 0] = [255, 255, 255]
            
            # Publish the masked image
            err_ros_image = self.bridge.cv2_to_imgmsg(masked_image, encoding="rgb8")
            self.err_pub.publish(err_ros_image)

if __name__ == '__main__':
    try:
        node = ImageProcessorNode()
    except rospy.ROSInterruptException:
        pass
