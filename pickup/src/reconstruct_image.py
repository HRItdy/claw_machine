#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import cv2
import struct
import ctypes
# def float_to_rgb(float_rgb):
#     # Convert the float to bytes
#     packed = struct.pack('f', float_rgb)
#     # Unpack bytes to integers
#     i = struct.unpack('I', packed)[0]
#     # Extract RGB values
#     r = (i & 0x00FF0000) >> 16
#     g = (i & 0x0000FF00) >> 8
#     b = (i & 0x000000FF)
#     return r, g, b

# def float_to_rgb(float_rgb):
#     # Convert the float to bytes (assuming little-endian byte order)
#     packed = struct.pack('<f', float_rgb)
#     # Unpack the bytes to an unsigned integer
#     i = struct.unpack('<I', packed)[0]
    
#     # Extract RGB components
#     r = (i >> 16) & 0xFF    # Red: bits 16-23
#     g = (i >> 8) & 0xFF     # Green: bits 8-15
#     b = i & 0xFF            # Blue: bits 0-7
    
#     return r, g, b

def pc2_to_xyzrgb(point):
	# Thanks to Panos for his code used in this function.
    x, y, z = point[:3]
    rgb = point[3]

    # cast float32 to int so that bitwise operations are possible
    s = struct.pack('>f', rgb)
    i = struct.unpack('>l', s)[0]
    # you can get back the float value by the inverse operations
    pack = ctypes.c_uint32(i).value
    r = (pack & 0x00FF0000) >> 16
    g = (pack & 0x0000FF00) >> 8
    return x, y, z, r, g, b

def pointcloud2_to_image(cloud_msg):
    # Get cloud data as a list of points
    points_list = list(pc2.read_points(cloud_msg, skip_nans=True, field_names=("x", "y", "z", "rgb")))
    
    # Get the dimensions from the PointCloud2 message
    height = 1200
    width = 1200
    
    # Create a blank image
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Find the minimum and maximum x and y values
    # x_values = [point[0] for point in points_list]
    # y_values = [point[1] for point in points_list]
    x_min = -0.5
    x_max = 0.5
    
    y_min = -0.5
    y_max = 0.5
    
    # rospy.loginfo(f"X range: {x_min} to {x_max}")
    # rospy.loginfo(f"Y range: {y_min} to {y_max}")
    
    for point in points_list:
        # Normalize x and y to fit within the image dimensions
        x = int((point[0] - x_min) / (x_max - x_min) * (width - 1))
        y = int((point[1] - y_min) / (y_max - y_min) * (height - 1))

        if x > width-1 or y > height-1:
            continue

        rgb_float = point[3]
        
        # Extract RGB using our custom function
        _, _, _, r, g, b = pc2_to_xyzrgb(point)

        
        # Assign the color to the image
        rgb_image[y, x] = [b, g, r]
        
        # Debug output for a few points
        #rospy.loginfo(f"Sample point: original x={point[0]}, y={point[1]}, mapped x={x}, y={y}, rgb={rgb_float}, r={r}, g={g}, b={b}")
    
    # Additional debug: print some non-zero pixel values
    non_zero_pixels = np.argwhere(rgb_image != 0)
    for i in range(min(5, len(non_zero_pixels))):
        y, x, c = non_zero_pixels[i]
        rospy.loginfo(f"Non-zero pixel: x={x}, y={y}, value={rgb_image[y, x]}")
    
    return rgb_image

def callback(data):
    image = pointcloud2_to_image(data)
    if image is not None:
        cv2.imshow("Reconstructed Image", image)
        cv2.waitKey(1)
    else:
        rospy.logerr("Failed to reconstruct image")

def listener():
    rospy.init_node('pointcloud2_to_image', anonymous=True)
    rospy.Subscriber("/realsense_wrist/depth_registered/points", PointCloud2, callback)
    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass