#!/usr/bin/env python

import rospy
import roslib
roslib.load_manifest('pickup')
from pickup.srv import Get3DPosition, Get3DPositionRequest, Get3DPositionResponse  # Replace 'your_package' with your actual package name
from geometry_msgs.msg import Point

def call_depth_service():
    if not rospy.has_param('/calibration/H'):
        rospy.ERROR('Calibration is required!')
    #rospy.init_node('get_3d_position_client', anonymous=True)
    rospy.wait_for_service('get_3d_position')
    try:
        get_3d_position = rospy.ServiceProxy('get_3d_position', Get3DPosition)
        response = get_3d_position() 
        position = [response.position.x, response.position.y, response.position.z]
        rospy.loginfo(f'Get the estimated centroid location: {position}')
        rospy.set_param('/3d_position', position)
        return position
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        return None

if __name__ == "__main__":
    position = call_depth_service()
    print('centroid position found')
