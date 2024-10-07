#!/usr/bin/env python

import rospy
from pickup.srv import Get3DPosition, Get3DPositionRequest, Get3DPositionResponse  # Replace 'your_package' with your actual package name
from geometry_msgs.msg import Point, PointStamped

def call_depth_service():
    rospy.init_node('get_3d_position_client')
    rospy.wait_for_service('get_3d_position')
    try:
        get_3d_position = rospy.ServiceProxy('get_3d_position', Get3DPosition)
        response = get_3d_position() 
        position = [response.position.x, response.position.y, response.position.z]
        if position == [-1, -1, -1]:
            rospy.logerr('No mask is assigned, please call the detection function first.')
            return None
        else:
            # position = [response.position.z, -response.position.x, response.position.y]
            rospy.loginfo(f'Get the estimated centroid location: {position} (under realsense_wrist_link frame)')
            rospy.set_param('/3d_position', position)
            return position
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        return None

if __name__ == "__main__":
    position = call_depth_service()
