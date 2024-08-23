#!/usr/bin/env python

import rospy
from pickup.srv import PixelTo3D

def project_pixel_to_3d_client(u, v):
    rospy.wait_for_service('pixel_to_3d')
    try:
        project_pixel_to_3d = rospy.ServiceProxy('pixel_to_3d', PixelTo3D)
        resp = project_pixel_to_3d(u, v)
        return resp.x, resp.y, resp.z
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s" % e)

if __name__ == "__main__":
    rospy.init_node('project_pixel_to_3d_client')
    u, v = 453, 125  # Example pixel coordinates
    x, y, z = project_pixel_to_3d_client(u, v)
    rospy.loginfo("3D Coordinates: x=%f, y=%f, z=%f" % (x, y, z))
