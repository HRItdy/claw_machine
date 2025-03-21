#!/usr/bin/env python

import rospy
import roslib
roslib.load_manifest('pickup')
from pickup.srv import GraspService, GraspServiceRequest

def call_pickup_service():
    rospy.wait_for_service('grasp_service')
    name = "example_grasp"
    timeout = 5.0
    id_marker = 1
    xyzh = [0.0, 0.0, 0.0, 0.0]
    pos = rospy.get_param('/3d_position')
    rot = [0.0, 0.0, 0.0, 1.0]  # Example quaternion for no rotation
    jpose = "home"
    try:
        grasp_service = rospy.ServiceProxy('grasp_service', GraspService)
        req = GraspServiceRequest(
            name=name,
            timeout=timeout,
            id_marker=id_marker,
            xyzh=xyzh,
            pos=pos,
            rot=rot,
            jpose=jpose
        )
        response = grasp_service(req)
        if response.success:
            rospy.loginfo("Grasp action completed successfully.")
        else:
            rospy.logwarn("Grasp action is still in progress or failed.")
        return response.success
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s" % e)
        return False

if __name__ == "__main__":
    success = call_pickup_service()
