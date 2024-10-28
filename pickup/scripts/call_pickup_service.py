#!/usr/bin/env python

import rospy
from pickup.srv import GraspService, GraspServiceRequest, AbortService, AbortServiceRequest

def call_pickup_service(task_name):
    rospy.wait_for_service('grasp_service')
    tasks = ['grasp', 'confirm', 'pass']
    assert task_name in tasks, 'Required action is not in the action list'
    timeout = 5.0
    id_marker = 1
    xyzh = [0.0, 0.0, 0.0, 0.0]
    if task_name in ['grasp', 'confirm']:
        if rospy.has_param('/3d_position'):
            pos = rospy.get_param('/3d_position')
        else:
            rospy.logerr('3d grasping position has not been set.')
    elif task_name in ['pass']:
        if rospy.has_param('/pass_position'):
            pos = rospy.get_param('/pass_position')
        else:
            rospy.logerr('Pass position has not been set.')
    rot = [0.0, 0.0, 0.0, 1.0]  # Example quaternion for no rotation
    jpose = "home"
    try:
        grasp_service = rospy.ServiceProxy('grasp_service', GraspService)
        req = GraspServiceRequest(
            name=task_name,
            timeout=timeout,
            id_marker=id_marker,
            xyzh=xyzh,
            pos=pos,
            rot=rot,
            jpose=jpose
        )
        response = grasp_service(req)
        if response.success:
            rospy.loginfo("Action completed successfully.")
        else:
            rospy.logwarn("Action is still in progress or failed.")
        return response.success
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s" % e)
        return False
    
def call_abort_service():
    rospy.wait_for_service('abort_service')
    try:
        abort_service = rospy.ServiceProxy('abort_service', AbortService)
        request = AbortServiceRequest() 
        response = abort_service(request)
        if response.success:
            rospy.loginfo("Abort service called successfully: " + response.message)
        else:
            rospy.logwarn("Failed to abort action: " + response.message)
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s" % e)

if __name__ == "__main__":
    #success = call_pickup_service('pass')
    success = call_abort_service()
