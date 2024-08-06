#!/usr/bin/env python3
import rospy
import roslib
roslib.load_manifest('pickup')
from pickup.srv import GroundingDINO, GroundingDINORequest

def call_service(instruct):
    rospy.init_node('grounding_dino_client')
    rospy.wait_for_service('grounding_dino')
    grounding_dino = rospy.ServiceProxy('grounding_dino', GroundingDINO)
    request = GroundingDINORequest(instruction=instruct)
    response = grounding_dino(request)
    print(f"Received coordinates: cX={response.cX}, cY={response.cY}")
    return response.cX, response.cY
   
if __name__ == '__main__':
    x, y = call_service('pick up the left red ball')
    print('done')
