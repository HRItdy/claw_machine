#!/usr/bin/env python3
import rospy
from pickup.srv import GroundingDINO, GroundingDINORequest

def call_detect_service():
    rospy.init_node('grounding_dino_client')
    rospy.wait_for_service('grounding_dino')
    grounding_dino = rospy.ServiceProxy('grounding_dino', GroundingDINO)
    request = GroundingDINORequest(instruction='pick up the left red ball')
    response = grounding_dino(request)
    print(f"Received coordinates: cX={response.cX}, cY={response.cY}")
    return response.cX, response.cY
   
if __name__ == '__main__':
    x, y = call_detect_service()
    print('done')
