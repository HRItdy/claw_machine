#!/usr/bin/env python3
import rospy
from pickup.srv import Centroid, CentroidRequest

def call_depth_service():
    rospy.init_node('depth_client')
    rospy.wait_for_service('get_depth')
    get_depth = rospy.ServiceProxy('get_depth', Centroid)
    request = CentroidRequest()
    response = get_depth(request)
    print(f"Received coordinates: {response.array}")
    return response.array
   
if __name__ == '__main__':
    centroid = call_depth_service()
    print('done')
