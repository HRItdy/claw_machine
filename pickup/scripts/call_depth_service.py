#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32MultiArray
from pickup.srv import Centroid, CentroidRequest

def call_depth_service():
    # Initialize the ROS node
    rospy.init_node('depth_client')
    
    # Wait for the service to be available
    rospy.wait_for_service('get_depth')
    
    try:
        # Create a service proxy to call the service
        get_depth = rospy.ServiceProxy('get_depth', Centroid)
        
        # Prepare the service request (if needed, here it's an empty request)
        request = CentroidRequest()
        
        # Call the service and get the response
        response = get_depth(request)
        
        # Extract the array from the response
        point_array = response.array.data
        
        # Print the received coordinates
        print(f"Received coordinates: {point_array}")
        
        return point_array

    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        return None

if __name__ == '__main__':
    # Call the depth service and print the results
    points = call_depth_service()
    print('done')
