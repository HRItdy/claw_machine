#!/usr/bin/env python

import rospy
from pickup.srv import Get3DPosition, Get3DPositionRequest, Get3DPositionResponse  # Replace 'your_package' with your actual package name
from geometry_msgs.msg import Point, PointStamped

def call_depth_service():
    if not rospy.has_param('/calibration/H'):
        rospy.ERROR('Calibration is required!')
    rospy.init_node('get_3d_position_client')
    rospy.wait_for_service('get_3d_position')
    try:
        get_3d_position = rospy.ServiceProxy('get_3d_position', Get3DPosition)
        response = get_3d_position() 
        # For the pointcloud frame, if u want to map a point onto the pointcloud
        # pointcloud.x = -point.y
        # pointcloud.y = -point.z
        # pointcloud.z = point.x

        # If want to map the pointcloud to point frame
        # point.x = pointcloud.z
        # point.y = -pointcloud.x
        # point.z = -pointcloud.y
        position = [response.position.x, response.position.y, response.position.z]
        # position = [response.position.z, -response.position.x, response.position.y]
        rospy.loginfo(f'Get the estimated centroid location: {position}')
        rospy.set_param('/3d_position', position)
        pub = rospy.Publisher('/extract_3d_position', PointStamped, queue_size=10)
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            # pos = np.dot(self.R_z, pos)
            # Create the PointStamped message
            point_stamped = PointStamped()
            point_stamped.header.stamp = rospy.Time.now()
            point_stamped.header.frame_id = 'realsense_wrist_link'
            point_stamped.point.x = position[0]
            point_stamped.point.y = position[1]
            point_stamped.point.z = position[2]
            # Publish the transformed point
            pub.publish(point_stamped)           
            rate.sleep()
        return position
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        return None

if __name__ == "__main__":
    position = call_depth_service()
    print('centroid position found')
