import rospy
import numpy as np
import math3d as m3d
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Quaternion
from tf2_ros import Buffer, TransformListener
from urx import Robot
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
from tf.transformations import quaternion_from_euler, euler_from_quaternion, quaternion_matrix, translation_matrix, concatenate_matrices, translation_from_matrix, quaternion_from_matrix
from claw_depth import inverse_transform_for_point

ACCE = 0.1
VELO = 0.2

class Transform:
    def __init__(self):
        self.rob = Robot("192.168.0.6")
        self.robotiqgrip = Robotiq_Two_Finger_Gripper(self.rob)

        rospy.init_node('transformer', log_level=rospy.DEBUG)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        # Align the robot with the standard axis
        TRANS_TCP = (0, 0, 0.19, 1.2092, -1.2092, 1.2092)
        #TRANS_BASE = [0, 0, 0, -0.61394313, 1.48218982, 0.61394313]
        TRANS_BASE = [0, 0, 0, 0, 0, 3.1415926]
        self.R_z = np.array([             # Transform matrix corresponding to TRANS_TCP
                        [-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, 1]
                    ])
        self.rob.set_tcp(TRANS_TCP)  # Replace with TCP
        self.rob.set_payload(0.5, (0, 0, 0.1))  # Replace with payload
        self.rob.set_csys(m3d.Transform(TRANS_BASE))
        #self.move_tcp_perpendicular()
        
        # confirm the coordinates
        # current_pose = self.rob.get_pose()
        # new_pose = current_pose.copy()
        # new_pose.pos.z -= 0.05
        # self.rob.set_pose(new_pose, wait=True)
        # finish the confirmation
        # rospy.init_node('position_publisher', anonymous=True)
        pub = rospy.Publisher('/robot_base_position', PointStamped, queue_size=10)
        sub = rospy.Subscriber('/clicked_point', PointStamped, self.transformCallBack, pub)
        # self.pub_point(pub)
        self.verify(pub)
        # Create action server

    def transformCallBack(self, data, pub):
        position = [data.point.x, data.point.y, data.point.z]
        print(data.header.frame_id)
        tf = self.tf_buffer.lookup_transform('realsense_wrist_link', 'arm_base_link', rospy.Time(0), rospy.Duration(0.1))
        transform_matrix = self.get_transform_matrix(tf)
        pos = self.transform_point(position, transform_matrix)
        self.pub_point(pub, pos)

    def verify(self, pub):
        pos = rospy.get_param('/3d_position')
        tf = self.tf_buffer.lookup_transform('arm_base_link', 'realsense_wrist_link', rospy.Time(0), rospy.Duration(1.0))
        transform_matrix = self.get_transform_matrix(tf)
        pos = self.transform_point(pos, transform_matrix)
        rospy.loginfo(f"the position of grasping is: {pos}")
        self.pub_point(pub, pos)


    def pub_point(self, pub, pos):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            # pos = np.dot(self.R_z, pos)
            # Create the PointStamped message
            point_stamped = PointStamped()
            point_stamped.header.stamp = rospy.Time.now()
            point_stamped.header.frame_id = 'realsense_wrist_link'
            point_stamped.point.x = pos[0]
            point_stamped.point.y = pos[1]
            point_stamped.point.z = pos[2]
            # Publish the transformed point
            pub.publish(point_stamped)           
            rate.sleep()

    def get_transform_matrix(self, tf):
        trans = translation_matrix([tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z])
        rot = quaternion_matrix([tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w])
        return concatenate_matrices(trans, rot)

    def transform_point(self, point, transform_matrix):
        point_homogeneous = np.array([point[0], point[1], point[2], 1.0])
        transformed_point = np.dot(transform_matrix, point_homogeneous)
        return transformed_point[:3]

    def move_linear(self, xyz=(0, 0, 0), rpy=(0, 0, 0), pose=None, frame_id='arm_base_link'):
        if pose is None:
            target_pose = PoseStamped()
            target_pose.pose.position = Point(*xyz)
            q = quaternion_from_euler(*rpy)
            target_pose.pose.orientation = Quaternion(*q)
        else:
            target_pose = pose

        tf = self.tf_buffer.lookup_transform('arm_base_link', frame_id, rospy.Time(), rospy.Duration(0.1))
        transform_matrix = self.get_transform_matrix(tf)
        target_pose_base = self.transform_pose(target_pose, transform_matrix)

        position = target_pose_base.pose.position
        orientation = target_pose_base.pose.orientation
        rpy = euler_from_quaternion((orientation.x, orientation.y, orientation.z, orientation.w))

        self.rob.movel((position.x, position.y, position.z, *rpy), acc=0.1, vel=0.2)
        return True
    
    def transform_pose(self, pose, transform_matrix):
        trans_point = self.transform_point([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z], transform_matrix)
        trans_quat = quaternion_from_matrix(transform_matrix)
        transformed_pose = PoseStamped()
        transformed_pose.pose.position = Point(*trans_point)
        transformed_pose.pose.orientation = Quaternion(*trans_quat)
        return transformed_pose

if __name__ == "__main__":
    executor = Transform()
    rospy.spin()
