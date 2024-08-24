import rospy
import numpy as np
import actionlib
import math3d as m3d
from pickup.msg import pickupAction, pickupResult, pickupFeedback
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Quaternion
from tf2_ros import Buffer, TransformListener
from urx import Robot
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
from tf.transformations import quaternion_from_euler, euler_from_quaternion, quaternion_matrix, translation_matrix, concatenate_matrices, translation_from_matrix, quaternion_from_matrix
from claw_depth import inverse_transform_for_point

ACCE = 0.1
VELO = 0.2

class GraspExecutor:
    def __init__(self):
        self.rob = Robot("192.168.0.6")
        self.robotiqgrip = Robotiq_Two_Finger_Gripper(self.rob)

        rospy.init_node('grasp_executor', log_level=rospy.DEBUG)
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
        # self.pub_point(pub)
        # Create action server
        self.action_server = actionlib.SimpleActionServer('grasp_action', pickupAction, self.execute_grasp_action, auto_start=False)
        self.action_server.start()

    def pub_point(self, pub):
        position = rospy.get_param('/3d_position')
        # map back to the pointcloud
        # position = inverse_transform_for_point(position)
        original_frame = 'realsense_wrist_link'

        # Set up the TF2 listener
        tf_buffer = Buffer()
        listener = TransformListener(tf_buffer)

        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            # Lookup transform from 'some_frame' to 'arm_base_link'
            transform = tf_buffer.lookup_transform('arm_base_link', original_frame, rospy.Time(0), rospy.Duration(1.0))
            # Transform the point to 'arm_base_link'
            transform_matrix = self.get_transform_matrix(transform)
            pos = self.transform_point(position, transform_matrix)  
            # pos = np.dot(self.R_z, pos)
            # Create the PointStamped message
            rospy.loginfo(f"The grasping position is: {pos}")
            point_stamped = PointStamped()
            point_stamped.header.stamp = rospy.Time.now()
            point_stamped.header.frame_id = original_frame
            point_stamped.point.x = pos[0]
            point_stamped.point.y = pos[1]
            point_stamped.point.z = pos[2]
            # Publish the transformed point
            pub.publish(point_stamped)           
            rate.sleep()

    def execute_grasp_action(self, goal):
        grasp_pos = goal.pos  # Assuming 'pos' contains the grasping position
        feedback = pickupFeedback()
        result = pickupResult()

        rospy.loginfo("Executing grasp with name: {} at position under frame realsense_wrist_link: {}".format(goal.name, grasp_pos))

        try:
            # Transform the grasp position from `realsense_wrist_link` to `arm_base_link`
            tf = self.tf_buffer.lookup_transform('arm_base_link', 'realsense_wrist_link', rospy.Time(0), rospy.Duration(1.0))
            transform_matrix = self.get_transform_matrix(tf)
            pos = self.transform_point(grasp_pos, transform_matrix)
            rospy.loginfo(f"Grasping position under arm_base_link: {pos}")

            feedback.feedback = "Moving to pre-grasp position"
            self.action_server.publish_feedback(feedback)

            # Move to pre-grasp position
            pre_grasp_depth = 0.2
            pre_grasp_pos = [pos[0], pos[1], pos[2] + pre_grasp_depth]
            if not self.move_linear(pre_grasp_pos, frame_id='arm_base_link'):
                result.result = "Failed to move to pre-grasp position"
                self.action_server.set_aborted(result, result.result)
                return

            # Open gripper
            self.move_gripper(0, attach=False)

            feedback.feedback = "Moving to grasp position"
            self.action_server.publish_feedback(feedback)

            # Move to grasp position
            if not self.move_linear(pos, frame_id='arm_base_link'):
                result.result = "Failed to move to grasp position"
                self.action_server.set_aborted(result, result.result)
                return

            # Close gripper
            self.move_gripper(0.1, attach=True)

            feedback.feedback = "Lifting object"
            self.action_server.publish_feedback(feedback)

            # Lift gripper
            lift_up = 0.05
            lift_pos = [pos[0], pos[1], pos[2] + lift_up]
            if not self.move_linear(lift_pos, frame_id='arm_base_link'):
                result.result = "Failed to lift gripper"
                self.action_server.set_aborted(result, result.result)
                return

            rospy.loginfo("Grasp executed successfully")
            result.result = "Grasp executed successfully"
            self.action_server.set_succeeded(result)

        except Exception as e:
            rospy.logerr(f"Error during grasp execution: {e}")
            result.result = f"Exception: {e}"
            self.action_server.set_aborted(result, result.result)

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
    
    def move_tcp_absolute(self, pose, wait=True):
        """
        move eff to absolute pose in robot base frame with position control
        :param pose: list [x y z R P Y] (meter, radian)
        :param wait: blocking wait
        :return: None
        """
        self.rob.set_pose(m3d.Transform(pose), ACCE, VELO, wait)

    def move_tcp_relative(self, pose, wait=True):
        """
        move eff to relative pose in tool frame with position control
        :param pose: relative differences in [x y z R P Y] (meter, radian)
        :param wait: blocking wait
        :return: None
        """
        self.rob.add_pose_tool(m3d.Transform(pose), ACCE, VELO, wait)

    def move_tcp_perpendicular(self, wait=True):
        """
        move eff perpendicular to xy-plane
        :param wait: blocking wait
        :return: None
        """
        tcp_pose = self.rob.getl()
        transformed_pose = [tcp_pose[0], tcp_pose[1], tcp_pose[2], 0, 0, tcp_pose[5] + 3.1415926]
        self.move_tcp_absolute(transformed_pose, wait)

    def transform_pose(self, pose, transform_matrix):
        trans_point = self.transform_point([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z], transform_matrix)
        trans_quat = quaternion_from_matrix(transform_matrix)
        transformed_pose = PoseStamped()
        transformed_pose.pose.position = Point(*trans_point)
        transformed_pose.pose.orientation = Quaternion(*trans_quat)
        return transformed_pose

    def move_joint(self, q):
        self.rob.movej(q, acc=0.1, vel=0.2)
        return True

    def move_gripper(self, q):
        # control gripper in realworld
        if q > 0:
            self.robotiqgrip.close_gripper()
        else:
            self.robotiqgrip.open_gripper()
        # wait
        rospy.sleep(0.5)
        return

if __name__ == "__main__":
    executor = GraspExecutor()
    rospy.spin()
