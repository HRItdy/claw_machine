import rospy
import numpy as np
import actionlib
import math3d as m3d
from pickup.msg import pickupAction, pickupResult, pickupFeedback
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Quaternion
from tf2_ros import Buffer, TransformListener
from urx import Robot, ursecmon
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
from tf.transformations import quaternion_from_euler, euler_from_quaternion, quaternion_matrix, translation_matrix, concatenate_matrices, translation_from_matrix, quaternion_from_matrix

ACCE = 0.1
VELO = 0.8
LIFTUP = 0.01
RQY = (0, 1.57, 0)  # Change to your own grasping orientation
HOMEJ = [0, -1.6376, 1.3327, -1.25, -1.5666, 0]
PASSJ = [0.7086, -0.894, 1.1006, -2.288, -1.602, -0.1693]

class CustomRobot(Robot):
    def __init__(self, *args, **kwargs):
        super(CustomRobot, self).__init__(*args, **kwargs)
        # Override the timeout in SecondaryMonitor
        self.secmon.wait(timeout=2)  # Increase timeout to 2 seconds

class GraspExecutor:
    def __init__(self):
        self.rob = CustomRobot("192.168.0.5")
        self.robotiqgrip = Robotiq_Two_Finger_Gripper(self.rob)

        rospy.init_node('grasp_executor', log_level=rospy.DEBUG)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        TRANS_TCP = (0, 0, 0.19, 1.2092, -1.2092, 1.2092)
        # Align the robot with the standard axis
        TRANS_BASE = [0, 0, 0, 0, 0, 3.1415926]
        self.rob.set_tcp(TRANS_TCP)  # Replace with TCP
        self.rob.set_payload(0.5, (0, 0, 0.1))  # Replace with payload
        self.rob.set_csys(m3d.Transform(TRANS_BASE))
        pub = rospy.Publisher('/robot_base_position', PointStamped, queue_size=10)
        # self.pub_point(pub)
        # Create action server
        self.action_server = actionlib.SimpleActionServer('grasp_action', pickupAction, self.execute_grasp_action, auto_start=False)
        self.action_server.start()

    def pub_point(self, pub):
        position = rospy.get_param('/3d_position')
        original_frame = 'realsense_wrist_link'

        # Set up the TF2 listener
        tf_buffer = Buffer()
        listener = TransformListener(tf_buffer)

        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            # Lookup transform from 'some_frame' to 'base_link'
            transform = tf_buffer.lookup_transform('base_link', original_frame, rospy.Time(0), rospy.Duration(1.0))
            # Transform the point to 'base_link'
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
        tasks = ['grasp', 'confirm', 'pass']
        assert goal.name in tasks, 'Required action is not in the action list'
        rospy.loginfo("Executing action with name: {} at position under frame realsense_wrist_link: {}".format(goal.name, grasp_pos))

        if goal.name == 'grasp':
            try:
                # Transform the grasp position from `realsense_wrist_link` to `base_link`
                tf = self.tf_buffer.lookup_transform('base_link', 'base_link', rospy.Time(0), rospy.Duration(1.0))
                transform_matrix = self.get_transform_matrix(tf)
                pos = self.transform_point(grasp_pos, transform_matrix)
                rospy.loginfo(f"Grasping position under base_link: {pos}")
                # Move to the grasp position
                feedback.feedback = "Moving to grasp position"
                self.action_server.publish_feedback(feedback)
                # Open gripper
                self.move_gripper(0)
                # Move to grasp position
                self.rob.movel((pos[0], pos[1], pos[2], *RQY), acc=0.1, vel=VELO, relative=False)
                # Close gripper
                self.move_gripper(0.1)
                # Liftup gripper
                feedback.feedback = "Lifting object"
                self.action_server.publish_feedback(feedback)
                self.rob.movel((pos[0], pos[1], pos[2] + LIFTUP + 0.1, *RQY), acc=0.1, vel=VELO, relative=False)
                rospy.loginfo("Grasp executed successfully")
                result.result = "Grasp executed successfully"
                self.action_server.set_succeeded(result)
            except Exception as e:
                rospy.logerr(f"Error during grasp execution: {e}")
                result.result = f"Exception: {e}"
                self.action_server.set_aborted(result, result.result)

        elif goal.name == 'pass':
            try:
                # Transform the pass position from `base_link` to `base_link`
                tf = self.tf_buffer.lookup_transform('base_link', 'base_link', rospy.Time(0), rospy.Duration(1.0))
                transform_matrix = self.get_transform_matrix(tf)
                pos = self.transform_point(grasp_pos, transform_matrix)
                rospy.loginfo(f"Pass position under base_link: {pos}")
                # Move to the pass position
                feedback.feedback = "Moving to pass position"
                self.action_server.publish_feedback(feedback)
                #self.rob.movel((pos[0], pos[1], pos[2] + LIFTUP + 0.005, *RQY), acc=0.1, vel=VELO, relative=False)
                self.rob.movej(PASSJ, 0.5, 0.5, wait=True)
                # Open gripper
                self.move_gripper(0)
                # Move back to home
                feedback.feedback = "Passed. Move back to home position"
                self.action_server.publish_feedback(feedback)
                self.rob.movej(HOMEJ, 0.5, 0.5, wait=True)
                rospy.loginfo("Pass executed successfully")
                result.result = "Pass executed successfully"
                self.action_server.set_succeeded(result)
            except Exception as e:
                rospy.logerr(f"Error during pass execution: {e}")
                result.result = f"Exception: {e}"
                self.action_server.set_aborted(result, result.result)

        elif goal.name == 'confirm':
            try:
                # Transform the confirm position from `base_link` to `base_link`
                tf = self.tf_buffer.lookup_transform('base_link', 'realsense_wrist_link', rospy.Time(0), rospy.Duration(1.0))
                transform_matrix = self.get_transform_matrix(tf)
                pos = self.transform_point(grasp_pos, transform_matrix)
                rospy.loginfo(f"Confrim position under base_link: {pos}")
                pos_list = pos.tolist()
                rospy.set_param('/3d_position', pos_list) # Set the pos param  #TODO: change the pos to base_link the first time.
                # Move to the confirm position
                feedback.feedback = "Moving to confirm position"
                self.action_server.publish_feedback(feedback)
                # Close gripper
                self.move_gripper_confirm(0.1)
                # Move to confirm position
                self.rob.movel((pos[0], pos[1], pos[2] + LIFTUP + 0.04, *RQY), acc=0.1, vel=VELO, relative=False)
                feedback.feedback = "Confirm object"
                self.action_server.publish_feedback(feedback)
                rospy.loginfo("Confirm executed successfully")
                result.result = "Confirm executed successfully"
                self.action_server.set_succeeded(result)
            except Exception as e:
                rospy.logerr(f"Error during confirm execution: {e}")
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
    
    def move_tcp_absolute(self, pose, world=False, wait=False):
        """
        move eff to absolute pose in robot base frame with position control
        :param pose: list [x y z R P Y] (meter, radian)
        :param world: boolean True for world frame, else tcp frame
        :param wait: blocking wait
        :return: None
        """
        ACCE = 0.1
        VELO = 0.2
        TOL_TARGET_POSE = 0.0005
        if len(pose) == 3:
            pose = pose + self.get_robot_position()[-3:]
        if (self.is_program_running() and
            self.dist_linear(pose, self.target_pose) > TOL_TARGET_POSE) or \
                (not self.is_program_running() and
                 self.dist_linear(pose, self.get_robot_position()) > TOL_TARGET_POSE):
            self.target_pose = pose
            if world:
                pose = self.to_world * m3d.Transform(pose)
            self.robot.set_pose(m3d.Transform(pose), ACCE, VELO, wait)

    def move_linear(self, xyz=(0, 0, 0), rpy=(1.57, 0, 1.57), pose=None, frame_id='base_link'):
        if pose is None:
            target_pose = PoseStamped()
            target_pose.pose.position = Point(*xyz)
            q = quaternion_from_euler(*rpy)
            target_pose.pose.orientation = Quaternion(*q)
        else:
            target_pose = pose

        tf = self.tf_buffer.lookup_transform('base_link', frame_id, rospy.Time(), rospy.Duration(0.1))
        transform_matrix = self.get_transform_matrix(tf)
        target_pose_base = self.transform_pose(target_pose, transform_matrix)

        position = target_pose_base.pose.position
        orientation = target_pose_base.pose.orientation
        rpy = euler_from_quaternion((orientation.x, orientation.y, orientation.z, orientation.w))

        self.rob.movel((position.x, position.y, position.z, *rpy), acc=0.1, vel=VELO)
        return True
    
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
            #self.robotiqgrip.close_gripper()
            self.robotiqgrip.gripper_action(124, force=50)
        else:
            self.robotiqgrip.gripper_action(0, force=50)
            #self.robotiqgrip.open_gripper()
        # wait
        rospy.sleep(0.5)
        return
    
    def move_gripper_confirm(self, q):
        # control gripper in realworld
        if q > 0:
            #self.robotiqgrip.close_gripper()
            self.robotiqgrip.gripper_action(255, force=50)
        else:
            self.robotiqgrip.gripper_action(0, force=50)
            #self.robotiqgrip.open_gripper()
        # wait
        rospy.sleep(0.5)
        return

if __name__ == "__main__":
    executor = GraspExecutor()
    rospy.spin()
