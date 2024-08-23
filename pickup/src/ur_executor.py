import rospy
import numpy as np
import actionlib
from pickup.msg import pickupAction, pickupResult, pickupFeedback
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Quaternion
import tf2_geometry_msgs
from tf2_ros import Buffer, TransformListener
from urx import Robot
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
from tf.transformations import quaternion_from_euler, euler_from_quaternion

class GraspExecutor:
    def __init__(self):
        self.rob = Robot("192.168.0.6")
        self.robotiqgrip = Robotiq_Two_Finger_Gripper(self.rob)

        rospy.init_node('grasp_executor', log_level=rospy.DEBUG)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        # Create action server
        self.action_server = actionlib.SimpleActionServer('grasp_action', pickupAction, self.execute_grasp_action, auto_start=False)
        self.action_server.start()

    def execute_grasp_action(self, goal):
        grasp_pos = goal.pos  # Assuming 'pos' contains the grasping position
        feedback = pickupFeedback()
        result = pickupResult()

        rospy.loginfo("Executing grasp with name: {} at position: {}".format(goal.name, grasp_pos))

        try:
            # Transform the grasp position from `realsense_wrist_link` to `arm_base_link`
            tf = self.tf_buffer.lookup_transform('arm_base_link', 'realsense_wrist_link', rospy.Time(0), rospy.Duration(0.1))
            pos = tf2_geometry_msgs.do_transform_point(PointStamped(point=Point(*grasp_pos)), tf)
            pos = [pos.point.x, pos.point.y, pos.point.z]

            feedback.feedback = "Moving to pre-grasp position"
            self.action_server.publish_feedback(feedback)

            # Move to pre-grasp position
            pre_grasp_depth = 0.1
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

    def move_linear(self, xyz=(0, 0, 0), rpy=(0, 0, 0), pose=None, frame_id='arm_base_link'): 
        if pose is None:
            target_pose = PoseStamped()
            target_pose.pose.position = Point(*xyz)
            q = quaternion_from_euler(*rpy)
            target_pose.pose.orientation = Quaternion(*q)
        else:
            target_pose = pose

        tf = self.tf_buffer.lookup_transform('arm_base_link', frame_id, rospy.Time(), rospy.Duration(0.1)) #check all the frame id with the original code!
        target_pose_base = tf2_geometry_msgs.do_transform_pose(target_pose, tf)

        position = target_pose_base.pose.position
        orientation = target_pose_base.pose.orientation
        rpy = euler_from_quaternion((orientation.x, orientation.y, orientation.z, orientation.w))

        self.robot.movel((position.x, position.y, position.z, *rpy), acc=0.1, vel=0.2)
        return True

    def move_joint(self, q):
        self.robot.movej(q, acc=0.1, vel=0.2)
        return True

    def move_gripper(self, q):
        # control gripper in realworld
        if q>0: self.robotiqgrip.close_gripper()
        else: self.robotiqgrip.open_gripper()
        # wait
        rospy.sleep(0.5)
        return 

if __name__ == "__main__":
    executor = GraspExecutor()
    rospy.spin()
