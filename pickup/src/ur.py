import rospy
import numpy as np
import tf2_geometry_msgs
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import Point, PointStamped, Quaternion, PoseStamped, Pose, TransformStamped, Transform, Vector3
from sensor_msgs.msg import PointCloud2
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

    def execute_grasp(self, grasp_pos):
        rospy.loginfo("Executing grasp at position: {}".format(grasp_pos))

        # Transform the grasp position from `realsense_wrist_link` to `arm_base_link`
        tf = self.tf_buffer.lookup_transform('arm_base_link', 'realsense_wrist_link', rospy.Time(0), rospy.Duration(0.1))
        pos = tf2_geometry_msgs.do_transform_point(PointStamped(point=Point(*grasp_pos)), tf)
        pos = [pos.point.x, pos.point.y, pos.point.z]

        # Move to pre-grasp position
        pre_grasp_depth = 0.1  # Adjust this value as needed
        pre_grasp_pos = [pos[0], pos[1], pos[2] + pre_grasp_depth]
        if not self.move_linear(pre_grasp_pos, frame_id='arm_base_link'):
            rospy.logerr("Failed to move to pre-grasp position")
            return

        # Open gripper
        self.move_gripper(0, attach=False)

        # Move to grasp position
        if not self.move_linear(pos, frame_id='arm_base_link'):
            rospy.logerr("Failed to move to grasp position")
            return

        # Close gripper
        self.move_gripper(0.1, attach=True)

        # Lift gripper
        lift_up = 0.05  # Adjust this value as needed
        lift_pos = [pos[0], pos[1], pos[2] + lift_up]
        if not self.move_linear(lift_pos, frame_id='arm_base_link'):
            rospy.logerr("Failed to lift gripper")
            return

        rospy.loginfo("Grasp executed successfully")

    def move_linear(self, xyz=(0, 0, 0), rpy=(0, 0, 0), frame_id='arm_base_link'):
        target_pose = PoseStamped()
        target_pose.pose.position = Point(*xyz)
        q = quaternion_from_euler(*rpy)
        target_pose.pose.orientation = Quaternion(*q)

        tf = self.tf_buffer.lookup_transform('arm_base_link', frame_id, rospy.Time(), rospy.Duration(0.1))
        target_pose_base = tf2_geometry_msgs.do_transform_pose(target_pose, tf)

        position = target_pose_base.pose.position
        orientation = target_pose_base.pose.orientation
        rpy = euler_from_quaternion((orientation.x, orientation.y, orientation.z, orientation.w))

        self.rob.movel((position.x, position.y, position.z, *rpy), acc=0.1, vel=0.2)
        return True

    def move_gripper(self, q, attach=False):
        if q > 0:
            self.robotiqgrip.close_gripper()
        else:
            self.robotiqgrip.open_gripper()

        rospy.sleep(0.5)  # Wait for the gripper to complete the operation
        return True


if __name__ == "__main__":
    grasp_pos = [0.2, 0.1, 0.3]  # Example grasp position in `realsense_wrist_link`
    executor = GraspExecutor()
    executor.execute_grasp(grasp_pos)
