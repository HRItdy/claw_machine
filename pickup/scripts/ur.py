#!/usr/bin/env python

import rospy
import numpy as np
import actionlib
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from urx import Robot

#!/usr/bin/env python3
import rospy
from tf2_ros import Buffer, TransformListener, StaticTransformBroadcaster
import tf2_geometry_msgs
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from std_msgs.msg import Header, Bool, Float64
from geometry_msgs.msg import Point, PointStamped, Quaternion, PoseStamped, Pose, TransformStamped, Transform, Vector3
from sensor_msgs.msg import PointCloud2, JointState
import urx
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
import numpy as np
np.float = np.float64
import ros_numpy
import actionlib
from pickup.msg import pickupAction, pickupGoal, pickupResult, pickupFeedback
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

import sys
from collections import defaultdict
import pickle, os, rospkg
import argparse
from std_srvs.srv import Trigger

eff_link = 'gripper_tip_link'
arm_base_link = 'arm_base_link'
# j_tuck = [0, -1.27, -1.87, -2.72, 1.5708, 3.14159]
# j_see = [-1.5375927130328577, -1.316888157521383, 3.1278576850891113, -4.099831883107321, -1.5893304983722132, 3.1201136112213135] #[3.1416, -0.87, -1.75, -4.33, -1.5708, 3.14159] #[3.1416, -1., -1.8, -3.9, -1.5708, 3.14159]  #[-1.3883803526507776, -1.081475559865133, 3.14849591255188, -4.082708303128378, -1.5666244665728968, 3.1452038288116455]
# see_id_marker = {0: {'see_pos': [0.265, -0.663, 0.607], 'see_rot': [-2.556, 0, 0]},
#                  3: {'see_pos': [-0.265, -0.663, 0.607], 'see_rot': [-2.556, 0, 0]}
#                 }
j_hold = [3.1421918869018555, -0.7707122007953089, -2.359290901814596, 3.1296582221984863, -1.5698922316180628, -3.137442175542013] #[3.1416, -0.6981, -2.1817, -3.4208, -1.5708, 3.14159]
obj_height_rng = [0.1, 0.4]
# obj_dist_rng = [0.3, 0.95]
# obj_width_rng = [0.04, 0.30]
# table_min_height = 0.78
# max_num_detection = 5

sim_wristcam_frame = 'cam_wrist_depth_optical_frame'
real_wristcam_frame = 'realsense_wrist_depth_optical_frame'


class PGTask:
    def __init__(self, args):
        self.prev_ocr_value = None
        self.ocr_stable_count = 0
        self.next_place_pos = None

        self.rob = urx.Robot("192.168.0.6")
        self.robotiqgrip = Robotiq_Two_Finger_Gripper(self.rob)

        rospy.init_node('pickup', log_level=rospy.DEBUG)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)
        self.tf_broadcaster = StaticTransformBroadcaster()
        self.act_task = actionlib.SimpleActionServer('png_task', pickupAction, execute_cb=self.cb_act_task, auto_start=False)
        self.act_task.start()
        rospy.spin()

    def cb_act_task(self, goal):
        if goal.name == 'pickup':
            '''
            name: 'pickup'
            timeout: 0
            id_marker: 0
            xyzh: []
            pos: []
            rot: []
            jpose: 'j_tuck'
            '''
            self.task_pick_up(goal)
        # elif goal.name == 'move_gripper':
        #     '''
        #     name: 'move_gripper'
        #     timeout: 0
        #     id_marker: 0
        #     xyzh: []
        #     pos: [0.1]
        #     rot: []
        #     jpose: ''
        #     '''
        #     self.move_gripper(goal.pos[0])
        #     self.act_task.set_succeeded(taskResult("[move_gripper]: Completed."))
        # elif goal.name == 'move_base':
        #     '''
        #     name: 'move_base'
        #     timeout: 30.0
        #     id_marker: 0
        #     xyzh: [-0.4, -0.5, 0, 0]
        #     pos: []
        #     rot: []
        #     jpose: ''
        #     '''
        #     self.task_move_base(goal.id_marker, goal.xyzh, goal.timeout)
        # elif goal.name == 'move_camera':
        #     '''
        #     name: 'move_camera'
        #     timeout: 0
        #     id_marker: 16
        #     xyzh: []
        #     pos: [0, -0.25, 0.25]
        #     rot: [-2.7, 0, 0]
        #     jpose: ''
        #     '''
        #     self.task_move_camera(goal.pos, goal.rot, goal.id_marker)
        # elif goal.name == 'see_marker':
        #     '''
        #     name: 'see_marker'
        #     timeout: 3.0
        #     id_marker: 0
        #     xyzh: []
        #     pos: []
        #     rot: []
        #     jpose: ''
        #     '''
        #     self.task_see_marker(goal.timeout)
        # elif goal.name == 'see_bottle':
        #     '''
        #     name: 'see_bottle'
        #     timeout: 5.0
        #     id_marker: 0
        #     xyzh: []
        #     pos: []
        #     rot: []
        #     jpose: ''
        #     '''
        #     self.task_see_bottle(goal.timeout, goal.id_marker)
        # elif goal.name == 'pick_bottle':
        #     '''
        #     name: 'pick_bottle'
        #     timeout: 0
        #     id_marker: 0
        #     xyzh: []
        #     pos: []
        #     rot: []
        #     jpose: ''
        #     '''
        #     self.task_pick_bottle(goal.pos)
        # elif goal.name == 'place_bottle':
        #     '''
        #     name: 'place_bottle'
        #     timeout: 0
        #     id_marker: 5
        #     xyzh: []
        #     pos: []
        #     rot: []
        #     jpose: ''
        #     '''
        #     self.task_place_bottle(goal.pos, goal.id_marker)
        # elif goal.name == 'push_button':
        #     '''
        #     name: 'push_button'
        #     timeout: 0
        #     id_marker: 1
        #     xyzh: []
        #     pos: [0, 0, 0]
        #     rot: [0, 0, 0]
        #     jpose: ''
        #     '''
        #     self.task_push_button(goal.id_marker, goal.pos, goal.rot)
        # elif goal.name == 'pump_wait':
        #     '''
        #     name: 'pump_wait'
        #     timeout: 8
        #     id_marker: 0
        #     xyzh: []
        #     pos: []
        #     rot: []
        #     jpose: ''
        #     '''
        #     self.task_pump_wait(goal.timeout)
        # elif goal.name == 'screen_ocr':
        #     '''
        #     name: 'screen_ocr'
        #     timeout: 8
        #     id_marker: 0
        #     xyzh: []
        #     pos: []
        #     rot: []
        #     jpose: ''
        #     '''
        #     self.task_screen_ocr(goal.timeout)
        # elif goal.name == 'move_dispenser':
        #     '''
        #     name: 'move_dispenser'
        #     timeout: 0
        #     id_marker: 2
        #     xyzh: []
        #     pos: [0, 0, 0]
        #     rot: []
        #     jpose: 'push'
        #     '''
        #     self.task_move_dispenser(goal.id_marker, goal.pos, goal.jpose)
        
    def task_pick_up(self, goal, grip_depth=0.1, grip_up=0.05):
        self.act_task.publish_feedback(pickupFeedback('[pick_up]: Start picking.'))
        # get the segmented point cloud from topic
        
        # # select detected cluster
        # if len(self.detections['bottle']) == 0:
        #     self.act_task.set_aborted(taskResult('[pick_bottle]: No bottle detection.'))
        #     return
        # #     tf = self.tf_buffer.lookup_transform('world', 'map', rospy.Time.now(), rospy.Duration(0.1))
        # #     pos = tf2_geometry_msgs.do_transform_point(PointStamped(point=Point(*pos)), tf)
        # #     cluster = min(self.detections['bottle'], key=lambda x: ((x['pos'][0] - pos.point.x) ** 2 +
        # #                                                             (x['pos'][1] - pos.point.y) ** 2 +
        # #                                                             (x['pos'][2] - pos.point.z) ** 2) ** 0.5)
        # #     self.detections['bottle'] = cluster
        # else:
        #     # pick the closest bottle, compute distance to arm_base_link in world-frame
        #     # tf = self.tf_buffer.lookup_transform('world', arm_base_link, rospy.Time.now(), rospy.Duration(0.1))
        #     # pos = [tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z]
        #     # cluster = min(self.detections['bottle'], key=lambda x: ((x['pos'][0] - pos[0]) ** 2 +
        #     #                                                         (x['pos'][1] - pos[1]) ** 2 +
        #     #                                                         (x['pos'][2] - pos[2]) ** 2) ** 0.5)
        #     # default pick order: [right -> left, front -> back]
        #     cluster = min(self.detections['bottle'], key=lambda x: (10*x['pos'][0] + x['pos'][1]))
        #     self.detections['bottle'] = cluster

        # print("pick_pos:", cluster['pos'][0], cluster['pos'][1], max(cluster['pos'][2], table_min_height))

        # pre grasp
        self.act_task.publish_feedback(pickupFeedback('[pick-up]: Heading to pre-grasp.'))
        tf = self.tf_buffer.lookup_transform(arm_base_link, 'world', rospy.Time(0), rospy.Duration(0.1))
        pos = tf2_geometry_msgs.do_transform_point(PointStamped(point=Point(goal.pos[0], goal.pos[1], goal.pos[2])), tf)
        pos = [pos.point.x, pos.point.y, pos.point.z]
        dxy = pos[:2] / np.linalg.norm(pos[:2]) * grip_depth
        dz = np.arctan2(pos[1], pos[0])
        if not self.move_linear((pos[0] - dxy[0], pos[1] - dxy[1], pos[2]),
                                (0, 0, dz), #(-np.pi, 0, dz), 
                                frame_id=arm_base_link, avoid_collision=False, path_res=0.1):
            self.act_task.set_aborted(pickupResult('[pick_up]: Failed to move to pre-grasp.'))
            return

        # open gripper
        self.move_gripper(0, attach=False)
        # gripper in
        self.act_task.publish_feedback(pickupFeedback('[pick_up]: Heading to grasp-pose.'))
        if not self.move_linear((grip_depth, 0, 0)):
            self.act_task.set_aborted(pickupResult('[pick_up]: Failed to move to grasp-pose.'))
            return

        # close gripper
        self.act_task.publish_feedback(pickupFeedback('[pick_up]: Closing gripper.'))
        self.move_gripper(0.1, attach=True)
        # self.scene.attach_box(link=eff_link, name='bottle',
        #                       pose=PoseStamped(header=Header(frame_id=eff_link, stamp=rospy.Time()),
        #                                        pose=Pose(position=Point(0, 0, 0), orientation=Quaternion(0, 0, 0, 1))),
        #                       size=(cluster['width'], cluster['width'], cluster['height']),
        #                       touch_links=['gripper_left_inner_finger_pad', 'gripper_right_inner_finger_pad',
        #                                    'gripper_left_inner_finger', 'gripper_right_inner_finger',
        #                                    'gripper_left_inner_knuckle', 'gripper_right_inner_knuckle',
        #                                    'gripper_left_outer_knuckle', 'gripper_right_outer_knuckle',
        #                                    'gripper_left_outer_finger', 'gripper_right_outer_finger',
        #                                    'gripper_robotiq_arg2f_base_link', 'gripper_tip_link'])
        # attached_objects = self.scene.get_attached_objects(['bottle'])
        # while len(attached_objects.keys()) <= 0:
        #     rospy.sleep(0.1)

        # gripper up
        self.act_task.publish_feedback(pickupFeedback('[pick_up]: Lifting gripper.'))
        if not self.move_linear((0, 0, grip_up)):
            self.act_task.set_aborted(pickupResult('[pick_up]: Failed to lift gripper.'))
            return

        # gripper out
        self.act_task.publish_feedback(pickupFeedback('[pick_up]: Heading to post-grasp.'))
        if not self.move_linear((-grip_depth, 0, 0)):
            self.act_task.set_aborted(pickupResult('[pick_up]: Failed to move to post-grasp.'))
            return

        """
        # restart ur_ctrl after running urx scripts on real robot
        if self.real:
            self.ur_ctrl_play_srv()
            rospy.sleep(1.0)
        """

        # gripper hold
        self.act_task.publish_feedback(pickupFeedback('[pick_up]: Heading to hold-grasp.'))
        if not self.move_joint(j_hold):
            self.act_task.set_aborted(pickupResult('[pick_up]: Failed to move to hold-grasp.'))
            return

        self.act_task.set_succeeded(pickupResult('[pick_up]: Completed.'))
        
    
    # def cb_cluster(self, pc_msg):
    #     # transform pointcloud to robot base frame
    #     xyz = ros_numpy.numpify(pc_msg)
    #     xyz = np.stack([xyz['x'], xyz['y'], xyz['z'], np.ones_like(xyz['z'])], axis=0)
    #     tf = self.tf_buffer.lookup_transform(arm_base_link, pc_msg.header.frame_id, pc_msg.header.stamp, rospy.Duration(0.1))
    #     tf = ros_numpy.numpify(tf.transform)
    #     xyz = tf.dot(xyz)[:3]

    #     # filters
    #     obj_pos = xyz.mean(axis=1).tolist()
    #     # height
    #     obj_height = xyz[2].max() - xyz[2].min()
    #     print("height: ", obj_height)
    #     if not obj_height_rng[0] < obj_height < obj_height_rng[1]:
    #         return
    #     # distance
    #     obj_dist = (obj_pos[0] ** 2 + obj_pos[1] ** 2) ** 0.5
    #     print("distance: ", obj_dist)
    #     if not obj_dist_rng[0] < obj_dist < obj_dist_rng[1]:
    #         return
    #     # width
    #     obj_width = ((xyz[0].max() - xyz[0].min()) ** 2 +
    #                 (xyz[1].max() - xyz[1].min()) ** 2) ** 0.5
    #     print("width: ", obj_width)
    #     if not obj_width_rng[0] < obj_width < obj_width_rng[1]:
    #         return

    #     print("bottle detected")
    #     # add to detection collections
    #     tf = self.tf_buffer.lookup_transform('world', arm_base_link, pc_msg.header.stamp, rospy.Duration(0.1))
    #     obj_pos = tf2_geometry_msgs.do_transform_point(PointStamped(point=Point(*obj_pos)), tf).point
    #     self.detections['bottle'].append({'pos': [obj_pos.x, obj_pos.y, obj_pos.z],
    #                                       'height': obj_height,
    #                                       'width': obj_width})
    #     # unsubscribe
    #     if len(self.detections['bottle']) >= max_num_detection:
    #         if self.sub_cluster is not None:
    #             self.sub_cluster.unregister()
    #             self.sub_cluster = None
                
    def move_linear(self, xyz=(0, 0, 0), rpy=(0, 0, 0), pose=None, frame_id=eff_link): 
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', help="run png_action_server for real-world setup", action="store_true")
    args = parser.parse_args(rospy.myargv()[1:])
    PGTask(args)

