from pickup.srv import GraspService, GraspServiceResponse, AbortService, AbortServiceResponse
from pickup.msg import pickupAction, pickupGoal  
from std_msgs.msg import Bool
import actionlib
import rospy

DROP = [0.58, 0.0713, 0.485] # The pass position under 'base_link' frame
class GraspServiceServer:
    def __init__(self):
        rospy.init_node('grasp_service_server', log_level=rospy.DEBUG)
        # Set the position of the pass station
        rospy.set_param('/pass_position', DROP)
        # Signal publish topics
        self.grasp_pub = rospy.Publisher('/grasp_finished', Bool, queue_size=1)
        self.confirm_pub = rospy.Publisher('/confirm_finished', Bool, queue_size=1)
        self.pass_pub = rospy.Publisher('/pass_finished', Bool, queue_size=1)

        self.client = actionlib.SimpleActionClient('grasp_action', pickupAction)
        self.client.wait_for_server()
        self.service = rospy.Service('grasp_service', GraspService, self.handle_grasp_request)
        self.abort_service = rospy.Service('abort_service', AbortService, self.handle_abort_request)

    def handle_grasp_request(self, req):
        if self.client.get_state() == actionlib.GoalStatus.ACTIVE:
            rospy.logwarn("Grasp action is still in progress. Postponing request.")
            return GraspServiceResponse(success=False)

        goal = pickupGoal(
            name=req.name,
            timeout=req.timeout,
            id_marker=req.id_marker,
            xyzh=req.xyzh,
            pos=req.pos,
            rot=req.rot,
            jpose=req.jpose
        )
        self.client.send_goal(goal, done_cb=lambda state, result: self.task_done_callback(state, result, goal.name))
        self.client.wait_for_result()

        result = self.client.get_result()
        return GraspServiceResponse(success=(result.result == "Action executed successfully"))
    
    def handle_abort_request(self, req):
        # Check if there is an active goal and cancel it
        if self.client.get_state() == actionlib.GoalStatus.ACTIVE:
            rospy.loginfo("Aborting current action...")
            self.client.cancel_goal()
            rospy.loginfo("Action aborted successfully.")
            return AbortServiceResponse(success=True, message="Action aborted successfully.")
        else:
            rospy.logwarn("No active action to abort.")
            return AbortServiceResponse(success=False, message="No active action to abort.")
    
    def task_done_callback(self, state, result, task_name):
        # Callback to handle task completion based on the task_name
        rospy.loginfo(f"{task_name} action finished with state: {state}")
        if task_name == 'grasp':
            # Handle completion of the 'grasp' task
            rospy.loginfo("Grasp task completed.")
            self.grasp_pub.publish(True)  # Notify GUI that grasp is finished
        elif task_name == 'confirm':
            rospy.loginfo("Confirm task completed.")
            self.confirm_pub.publish(True)
        elif task_name == 'pass':
            rospy.loginfo("Pass task completed.")
            self.pass_pub.publish(True)
        else:
            rospy.logwarn(f"Unknown task: {task_name}")

if __name__ == "__main__":
    server = GraspServiceServer()
    rospy.spin()
