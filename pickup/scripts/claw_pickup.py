from pickup.srv import GraspService, GraspServiceResponse 
from pickup.msg import pickupAction, pickupGoal  
import actionlib
import rospy

class GraspServiceServer:
    def __init__(self):
        rospy.init_node('grasp_service_server', log_level=rospy.DEBUG)
        self.client = actionlib.SimpleActionClient('grasp_action', pickupAction)
        self.client.wait_for_server()

        self.service = rospy.Service('grasp_service', GraspService, self.handle_grasp_request)

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
        self.client.send_goal(goal)
        self.client.wait_for_result()

        result = self.client.get_result()
        return GraspServiceResponse(success=(result.result == "Grasp executed successfully"))


if __name__ == "__main__":
    server = GraspServiceServer()
    rospy.spin()
