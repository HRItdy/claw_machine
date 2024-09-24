#!/usr/bin/env python3
import rospy
from pickup.srv import GroundingDINO, GroundingDINORequest, SamPoint, SamPointRequest, OwlGpt, OwlGptRequest, GraspService, GraspServiceRequest, Get3DPosition, Get3DPositionRequest, Get3DPositionResponse

def call_detect_service(instruction):
    rospy.wait_for_service('grounding_dino')
    grounding_dino = rospy.ServiceProxy('grounding_dino', GroundingDINO)
    request = GroundingDINORequest(instruction=instruction, fast_sam=False)
    response = grounding_dino(request)
    print(f"Received coordinates: cX={response.cX}, cY={response.cY}")
    return response.cX, response.cY

def call_segment_service(x, y, fast_sam=True):
    rospy.wait_for_service('sam_point')
    sam_service = rospy.ServiceProxy('sam_point', SamPoint)
    request = SamPointRequest(cX=x, cY=y, fast_sam=fast_sam)
    response = sam_service(request)
    print('SAM recognition with point is done')

def call_owlgpt_service(input, enhance, fast_sam=True):
    rospy.wait_for_service('owl_gpt')
    prompt = ["a red ball", "a purple ball"] # for owl-vit to detect all the balls
    prompt.append(input)
    prompt = ','.join(prompt)
    owl_service = rospy.ServiceProxy('owl_gpt', OwlGpt)
    request = OwlGptRequest(instruction=prompt, enhance=enhance, fast_sam=fast_sam)
    response = owl_service(request)
    print('owl recognition with point is done')

def call_depth_service():
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
        # pub = rospy.Publisher('/extract_3d_position', PointStamped, queue_size=10)
        # rate = rospy.Rate(10)  # 10 Hz
        # while not rospy.is_shutdown():
        #     # pos = np.dot(self.R_z, pos)
        #     # Create the PointStamped message
        #     point_stamped = PointStamped()
        #     point_stamped.header.stamp = rospy.Time.now()
        #     point_stamped.header.frame_id = 'realsense_wrist_link'
        #     point_stamped.point.x = position[0]
        #     point_stamped.point.y = position[1]
        #     point_stamped.point.z = position[2]
        #     # Publish the transformed point
        #     pub.publish(point_stamped)           
        #     rate.sleep()
        return position
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        return None
    
def call_pickup_service():
    rospy.wait_for_service('grasp_service')
    name = "example_grasp"
    timeout = 5.0
    id_marker = 1
    xyzh = [0.0, 0.0, 0.0, 0.0]
    pos = rospy.get_param('/3d_position')
    rot = [0.0, 0.0, 0.0, 1.0]  # Example quaternion for no rotation
    jpose = "home"
    try:
        grasp_service = rospy.ServiceProxy('grasp_service', GraspService)
        req = GraspServiceRequest(
            name=name,
            timeout=timeout,
            id_marker=id_marker,
            xyzh=xyzh,
            pos=pos,
            rot=rot,
            jpose=jpose
        )
        response = grasp_service(req)
        if response.success:
            rospy.loginfo("Grasp action completed successfully.")
        else:
            rospy.logwarn("Grasp action is still in progress or failed.")
        return response.success
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s" % e)
        return False

if __name__ == '__main__':
    x, y = call_detect_service('pick up the left most red ball')
    print('detection done')

    # call_segment_service(233, 111)

    # input = "a purple ball between two red balls"
    # call_owlgpt_service(input, True, True)