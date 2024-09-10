#!/usr/bin/env python3
import rospy
from pickup.srv import GroundingDINO, GroundingDINORequest, SamPoint, SamPointRequest, OwlGpt, OwlGptRequest

def call_detect_service():
    rospy.wait_for_service('grounding_dino')
    grounding_dino = rospy.ServiceProxy('grounding_dino', GroundingDINO)
    request = GroundingDINORequest(instruction='pick up the left red ball', fast_sam=True)
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

if __name__ == '__main__':
    x, y = call_detect_service()
    # print('detection done')

    # call_segment_service(233, 111)

    # input = "a purple ball between two red balls"
    # call_owlgpt_service(input, True, True)