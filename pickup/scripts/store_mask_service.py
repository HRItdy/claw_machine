#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32MultiArray, MultiArrayDimension
from pickup.srv import StoreMask, StoreMaskResponse
import numpy as np

# Global variable to store the mask
stored_mask = None

def handle_store_mask(req):
    global stored_mask
    if req.store:
        # Store the mask
        stored_mask = np.array(req.mask.data).reshape(req.mask.layout.dim[0].size, -1)
        rospy.loginfo("mask stored successfully.")
        return StoreMaskResponse(stored_mask=req.mask, success=True)
    else:
        # Retrieve the stored mask
        if stored_mask is not None:
            response_mask = Int32MultiArray()
            response_mask.data = stored_mask.flatten().tolist()
            dim = MultiArrayDimension()
            dim.size = stored_mask.shape[0]
            dim.stride = stored_mask.size
            dim.label = "rows"
            response_mask.layout.dim = [dim]
            return StoreMaskResponse(stored_mask=response_mask, success=True)
        else:
            rospy.logwarn("No mask is stored yet.")
            return StoreMaskResponse(stored_mask=Int32MultiArray(), success=False)

def store_mask_server():
    rospy.init_node('store_mask_server')
    s = rospy.Service('store_mask', StoreMask, handle_store_mask)
    rospy.loginfo("Ready to store and retrieve mask.")
    rospy.spin()

def store_mask_client(mask=None, store=True):
    rospy.wait_for_service('store_mask')
    store_mask = rospy.ServiceProxy('store_mask', StoreMask)
    array_msg = Int32MultiArray()
    if mask is not None:
        array_msg.data = mask.flatten().tolist()
        dim = MultiArrayDimension()
        dim.size = mask.shape[0]
        dim.stride = mask.size
        dim.label = "rows"
        array_msg.layout.dim = [dim]
    resp = store_mask(store, array_msg)
    stored_mask = np.array(resp.stored_mask.data).reshape(resp.stored_mask.layout.dim[0].size, -1)
    return stored_mask, resp.success

if __name__ == "__main__":
    store_mask_server()
