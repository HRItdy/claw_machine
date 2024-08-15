import rospy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
import numpy as np
from store_mask_service import store_mask_client

def get_index(u, v, width):
    return v * width + u

def publish_pointcloud(points):
    pub = rospy.Publisher('output_pointcloud', PointCloud2, queue_size=10)
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
              PointField('y', 4, PointField.FLOAT32, 1),
              PointField('z', 8, PointField.FLOAT32, 1),
              PointField('rgb', 12, PointField.UINT32, 1)]
    
    header = Header(frame_id='realsense_wrist_color_optical_frame')
    header.stamp = rospy.Time.now()
    pc2_msg = pc2.create_cloud(header, fields, points)
    pub.publish(pc2_msg)

def pointcloud_callback(input_cloud):
    width = input_cloud.width
    mask, success = store_mask_client(store=False)
    
    if not success:
        rospy.logwarn("Failed to retrieve mask from service.")
        return

    indices = np.argwhere(mask)[2:, :].transpose(0, 1) # Assuming mask is a binary array; transpose to match (u, v) format

    # Retrieve 3D points
    points = []
    for (u, v) in indices:
        try:
            # Extract x, y, z, and rgb from the point cloud
            point = next(pc2.read_points(input_cloud, field_names=("x", "y", "z", "rgb"), skip_nans=True, uvs=[(u, v)]))
            points.append(point)
        except StopIteration:
            rospy.logwarn(f"Point at ({u}, {v}) is NaN or not available.")
    
    # Publish the new point cloud with the retrieved points
    if points:
        publish_pointcloud(points)
    else:
        rospy.logwarn("No valid points were retrieved from the point cloud based on the mask.")

def main():
    rospy.init_node('pointcloud_processor')
    rospy.Subscriber('/camera/color_point', PointCloud2, pointcloud_callback)
    rospy.spin()  # Keep the node running

if __name__ == '__main__':
    main()
