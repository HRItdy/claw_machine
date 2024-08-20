#!/usr/bin/env python

import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import struct
import open3d as o3d
from std_msgs.msg import Header

def transform_cloud(cloud_msg):
    # Convert PointCloud2 to numpy array
    points = np.array(list(pc2.read_points(cloud_msg, skip_nans=True, field_names=("x", "y", "z"))))

    # Create an open3d point cloud from numpy array
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Remove outliers using Radius Outlier Removal
    pcd, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
    
    # Plane segmentation using RANSAC
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                             ransac_n=3,
                                             num_iterations=1000)

    # Extract the plane points
    plane_points = pcd.select_by_index(inliers)
    points = np.asarray(pcd.points)

    # Compute the transformation matrix to align the plane with the z-axis
    normal = np.array(plane_model[:3])
    rotation_matrix = calculate_rotation_matrix(normal)
    print('Rotation matrix is:', rotation_matrix)

    rospy.set_param('/pc_transform/normal', normal.tolist())
    rospy.set_param('/pc_transform/rotation_matrix', rotation_matrix.tolist())
    rospy.loginfo("Normal of the biggest plane has been stored to /pc_transform/normal.")
    rospy.loginfo("Rotation of the plane has been stored to /pc_transform/rotation_matrix.")

    # Apply the rotation to the entire point cloud
    #rotated_points = points.dot(rotation_matrix.T)
    rotated_points = apply_transformation(points, rotation_matrix)
    # # Translate the points so that the plane lies on z=0
    # plane_z_values = rotated_points[inliers][:, 2]
    # translation_z = -np.mean(plane_z_values)
    # rospy.set_param('/pc_transform/z_bias', translation_z.item())
    # rospy.loginfo("Translate along z-axis has been stored to /pc_transform/z_bias")
    # rotated_points[:, 0] += translation_z  # add translation_z here, so when transform back, remember to first substract.

    origin = reverse_transformation(rotated_points, rotation_matrix)
    # Publish the transformed cloud
    # transformed_cloud_msg = create_pointcloud2_msg(rotated_points, cloud_msg.header)
    transformed_cloud_msg = create_pointcloud2_msg(origin, cloud_msg.header)
    transformed_cloud_pub.publish(transformed_cloud_msg)

def calculate_rotation_matrix(normal_vector):
    # Normalize the normal vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    # Define the target z-axis vector
    z_axis = np.array([0, 0, 1])

    # Calculate the cross product and sine of the angle
    cross_product = np.cross(normal_vector, z_axis)
    sin_angle = np.linalg.norm(cross_product)
    cos_angle = np.dot(normal_vector, z_axis)

    if sin_angle == 0:
        # The vectors are already aligned, return the identity matrix
        return np.eye(3)

    # Create the skew-symmetric cross-product matrix
    cross_product_matrix = np.array([
        [0, -cross_product[2], cross_product[1]],
        [cross_product[2], 0, -cross_product[0]],
        [-cross_product[1], cross_product[0], 0]
    ])

    # Calculate the rotation matrix using the Rodrigues' rotation formula
    rotation_matrix = (
        np.eye(3) +
        cross_product_matrix +
        np.matmul(cross_product_matrix, cross_product_matrix) * (1 - cos_angle) / (sin_angle ** 2)
    )

    return rotation_matrix

def calculate_transformation_matrix(points_2d, points_3d):
    """
    Calculate the transformation matrix from 2D to 3D using corresponding points.
    
    :param points_2d: List of 2D points [(x1, y1), (x2, y2), ...]
    :param points_3d: List of corresponding 3D points [(X1, Y1, Z1), (X2, Y2, Z2), ...]
    :return: 3x3 transformation matrix
    """
    assert len(points_2d) == len(points_3d), "Number of 2D and 3D points must be the same"
    assert len(points_2d) >= 4, "At least 4 point correspondences are required"

    A = []
    b = []
    for (x, y), (X, Y, Z) in zip(points_2d, points_3d):
        A.append([x, y, 1, 0, 0, 0, -X*x, -X*y])
        A.append([0, 0, 0, x, y, 1, -Y*x, -Y*y])
        b.append(X)
        b.append(Y)

    A = np.array(A)
    b = np.array(b)

    # Solve the system of linear equations
    h = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Reshape the solution into a 3x3 matrix
    H = np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1]
    ])

    return H

def transform_2d_to_3d(point_2d, H):
    """
    Transform a 2D point to 3D using the transformation matrix.
    
    :param point_2d: 2D point (x, y)
    :param H: 3x3 transformation matrix
    :return: 3D point (X, Y, Z)
    """
    x, y = point_2d
    p_2d = np.array([x, y, 1])
    p_3d = np.dot(H, p_2d)
    X, Y, Z = p_3d / p_3d[2]  # Normalize by Z
    return X, Y, Z

    
def create_pointcloud2_msg(points, header):
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
    ]

    cloud_data = []
    for p in points:
        cloud_data.append(struct.pack('fff', *p[:3]))

    cloud_msg = PointCloud2()
    cloud_msg.header = header
    cloud_msg.height = 1
    cloud_msg.width = len(points)
    cloud_msg.fields = fields
    cloud_msg.is_bigendian = False
    cloud_msg.point_step = 12
    cloud_msg.row_step = cloud_msg.point_step * len(points)
    cloud_msg.is_dense = True
    cloud_msg.data = b''.join(cloud_data)

    return cloud_msg

def callback(msg):
    transform_cloud(msg)

if __name__ == '__main__':
    rospy.init_node('pointcloud_transformer')

    input_topic = rospy.get_param('~input_topic', '/realsense_wrist/depth_registered/points')
    output_topic = rospy.get_param('~output_topic', '/transformed_pointcloud2')

    rospy.Subscriber(input_topic, PointCloud2, callback)
    transformed_cloud_pub = rospy.Publisher(output_topic, PointCloud2, queue_size=10)

    rospy.spin()
