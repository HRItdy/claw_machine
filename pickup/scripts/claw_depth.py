import open3d as o3d
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from pc_calibration import Calibrator
from geometry_msgs.msg import PointStamped
from scipy.optimize import leastsq
from pickup.srv import Get3DPosition, Get3DPositionRequest, Get3DPositionResponse  # Replace 'your_package' with your actual package name

# TODO:
# Now only filter the distance with the porjected point, need to calculate z-axis coordinate and filter it using a ball
# Transform the pointcloud, and re-try whether rs2 can work

def eliminate_planes(pointcloud, num_planes=2):
    for _ in range(num_planes):
        plane_model, inliers = pointcloud.segment_plane(distance_threshold=0.01,
                                                        ransac_n=3,
                                                        num_iterations=1000)
        if len(inliers) == 0:
            print("No plane found.")
            break
        
        # Remove the plane points
        pointcloud = pointcloud.select_by_index(inliers, invert=True)
    
    return pointcloud

def create_line_marker(point3D, frame_id):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "line_marker"
    marker.id = 0
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD

    # Set the scale of the marker - line width
    marker.scale.x = 0.01  # Line width

    # Set the color - Red
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.color.a = 1.0

    # Set the marker points
    for z in np.linspace(0, 2, num=10):  # Generate points from z=0 to z=2
        point = (0.4, 0.1, z)
        p = Point()
        p.x = point[0]
        p.y = point[1]
        p.z = point[2]
        marker.points.append(p)

    return marker

def create_point_marker(point3D, frame_id, color):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "point_marker"
    marker.id = 0
    marker.type = Marker.POINTS
    marker.action = Marker.ADD

    # Set the scale of the marker - point size
    marker.scale.x = 0.02  # Point width
    marker.scale.y = 0.02  # Point height

    # Set the color based on user input
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]

    # Add the input 3D point
    p = Point()
    p.x = point3D[0]
    p.y = point3D[1]
    p.z = point3D[2]
    marker.points.append(p)

    k = Point()
    k.x = 0
    k.y = 0
    k.z = 0
    marker.points.append(k)

    # For the pointcloud frame, if u want to map a point onto the pointcloud
    # pointcloud.x = -point.y
    # pointcloud.y = -point.z
    # pointcloud.z = point.x

    # If want to map the pointcloud to point frame
    # point.x = pointcloud.z
    # point.y = -pointcloud.x
    # point.z = -pointcloud.y
    aim = Point()   #(0.39, 0.096, -0.02)
    aim.x = -0.096
    aim.y = 0.02
    aim.z = 0.39
    marker.points.append(aim)

    return marker

def transform(points_A):
    # Define the transformation matrix
    T = np.array([
        [0, 0, 1, 0],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ])
    # Convert pointcloud_A to homogeneous coordinates (N x 4)
    ones = np.ones((points_A.shape[0], 1))
    points_A_hom = np.hstack((points_A, ones))
    # Apply the transformation matrix
    points_B_hom = T @ points_A_hom.T
    # Convert back to 3D coordinates
    points_B = points_B_hom[:3, :].T
    return points_B

def inverse_transform(points_B):
    # Define the transformation matrix
    T = np.array([
        [0, 0, 1, 0],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ])
    # Calculate the inverse of the transformation matrix
    T_inv = np.linalg.inv(T)
    # Convert pointcloud_B to homogeneous coordinates (N x 4)
    ones = np.ones((points_B.shape[0], 1))
    points_B_hom = np.hstack((points_B, ones))
    # Apply the inverse transformation matrix
    points_A_hom = T_inv @ points_B_hom.T
    # Convert back to 3D coordinates
    points_A = points_A_hom[:3, :].T
    return points_A

def inverse_transform_for_point(point_B):
    # Define the transformation matrix
    T = np.array([
        [0, 0, 1, 0],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ])
    # Calculate the inverse of the transformation matrix
    T_inv = np.linalg.inv(T)
    # Convert point_B to homogeneous coordinates (1 x 4)
    point_B_hom = np.append(point_B, 1)
    # Apply the inverse transformation matrix
    point_A_hom = T_inv @ point_B_hom
    # Convert back to 3D coordinates
    point_A = point_A_hom[:3]
    return point_A

# def pointcloud_callback(data, pc_pub, marker_pub, centroid_pub):
#     points_ = np.array(list(pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True))) # eliminate color in pointcloud2
#     # Transform the pointcloud (there is a dismatch between the frame of realsense depth camera and realsense_wrist_link.)
#     # Transform from realsense depth -> realsense_wrist_link
#     points = transform(points_)
#     pc = o3d.geometry.PointCloud()
#     pc.points = o3d.utility.Vector3dVector(points)

#     # Eliminate the largest and second largest planes
#     filtered_pc = eliminate_planes(pc)
    
#     # Convert the 2D point to 3D using the homography
#     H = np.array(rospy.get_param('/calibration/H'))
#     point2D = np.array(rospy.get_param('/pc_transform/bottom')) 
#     point3D = Calibrator.transform_2d_to_3d(point2D, H)

#     # Extract the pointcloud
#     min_z = -0.3 # bound of z-axis
#     max_z = 0.3

#     # Convert filtered_pc.points to a numpy array
#     filtered_points = np.asarray(filtered_pc.points)
#     cropped_pc = extract_cylinder_points(filtered_points, point3D[0], point3D[1], 0.06, min_z, max_z)
    
#     # Convert numpy array back to Open3D PointCloud
#     cropped_o3d_pc = o3d.geometry.PointCloud()
#     cropped_o3d_pc.points = o3d.utility.Vector3dVector(cropped_pc)
    
#     # Convert Open3D PointCloud to ROS PointCloud2
#     # ros_pc2 = pc2.create_cloud_xyz32(data.header, np.asarray(cropped_o3d_pc.points))
#     # ros_pc2 = pc2.create_cloud_xyz32(data.header, cropped_pc)
#     # # Publish the cropped point cloud
#     # pc_pub.publish(ros_pc2)

#     # Create and publish the line marker
#     marker = create_point_marker([0.4, 0.1, 0], data.header.frame_id, [1, 0, 0, 1])
#     marker_pub.publish(marker)

#     # The target regional pointcloud has been segmented out. Filter out the noise points
#     fc_ = filter(cropped_o3d_pc)
#     # Convert Open3D PointCloud to ROS PointCloud2
#     # ros_pc2 = pc2.create_cloud_xyz32(data.header, np.asarray(cropped_o3d_pc.points))
#     ros_pc2 = pc2.create_cloud_xyz32(data.header, fc_.points)
#     # Publish the cropped point cloud
#     pc_pub.publish(ros_pc2)

#     fc = np.asarray(fc_.points)
#     # Get the centroid
#     cluster(fc, point3D, data.header, centroid_pub)

def filter(pcd):
    # pcd = pcd.voxel_down_sample(voxel_size=0.05)
    # # Step 1: Apply Statistical Outlier Removal
    # pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    # filtered_pcd = pcd.select_by_index(ind)

    # # # Step 2: Apply Radius Outlier Removal
    # filtered_pcd, ind = filtered_pcd.remove_radius_outlier(nb_points=16, radius=0.05)
    # filtered_pcd = filtered_pcd.select_by_index(ind)

    # # Visualize the point cloud after noise removal
    # print("Point cloud after noise removal:")
    # o3d.visualization.draw_geometries([filtered_pcd])

    # Step 3: Apply DBSCAN clustering to keep the largest cluster
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    #     labels = np.array(filtered_pcd.cluster_dbscan(eps=0.05, min_points=10, print_progress=True))

    # max_label = labels.max()
    # print(f"point cloud has {max_label + 1} clusters")

    # # Retain only the largest cluster
    # largest_cluster_idx = np.where(labels == np.bincount(labels[labels >= 0]).argmax())[0]
    # filtered_pcd = filtered_pcd.select_by_index(largest_cluster_idx)
    # Downsample the point cloud (optional)
    pcd = pcd.voxel_down_sample(voxel_size=0.01)
    # Estimate normals (optional)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    # Perform DBSCAN clustering
    labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))  # `eps`` is the distance that defines the neighborhood of a point, 
                                                                                         # and `min_points`` is the minimum number of points required to form a cluster
    # Find the largest cluster
    max_label = labels.max()
    largest_cluster = pcd.select_by_index(np.where(labels == max_label)[0])
    return largest_cluster


def extract_cylinder_points(pointcloud, center_x, center_y, radius, min_z=-1, max_z=1):
    points = []
    for point in pointcloud:
        distances_sq = (point[0] - center_x) ** 2 + (point[1] - center_y) ** 2
        if distances_sq <= radius ** 2 and min_z <= point[2] <= max_z:
            points.append(point)
    cylinder_points = np.array(points)
    return cylinder_points

def cluster(points, estimate, header, centroid_pub):
    if points.shape[0] > 3:  # Need at least 4 points to fit a sphere
        center, radius = fit_sphere(points)
        # Exam whether this center is far away from the estimated center
        center = None if (center[0] - estimate[0]) ** 2 + (center[1] - estimate[1]) ** 2 > 0.06 ** 2 else center

        if center is not None:
            rospy.loginfo(f'Estimate Successfully: {center}')
            rospy.set_param('/pc_transform/center', center.tolist())
            # Create a PointStamped message for the centroid
            centroid_msg = PointStamped()
            centroid_msg.header = header
            centroid_msg.point.x = center[0]
            centroid_msg.point.y = center[1]
            centroid_msg.point.z = center[2]
            # Publish the centroid
            centroid_pub.publish(centroid_msg)

def fit_sphere(points):
    # Initial guess for the center and radius
    x_m = np.mean(points[:, 0])
    y_m = np.mean(points[:, 1])
    z_m = np.mean(points[:, 2])
    r_guess = np.mean(np.linalg.norm(points - np.array([x_m, y_m, z_m]), axis=1))

    def residuals(p):
        center, radius = p[:3], p[3]
        return np.linalg.norm(points - center, axis=1) - radius

    # Initial guess: [center_x, center_y, center_z, radius]
    p0 = [x_m, y_m, z_m, r_guess]
    result = leastsq(residuals, p0)

    center = result[0][:3]
    radius = result[0][3]
    return center, radius

def pointcloud_callback(data):
    points_ = np.array(list(pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True))) # eliminate color in pointcloud2
    # Transform the pointcloud (there is a dismatch between the frame of realsense depth camera and realsense_wrist_link.)
    # Transform from realsense depth -> realsense_wrist_link
    points = transform(points_)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)

    # Eliminate the largest and second largest planes
    filtered_pc = eliminate_planes(pc)
    
    # Convert the 2D point to 3D using the homography
    H = np.array(rospy.get_param('/calibration/H'))
    point2D = np.array(rospy.get_param('/pc_transform/bottom')) 
    point3D = Calibrator.transform_2d_to_3d(point2D, H)

    # Extract the pointcloud
    min_z = -0.3 # bound of z-axis
    max_z = 0.3

    # Convert filtered_pc.points to a numpy array
    filtered_points = np.asarray(filtered_pc.points)
    cropped_pc = extract_cylinder_points(filtered_points, point3D[0], point3D[1], 0.06, min_z, max_z)
    
    # Convert numpy array back to Open3D PointCloud
    cropped_o3d_pc = o3d.geometry.PointCloud()
    cropped_o3d_pc.points = o3d.utility.Vector3dVector(cropped_pc)
    
    # Filter out the noise points
    fc_ = filter(cropped_o3d_pc)
    fc = np.asarray(fc_.points)

    # Get the centroid
    center, radius = fit_sphere(fc)

    # Exam whether this center is far away from the estimated center
    center = None if (center[0] - point3D[0]) ** 2 + (center[1] - point3D[1]) ** 2 > 0.06 ** 2 else center

    if center is not None:
        rospy.loginfo(f'Estimate Successfully: {center}')
        return center
    else:
        return None

def get_3d_position(req):
    pointcloud_data = rospy.wait_for_message('/realsense_wrist/depth_registered/points', PointCloud2)
    center = pointcloud_callback(pointcloud_data)
    
    if center is not None:
        return Get3DPositionResponse(Point(center[0], center[1], center[2]))
    else:
        rospy.logwarn("Failed to estimate the 3D position.")
        return Get3DPositionResponse(Point())

def get_3d_position_server():
    rospy.init_node('get_3d_position_server')
    service = rospy.Service('get_3d_position', Get3DPosition, get_3d_position)
    rospy.loginfo("Ready to get 3D position.")
    rospy.spin()

# def main():
#     rospy.init_node('bounding_box_node')
#     pc_pub = rospy.Publisher('/cropped_pointcloud', PointCloud2, queue_size=1)
#     marker_pub = rospy.Publisher('/bounding_box_marker', Marker, queue_size=1)
#     centroid_pub = rospy.Publisher("/ball_centroid", PointStamped, queue_size=10)
#     rospy.Subscriber('/realsense_wrist/depth_registered/points', PointCloud2, lambda data: pointcloud_callback(data, pc_pub, marker_pub, centroid_pub))
#     rospy.spin()

if __name__ == "__main__":
    # main()
    get_3d_position_server()
