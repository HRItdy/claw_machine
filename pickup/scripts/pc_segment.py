import open3d as o3d
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from pc_calibration import Calibrator
from pc_transform import inverse_cloud

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

def generate_bounding_box(point3D, normal_vector, box_size):
    # Same function as before to generate corners of the bounding box
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    
    if normal_vector[0] != 0 or normal_vector[1] != 0:
        ortho_vector1 = np.array([-normal_vector[1], normal_vector[0], 0])
    else:
        ortho_vector1 = np.array([1, 0, 0])
    ortho_vector1 = ortho_vector1 / np.linalg.norm(ortho_vector1)
    ortho_vector2 = np.cross(normal_vector, ortho_vector1)
    
    half_size = box_size / 2.0
    corners = [
        point3D + half_size * (ortho_vector1 + ortho_vector2 + normal_vector),
        point3D + half_size * (ortho_vector1 - ortho_vector2 + normal_vector),
        point3D + half_size * (-ortho_vector1 + ortho_vector2 + normal_vector),
        point3D + half_size * (-ortho_vector1 - ortho_vector2 + normal_vector),
        point3D + half_size * (ortho_vector1 + ortho_vector2 - normal_vector),
        point3D + half_size * (ortho_vector1 - ortho_vector2 - normal_vector),
        point3D + half_size * (-ortho_vector1 + ortho_vector2 - normal_vector),
        point3D + half_size * (-ortho_vector1 - ortho_vector2 - normal_vector)
    ]
    
    return np.array(corners)

def filter_points_in_bounding_box(pointcloud, bbox_corners):
    bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bbox_corners))
    cropped_pc = pointcloud.crop(bbox)
    return cropped_pc

def create_bounding_box_marker(bbox_corners, frame_id="map"):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "bounding_box"
    marker.id = 0
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    
    marker.scale.x = 0.01  # Line width
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.color.a = 1.0  # Transparency

    # Define the 12 edges of the bounding box
    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),  # Bottom square
        (4, 5), (5, 7), (7, 6), (6, 4),  # Top square
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical lines
    ]
    
    for edge in edges:
        p1 = Point(x=bbox_corners[edge[0]][0], y=bbox_corners[edge[0]][1], z=bbox_corners[edge[0]][2])
        p2 = Point(x=bbox_corners[edge[1]][0], y=bbox_corners[edge[1]][1], z=bbox_corners[edge[1]][2])
        marker.points.append(p1)
        marker.points.append(p2)

    return marker

def pointcloud_callback(data, pc_pub, marker_pub):
    points = np.array(list(pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True))) # eliminate color in pointcloud2
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    
    # Eliminate the largest and second largest planes
    filtered_pc = eliminate_planes(pc)
    pc_publish(data.header, filtered_pc, pc_pub)
    
    # Convert the 2D point to 3D using the homography
    bottom = rospy.get_param('/pc_transform/bottom')
    point2D = np.array([bottom[0], bottom[1], 1]) 
    homography_matrix = np.array(rospy.get_param('/pc_transform/homography'))
    point3D = Calibrator.apply_homography(homography_matrix, point2D)
    point3D = np.array([[0.001, point3D[0], point3D[1]]]) # Here the x-axis is almost 0
    
    # Convert the 3D point to the original point cloud position
    point3D = inverse_cloud(point3D)
    # Get the normal vector from the ROS parameter server
    normal_vector = np.array(rospy.get_param('/pc_transform/normal'))
    
    # Generate the bounding box
    box_size = 0.2  # Replace with the desired box size
    bounding_box_corners = generate_bounding_box(point3D, normal_vector, box_size)
    
    # Filter points within the bounding box
    cropped_pc = filter_points_in_bounding_box(filtered_pc, bounding_box_corners)
    
    # Convert Open3D PointCloud to ROS PointCloud2
    ros_pc2 = pc2.create_cloud_xyz32(data.header, np.asarray(cropped_pc.points))
    
    # Publish the cropped point cloud
    pc_pub.publish(ros_pc2)
    
    # Create and publish the bounding box marker
    marker = create_bounding_box_marker(bounding_box_corners, frame_id=data.header.frame_id)
    marker_pub.publish(marker)

def pc_publish(header, pc, pc_pub):
    ros_pc2 = pc2.create_cloud_xyz32(header, np.asarray(pc.points))
    pc_pub.publish(ros_pc2)

def main():
    rospy.init_node('bounding_box_node')
    pc_pub = rospy.Publisher('/cropped_pointcloud', PointCloud2, queue_size=1)
    marker_pub = rospy.Publisher('/bounding_box_marker', Marker, queue_size=1)
    rospy.Subscriber('/realsense_wrist/depth_registered/points', PointCloud2, lambda data: pointcloud_callback(data, pc_pub, marker_pub))
    rospy.spin()

if __name__ == "__main__":
    main()
