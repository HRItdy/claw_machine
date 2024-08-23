import pcl_ros
def load_point_cloud(file_path):
    cloud = pcl.load(file_path)
    return cloud

def downsample_point_cloud(cloud, leaf_size=0.01):
    voxel_grid = cloud.make_voxel_grid_filter()
    voxel_grid.set_leaf_size(leaf_size, leaf_size, leaf_size)
    cloud_filtered = voxel_grid.filter()
    return cloud_filtered

def remove_outliers(cloud, mean_k=50, std_dev_mul_thresh=1.0):
    sor = cloud.make_statistical_outlier_filter()
    sor.set_mean_k(mean_k)
    sor.set_std_dev_mul_thresh(std_dev_mul_thresh)
    cloud_filtered = sor.filter()
    return cloud_filtered

def segment_clusters(cloud, cluster_tolerance=0.02, min_cluster_size=100, max_cluster_size=25000):
    tree = cloud.make_kdtree()
    ec = cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(cluster_tolerance)
    ec.set_MinClusterSize(min_cluster_size)
    ec.set_MaxClusterSize(max_cluster_size)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()
    return cluster_indices

def extract_clusters(cloud, cluster_indices):
    clusters = []
    for indices in cluster_indices:
        cluster = pcl.PointCloud()
        points = []
        for idx in indices:
            points.append(cloud[idx])
        cluster.from_list(points)
        clusters.append(cluster)
    return clusters

def find_largest_cluster(clusters):
    largest_cluster = max(clusters, key=lambda cluster: cluster.size)
    return largest_cluster

def fit_sphere_to_cluster(cluster):
    seg = cluster.make_segmenter_normals(ksearch=50)
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_SPHERE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_MaxIterations(1000)
    seg.set_DistanceThreshold(0.01)
    
    inliers, coefficients = seg.segment()
    if len(inliers) == 0:
        raise ValueError("Could not estimate a spherical model for the given cluster.")
    
    center = coefficients[:3]
    return center

# Example usage
file_path = "path_to_your_point_cloud.pcd"
cloud = load_point_cloud(file_path)
cloud_filtered = downsample_point_cloud(cloud)
cloud_filtered = remove_outliers(cloud_filtered)
cluster_indices = segment_clusters(cloud_filtered)
clusters = extract_clusters(cloud_filtered, cluster_indices)
largest_cluster = find_largest_cluster(clusters)
center = fit_sphere_to_cluster(largest_cluster)
print("Center of the sphere:", center)
