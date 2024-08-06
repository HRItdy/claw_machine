#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

namespace LargestCluster
{

class LargestClusterExtractor : public nodelet::Nodelet
{
public:
  virtual void onInit()
  {
    ros::NodeHandle& private_nh = getPrivateNodeHandle();
    sub_ = private_nh.subscribe("input", 1, &LargestClusterExtractor::cloudCallback, this);
    pub_ = private_nh.advertise<pcl::PointCloud<pcl::PointXYZ>>("output", 1);
  }

private:
  void cloudCallback(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud_msg)
  {
    if (cloud_msg->points.empty()) return;

    // Find the largest cluster
    size_t largest_cluster_size = 0;
    pcl::PointCloud<pcl::PointXYZ>::Ptr largest_cluster(new pcl::PointCloud<pcl::PointXYZ>);

    for (const auto& cluster : cloud_msg->height == 1 ? std::vector<pcl::PointCloud<pcl::PointXYZ>>{*cloud_msg} : *reinterpret_cast<const std::vector<pcl::PointCloud<pcl::PointXYZ>>*>(cloud_msg.get()))
    {
      if (cluster.points.size() > largest_cluster_size)
      {
        largest_cluster_size = cluster.points.size();
        *largest_cluster = cluster;
      }
    }

    if (largest_cluster->points.empty())
    {
      ROS_WARN("No clusters found");
      return;
    }

    // Publish the largest cluster
    largest_cluster->header = cloud_msg->header;
    pub_.publish(largest_cluster);
  }

  ros::Subscriber sub_;
  ros::Publisher pub_;
};

}  // namespace LargestCluster

PLUGINLIB_EXPORT_CLASS(LargestCluster::LargestClusterExtractor, nodelet::Nodelet)