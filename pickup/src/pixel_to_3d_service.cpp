#include <ros/ros.h>
#include <cam_pkg/PixelTo3D.h>
#include <cam_pkg/rs.hpp>

bool pixelTo3D(cam_pkg::PixelTo3D::Request &req,
               cam_pkg::PixelTo3D::Response &res) {
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    pipe.start(cfg);

    rs2::frameset frames = pipe.wait_for_frames();
    rs2::depth_frame depth_frame = frames.get_depth_frame();
    if (!depth_frame) {
        return false;
    }

    rs2_intrinsics depth_intrin = depth_frame.get_profile().as<rs2::video_stream_profile>().get_intrinsics();
    float depth = depth_frame.get_distance(req.u, req.v);
    float pixel[2] = {static_cast<float>(req.u), static_cast<float>(req.v)};
    float point[3];
    rs2_deproject_pixel_to_point(point, &depth_intrin, pixel, depth);

    res.x = point[0];
    res.y = point[1];
    res.z = point[2];

    pipe.stop();
    return true;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "pixel_to_3d_service");
    ros::NodeHandle n;

    ros::ServiceServer service = n.advertiseService("pixel_to_3d", pixelTo3D);
    ROS_INFO("Ready to convert pixel to 3D point.");
    ros::spin();

    return 0;
}
