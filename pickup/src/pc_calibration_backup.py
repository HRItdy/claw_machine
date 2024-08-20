import rospy
from sensor_msgs.msg import Image
import matplotlib.pyplot as plt
import numpy as np

class Calibrator:
    def __init__(self):
        self.image = None
        self.pts_2d = []
        self.pts_3d = []

        rospy.init_node('calibration', anonymous=True)
        rospy.Subscriber('/realsense_wrist/color/image_raw', Image, self.image_callback)

    def image_callback(self, msg):
        height = msg.height
        width = msg.width
        channels = 3  # Assuming RGB image
        self.image = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, channels))

    def on_click(self, event):
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            self.pts_2d.append((x, y))
            print(f"Point {len(self.pts_2d)}: ({x}, {y})")
            plt.scatter(x, y, color='red')
            plt.annotate(f"{len(self.pts_2d)}", (x, y))
            plt.draw()

            if len(self.pts_2d) == 4:
                rospy.set_param('/calibration/points_2d', self.pts_2d)
                rospy.loginfo("2D points saved to ROS parameter server.")
                self.input_3d_points()

    def input_3d_points(self):
        print("Enter the 3D coordinates for the 4 points (x, y, z):")
        for i in range(4):
            point = input(f"Point {i+1}: ").strip()
            self.pts_3d.append([float(coord) for coord in point.split()])

        self.pts_3d = np.array(self.pts_3d)
        rospy.set_param('/calibration/points_3d', self.pts_3d.tolist())
        rospy.loginfo("3D points saved to ROS parameter server.")
        self.compute_and_store_homography()

    def compute_and_store_homography(self):
        pts_2d = np.array(self.pts_2d, dtype=np.float32)
        pts_3d_projected = self.pts_3d[:, 1:]

        H = self.compute_homography(pts_2d, pts_3d_projected)
        rospy.set_param('/pc_transform/homography', H.tolist())
        rospy.loginfo("The homography has been saved to ROS parameter server. Refer to /pc_transform/homography param.")

        example_pts_2d = np.array([355, 214])  # Example point to transform
        point = self.apply_homography(H, example_pts_2d)
        print("Example 2D point:", example_pts_2d)
        print("Computed 3D point based on the example 2D point:", point)

    def compute_homography(self, pts1, pts2):
        A = []
        for i in range(pts1.shape[0]):
            x1, y1 = pts1[i, 0], pts1[i, 1]
            x2, y2 = pts2[i, 0], pts2[i, 1]
            A.append([-x1, -y1, -1, 0, 0, 0, x1 * x2, y1 * x2, x2])
            A.append([0, 0, 0, -x1, -y1, -1, x1 * y2, y1 * y2, y2])

        A = np.array(A)
        U, S, Vh = np.linalg.svd(A)
        H = Vh[-1].reshape((3, 3))

        return H / H[-1, -1]

    @staticmethod
    def apply_homography(H, points_2d):
        homogeneous_point = np.array([points_2d[0], points_2d[1], 1.0])
        transformed_point = np.dot(H, homogeneous_point)
        return transformed_point[:2] / transformed_point[2]

    def run(self):
        rospy.sleep(2)  # Wait a bit to ensure we have the image data
        if self.image is not None:
            plt.imshow(self.image)
            plt.gcf().canvas.mpl_connect('button_press_event', self.on_click)
            plt.show()

if __name__ == '__main__':
    calibrator = Calibrator()
    calibrator.run()
