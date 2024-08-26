import rospy
from sensor_msgs.msg import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import json

class Calibrator:
    def __init__(self):
        self.image = None
        self.pts_2d = []
        self.pts_3d = []
        self.calibration_file = "calibration_data.json"

        rospy.init_node('calibration', anonymous=True)
        rospy.Subscriber('/realsense_wrist/color/image_raw', Image, self.image_callback)

        # Check if calibration data exists
        if os.path.exists(self.calibration_file):
            self.load_calibration()
        else:
            rospy.loginfo("No existing calibration data found. Starting new calibration.")

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
        self.calculate_transformation_matrix()

    def calculate_transformation_matrix(self):
        assert len(self.pts_2d) == len(self.pts_3d), "Number of 2D and 3D points must be the same"
        assert len(self.pts_2d) >= 4, "At least 4 point correspondences are required"

        A = []
        b = []
        for (x, y), (X, Y, Z) in zip(self.pts_2d, self.pts_3d):
            A.append([x, y, 1, 0, 0, 0, -X*x, -X*y])
            A.append([0, 0, 0, x, y, 1, -Y*x, -Y*y])
            b.append(X)
            b.append(Y)

        A = np.array(A)
        b = np.array(b)

        h = np.linalg.lstsq(A, b, rcond=None)[0]

        H = np.array([
            [h[0], h[1], h[2]],
            [h[3], h[4], h[5]],
            [h[6], h[7], 1]
        ])

        rospy.set_param('/calibration/H', H.tolist())
        rospy.loginfo("Stored H successfully!")

        # Save calibration data to local file
        self.save_calibration(H)

        example_pts_2d = np.array([355, 214])
        point = self.transform_2d_to_3d(example_pts_2d, H)
        print("Example 2D point:", example_pts_2d)
        print("Computed 3D point based on the example 2D point:", point)

    def save_calibration(self, H):
        data = {
            "points_2d": self.pts_2d,
            "points_3d": self.pts_3d.tolist(),
            "H": H.tolist()
        }
        with open(self.calibration_file, 'w') as f:
            json.dump(data, f)
        rospy.loginfo(f"Calibration data saved to {self.calibration_file}")

    def load_calibration(self):
        with open(self.calibration_file, 'r') as f:
            data = json.load(f)
        self.pts_2d = data['points_2d']
        self.pts_3d = np.array(data['points_3d'])
        H = np.array(data['H'])

        rospy.set_param('/calibration/points_2d', self.pts_2d)
        rospy.set_param('/calibration/points_3d', self.pts_3d.tolist())
        rospy.set_param('/calibration/H', H.tolist())
        rospy.loginfo("Calibration data loaded from file and stored in ROS parameter server.")

    @staticmethod
    def transform_2d_to_3d(point_2d, H):
        x, y = point_2d
        p_2d = np.array([x, y, 1])
        p_3d = np.dot(H, p_2d)
        X, Y, Z = p_3d / p_3d[2]
        return X, Y, Z

    def run(self):
        rospy.sleep(2)
        if self.image is not None and not os.path.exists(self.calibration_file):
            plt.imshow(self.image)
            plt.gcf().canvas.mpl_connect('button_press_event', self.on_click)
            plt.show()

if __name__ == '__main__':
    calibrator = Calibrator()
    calibrator.run()
