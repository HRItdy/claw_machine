import rospy
from sensor_msgs.msg import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Need manual input in advance
pts_3d = np.array([[-0.004, 0.116,0.298], [-0.00409, -0.0977, 0.281], [0.0006, -0.112, 0.469], [-0.001, 0.0931, 0.493]])  # left down to left up, choro

class Calibrator:
    def __init__(self):
        self.image = None
        self.pts_2d = []

        rospy.init_node('calibration', anonymous=True)
        rospy.Subscriber('/realsense_wrist/color/image_raw', Image, self.image_callback)

    def image_callback(self, msg):
        # Convert the ROS Image message to a NumPy array without using cv_bridge
        height = msg.height
        width = msg.width
        channels = 3  # Assuming RGB image

        # Convert the byte array to a NumPy array
        self.image = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, channels))

    def on_click(self, event):
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            self.pts_2d.append((x, y))
            print(f"Point {len(self.pts_2d)}: ({x}, {y})")
            plt.scatter(x, y, color='red')
            plt.draw()

            if len(self.pts_2d) == 4:
                np.savez('clicked_image_and_points.npz', image=self.image, points=self.pts_2d)
                print("Image and points saved:", self.pts_2d)
                rospy.signal_shutdown("Points selected")

    def run(self):
        rospy.sleep(2)  # Wait a bit to ensure we have the image data
        if self.image is not None:
            plt.imshow(self.image)
            cid = plt.gcf().canvas.mpl_connect('button_press_event', self.on_click)
            plt.show()
            self.pts_2d = np.asarray(self.pts_2d, dtype=np.float32)
            example_pts_2d = np.array([355, 214]) #middle ball
            # example_pts_2d = np.array([185,287]) #left down ball
            projected_pts_3d = pts_3d[:, 1:]
            H = self.compute_homography(self.pts_2d, projected_pts_3d)
            point = self.apply_homography(H, example_pts_2d)

            # find the x through Ax+By+Cz+D=0?
            # or just project back to get the 3D bounding box and do the ball detection
            print('done')

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
        
        return H / H[-1, -1]  # Normalize the matrix

    def apply_homography(self, H, points_2d):
        homogeneous_point = np.array([points_2d[0], points_2d[1], 1.0])
        transformed_point = np.dot(H, homogeneous_point)
        return transformed_point[:2] / transformed_point[2]
    
    # # Reverse the rotation to find the 3D coordinates in the original plane
    # x, y = point_projected_2d[:2]
    # z = 0  # Starting with z = 0 in the projected plane
    # point_3d_projected = np.array([x, y, z])
    # point_3d = np.dot(point_3d_projected, R.T)  # Rotate back to the original orientation
           
    def inverse_homography(self, H, points_3d):
        H_inv = np.linalg.inv(H)
        points_3d_homogeneous = np.append(points_3d, np.ones((len(points_3d), 1)), axis=1)
        points_2d_mapped_homogeneous = H_inv @ points_3d_homogeneous.T
        points_2d_mapped_homogeneous /= points_2d_mapped_homogeneous[-1, :]
        return points_2d_mapped_homogeneous.T[:, :2]


if __name__ == '__main__':
    clicker = Calibrator()
    clicker.run()
