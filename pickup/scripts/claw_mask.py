import rospy
from sensor_msgs.msg import Image
from tkinter import Tk, Label
from PIL import Image as PILImage, ImageTk
import numpy as np

class CameraViewer:
    def __init__(self, topic):
        # Initialize ROS node
        rospy.init_node("camera_viewer")
        self.topic = topic
        self.image_data = None

        # Initialize Tkinter GUI
        self.root = Tk()
        self.label = Label(self.root)
        self.label.pack()
        self.label.bind("<Button-1>", self.on_click)

        # Subscribe to the image topic
        rospy.Subscriber(self.topic, Image, self.image_callback)

    def image_callback(self, msg):
        # Convert ROS Image message to numpy array
        self.image_data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        
        # Convert to Tkinter-compatible format and update the label
        image_pil = PILImage.fromarray(self.image_data)
        photo = ImageTk.PhotoImage(image=image_pil)
        self.label.configure(image=photo)
        self.label.image = photo

    def on_click(self, event):
        # Get the clicked pixel's color
        x, y = event.x, event.y
        if self.image_data is not None:
            color = self.image_data[y, x]
            rgba_list = [int(color[0]), int(color[1]), int(color[2]), 255]  # RGB + Alpha (as regular integers)

            # Send the RGBA list to the ROS parameter server
            rospy.set_param('/selected_mask_color', rgba_list)
            rospy.loginfo(f"Color {rgba_list} at pixel ({x}, {y}) sent to parameter server.")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    viewer = CameraViewer("/realsense_wrist/color/image_raw")
    viewer.run()
