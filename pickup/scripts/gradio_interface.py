import subprocess
import threading
import os
# import queue
import gradio as gr
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import threading
from PIL import Image as Img

usrp = os.path.expanduser("~")
# output_queue = queue.Queue()
# output_log = []  # List to store all output lines
bridge = CvBridge()
# Global variable to store the masked image
masked_image = None
image_size = (640, 480)

# ROS callback function to receive the processed image
def image_callback(msg):
    global masked_image
    # Convert ROS Image message to a numpy array
    masked_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    pil_image = Img.fromarray(masked_image, 'RGB')
    # Resize the image to half its size
    pil_image = pil_image.resize(image_size)
    # Convert the PIL image back to a numpy array for Gradio
    masked_image = np.array(pil_image)
       
# Initialize the ROS node and subscriber in the Gradio app
def ros_listener():
    rospy.Subscriber('/masked_image', Image, image_callback)
    rospy.spin()

def launch_ros_services():
    # Call the shell script to start ROS services
    shell_script_path = "/home/lab_cheem/claw_machine/src/pickup/launch/claw_machine_gradio.sh"
    process = subprocess.Popen(['bash', shell_script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Capture the output (optional)
    # threading.Thread(target=enqueue_output, args=(process.stdout, output_queue)).start()
    # threading.Thread(target=enqueue_output, args=(process.stderr, output_queue)).start()
    return "ROS Services started"

# def enqueue_output(pipe, q):
#     for line in iter(pipe.readline, ''):
#         q.put(line)
#     pipe.close()

# def store_and_get_ros_output():
#     global output_log
#     while not output_queue.empty():
#         line = output_queue.get()
#         output_log.append(line)
#     return "".join(output_log)  # Combine all lines in the list into a single string

def stop_ros_services():
    shell_script_path = "/home/lab_cheem/claw_machine/src/pickup/launch/shutdown_claw.sh"
    process = subprocess.Popen(['bash', shell_script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return "ROS Services stopped"

# === Gradio Interface ===
with gr.Blocks() as demo:
    gr.Markdown("# Humanoid Robot Control Interface")
    
    with gr.Row():
        start_services_btn = gr.Button("Start ROS Services")
        shutdown_btn = gr.Button("Shutdown ROS Services")
    
    start_services_btn.click(fn=launch_ros_services, inputs=None, outputs=None)
    shutdown_btn.click(fn=stop_ros_services, inputs=None, outputs=None)
    
    # # Scrollable output window
    # ros_output_textbox = gr.Textbox(label="ROS Output", lines=10, interactive=False, elem_id="scrollable-output")
    # refresh_btn = gr.Button("Refresh ROS Output")
    # refresh_btn.click(fn=store_and_get_ros_output, inputs=None, outputs=ros_output_textbox)
    # demo.load(fn=store_and_get_ros_output, outputs=ros_output_textbox, every=2.0)
    
    gr.Markdown("## Detection Result Image")
    detection_img = gr.Image(label="Masked Image", elem_id="mask-result-image", width=image_size[0], height=image_size[1], type='numpy')
    # Periodically check if a new image is received
    def update_image():
        return masked_image if masked_image is not None else None
    demo.load(fn=update_image, outputs=detection_img, every=1.0)

if __name__ == "__main__":
    # Start the ROS listener in a separate thread
    rospy.init_node('gradio_listener', anonymous=True)
    ros_thread = threading.Thread(target=ros_listener, daemon=True)
    ros_thread.start()

    demo.launch()