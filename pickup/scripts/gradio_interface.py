import rospy
import queue
import tkinter as tk
from sensor_msgs.msg import Image
from PIL import Image as PILImage, ImageTk, ImageDraw
from call_detect_service import call_segment_service

# Global variables
latest_image = None  # For camera feed
image_queue = queue.Queue()  # Thread-safe queue for detected images
clickable_buttons = False  # For enabling/disabling buttons
click_x, click_y = None, None  # Coordinates for the point marker

# Publisher for the modified image with the marker
marker_image_pub = rospy.Publisher('/masked_image_with_marker', Image, queue_size=1)

# Function to safely update the camera image in Tkinter
def update_image():
    global latest_image
    if latest_image:
        resized_image = latest_image.resize((640, 480))  # Resize to fixed size
        imgtk = ImageTk.PhotoImage(image=resized_image)
        camera_label.imgtk = imgtk
        camera_label.config(image=imgtk)
    root.after(10, update_image)

# Update the detected window safely in the main thread
def update_detected_window():
    global clickable_buttons, click_x, click_y
    try:
        # Check if there's a new image in the queue
        if not image_queue.empty():
            detected_image = image_queue.get_nowait()

            # # If a click was registered, draw a red marker on the image
            # if click_x is not None and click_y is not None:
            #     detected_image = detected_image.copy()
            #     draw = ImageDraw.Draw(detected_image)
            #     radius = 5
            #     draw.ellipse((click_x - radius, click_y - radius, click_x + radius, click_y + radius), fill='red')

            #     # Publish the image with the marker
            #     publish_image_with_marker(detected_image)

            # Resize image to 640x480 for Tkinter display
            resized_image = detected_image.resize((640, 480))
            imgtk = ImageTk.PhotoImage(image=resized_image)
            detected_window_label.imgtk = imgtk
            detected_window_label.config(image=imgtk)

            # Enable buttons and update chat message
            clickable_buttons = True
            chat_text.config(state=tk.NORMAL)
            chat_text.delete(1.0, tk.END)
            chat_text.insert(tk.END, "Is this the ball you want?")
            chat_text.config(state=tk.DISABLED)
            yes_button.config(state=tk.NORMAL)
            no_button.config(state=tk.NORMAL)
    except queue.Empty:
        pass

    # Schedule this function to be called again after 100ms
    root.after(100, update_detected_window)

# Callback for camera feed topic (runs in ROS thread)
def image_callback(ros_image):
    global latest_image
    latest_image = convert_ros_image_to_pil(ros_image)

# Callback for masked image topic (runs in ROS thread)
def masked_image_callback(ros_image):
    detected_image = convert_ros_image_to_pil(ros_image)
    if detected_image:
        # Add the image to the queue for the main thread to handle
        image_queue.put(detected_image)

# Helper function to convert ROS Image to PIL Image
def convert_ros_image_to_pil(ros_image):
    try:
        width, height = ros_image.width, ros_image.height
        image_data = bytes(ros_image.data)

        # Create PIL image from raw byte data
        pil_image = PILImage.frombytes('RGB', (width, height), image_data)
        return pil_image
    except Exception as e:
        rospy.logerr(f"Error processing image: {e}")
        return None

# Function to convert a PIL Image back to ROS Image message
def convert_pil_to_ros_image(pil_image, ros_image):
    try:
        ros_image.header.stamp = rospy.Time.now()
        ros_image.height, ros_image.width = pil_image.size[1], pil_image.size[0]
        ros_image.encoding = "rgb8"
        ros_image.is_bigendian = False
        ros_image.step = ros_image.width * 3
        ros_image.data = pil_image.tobytes()
        return ros_image
    except Exception as e:
        rospy.logerr(f"Error converting PIL to ROS Image: {e}")
        return None

# Function to handle mouse clicks and get coordinates in detected window
def get_click(event):
    global click_x, click_y
    if clickable_buttons:
        click_x, click_y = event.x, event.y
        print(f"Clicked at: {click_x}, {click_y}")
        # Redraw the detected window with a marker
        update_detected_window()
        # Call the object detection function with the click coordinates
        call_segment_service(click_x, click_y)

# Function to publish the modified image with the marker
def publish_image_with_marker(pil_image):
    try:
        ros_image = Image()
        # Convert the PIL image to a ROS Image message
        ros_image = convert_pil_to_ros_image(pil_image, ros_image)

        # Publish the image with the marker
        marker_image_pub.publish(ros_image)
    except Exception as e:
        rospy.logerr(f"Error publishing image with marker: {e}")

# Function for "Yes" button click
def on_yes_click():
    print("Yes button clicked!")
    call_gpt()  # Placeholder for the GPT function

# Function for "No" button click
def on_no_click():
    print("No button clicked!")
    detected_window_label.bind("<Button-1>", get_click)  # Bind click event to the detected window

# Placeholder function for GPT processing
def call_gpt():
    print("Processing with GPT...")

# Initialize ROS node
rospy.init_node('camera_gui', anonymous=True)

# Subscribe to camera topic and masked image topic
rospy.Subscriber('/realsense_wrist/color/image_raw', Image, image_callback)
rospy.Subscriber('/masked_image', Image, masked_image_callback)

# Create the main Tkinter window
root = tk.Tk()
root.title("Speech Interaction Robot")

# Create a frame to hold both the camera feed and detected image windows horizontally
image_frame = tk.Frame(root)
image_frame.pack()

# Create a label to display the camera feed
camera_label = tk.Label(image_frame, bg="gray")
camera_label.pack(side=tk.LEFT, padx=10, pady=10)

# Create a "Detected Window" for displaying the masked image, with a blank placeholder initially
detected_window_label = tk.Label(image_frame, bg="gray")
detected_window_label.pack(side=tk.LEFT, padx=10, pady=10)

# Fix the window size for both labels to 640x480
camera_label.config(width=576, height=432)
detected_window_label.config(width=576, height=432)

# Create a chat text box to show robot's messages
chat_label = tk.Label(root, text="Chat from Robot:")
chat_label.pack()
chat_text = tk.Text(root, height=2, width=30, state=tk.DISABLED)
chat_text.pack()

# Add "Yes" and "No" buttons, initially disabled
yes_button = tk.Button(root, text="Yes", command=on_yes_click, state=tk.DISABLED)
no_button = tk.Button(root, text="No", command=on_no_click, state=tk.DISABLED)
yes_button.pack(side=tk.LEFT, padx=10, pady=10)
no_button.pack(side=tk.LEFT, padx=10, pady=10)

# Start the Tkinter image update loop
root.after(10, update_image)

# Start the detected image update loop
root.after(100, update_detected_window)

# Run Tkinter main loop and ROS spin in parallel
def tk_loop():
    while not rospy.is_shutdown():
        root.update_idletasks()
        root.update()

tk_loop()
