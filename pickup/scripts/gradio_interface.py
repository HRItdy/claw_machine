import rospy
import queue
import tkinter as tk
from sensor_msgs.msg import Image
from std_msgs.msg import String
from PIL import Image as PILImage, ImageTk, ImageDraw
from call_detect_service import call_segment_service
from models import SpeechTextTrans

# Global variables
latest_image = None  # For camera feed
image_queue = queue.Queue()  # Thread-safe queue for detected images
clickable_buttons = False  # For enabling/disabling buttons
click_x, click_y = None, None  # Coordinates for the point marker
transcriber = SpeechTextTrans() # Speech and text transcriber
click_ball = False # Flag to only output synthesized speech once. Without this the machine will speak several times after multiple click events.

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

# Function to update chat_text safely from the main thread
def update_chat_text(response_data):
    chat_text.config(state=tk.NORMAL)  # Enable editing
    chat_text.delete(1.0, tk.END)  # Clear previous text
    chat_text.insert(tk.END, f"Robot: {response_data}")  # Show the robot's response
    chat_text.config(state=tk.DISABLED)  # Disable editing again

# Callback function to handle messages from the '/chat_response' topic
def response_callback(response):
    # Use after() to ensure this runs in the Tkinter main thread
    root.after(0, update_chat_text, response.data)

# Callback for masked image topic (runs in ROS thread)
def masked_image_callback(ros_image):
    global click_ball
    detected_image = convert_ros_image_to_pil(ros_image)
    if detected_image:
        # Add the image to the queue for the main thread to handle
        image_queue.put(detected_image)
        if not click_ball:
            transcriber.text_to_speech('Is this the ball you want?')
            click_ball = True

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
    pro = "You have confirmed the selected ball, will proceed to grasp it."
    transcriber.text_to_speech(pro)
    speech_pub.publish(pro)

# Function for "No" button click
def on_no_click():
    global click_ball
    print("No button clicked!")
    transcriber.text_to_speech("Please restate the prompt or click on the ball you want.")
    detected_window_label.bind("<Button-1>", get_click)  # Bind click event to the detected window
    click_ball = False  # Reset the flag to allow selecting a new ball

# Function for speech-to-text transcription
def on_speech_button_click():
    text = transcriber.speech_to_text()
    print(f"Speech prompt: {text}")
    # Display the transcribed text in the speech output text box
    speech_output_text.config(state=tk.NORMAL)
    speech_output_text.delete(1.0, tk.END)  # Clear previous text
    speech_output_text.insert(tk.END, text)  # Insert the transcribed text
    speech_output_text.config(state=tk.DISABLED)  # Make it read-only
    speech_pub.publish(text)

# Initialize ROS node
rospy.init_node('camera_gui', anonymous=True)

# Subscribe to camera topic and masked image topic
rospy.Subscriber('/realsense_wrist/color/image_raw', Image, image_callback)
rospy.Subscriber('/masked_image', Image, masked_image_callback)
# Publisher for the modified image with the marker
marker_image_pub = rospy.Publisher('/masked_image_with_marker', Image, queue_size=1)
speech_pub = rospy.Publisher('/usr_input', String, queue_size=1)
response_sub = rospy.Subscriber('/chat_response', String, response_callback)

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

# Create a frame for displaying the transcribed speech
speech_output_label = tk.Label(root, text="Transcribed Speech:")
speech_output_label.pack()

# Create a text box to show the transcribed speech from the 'Speak' button
speech_output_text = tk.Text(root, height=2, width=50, state=tk.DISABLED)
speech_output_text.pack(padx=10, pady=10)

# Existing 'Speak' button to trigger transcription
speech_button = tk.Button(root, text="Speak", command=on_speech_button_click)
speech_button.pack(side=tk.LEFT, padx=10, pady=10)

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
