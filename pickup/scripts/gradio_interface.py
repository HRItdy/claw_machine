import gradio as gr
import subprocess
import threading
import queue
import soundfile as sf
import numpy as np
import openai
import azure.cognitiveservices.speech as speechsdk
from textblob import TextBlob  # For sentiment analysis
import cv2
import json

# === Azure OpenAI and Speech SDK Setup ===
with open("config.json", "r") as f:
    config = json.load(f)

print("Initializing ChatGPT...")
openai.api_type = "azure"
openai.api_version = config["AZURE_OPENAI_VERSION"]
openai.api_key = config["AZURE_OPENAI_API_KEY"]
openai.api_base = config["AZURE_OPENAI_ENDPOINT"]
endpoint = "https://YOUR_OPENAI_ENDPOINT"
deployment_id = "YOUR_DEPLOYMENT_ID"  # Azure GPT-4 deployment

speech_key = "YOUR_AZURE_SPEECH_KEY"
service_region = "YOUR_SERVICE_REGION"
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

# === ROS Service Management ===
def launch_ros_services():
    script_path = "../launch/claw_machine.sh"
    process = subprocess.Popen(
        ["bash", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return process

def stop_ros_services(process):
    process.terminate()

def enqueue_output(pipe, q):
    for line in iter(pipe.readline, ''):
        q.put(line)
    pipe.close()

# === Speech-to-Speech Function with Tone Generation ===
def generate_random_text(prompt, max_length=50, temperature=0.8):
    response = openai.Completion.create(
        model="gpt-3.5-turbo",
        deployment_id="gpt35turbo0125",
        messages=chat_history,
        temperature=0
    )
    return response.choices[0].text.strip()

def analyze_tone(text):
    analysis = TextBlob(text)
    sentiment = analysis.sentiment.polarity
    if sentiment > 0.5:
        tone = "excited"
    elif sentiment < -0.5:
        tone = "confused"
    else:
        tone = "neutral"
    return tone

def apply_tone_to_text(text, tone):
    if tone == "excited":
        return f"Wow! {text} That's really something!"
    elif tone == "confused":
        return f"{text} I cannot find the desired target."
    else:
        return f"{text} Okay, let's move forward."

def text_to_speech(input_text):
    # Generate random content
    random_text = generate_random_text(input_text)
    
    # Analyze tone
    tone = analyze_tone(random_text)
    
    # Apply tone to the response
    toned_text = apply_tone_to_text(random_text, tone)
    
    # Use Azure Speech SDK to synthesize the speech
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    result = synthesizer.speak_text_async(toned_text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized for text [{}]".format(toned_text))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print("Error details: {}".format(cancellation_details.error_details))

    return toned_text  # Return the toned text for display in Gradio

# === SAM Detection Function ===
def sam_detect(image, click):
    # Implement SAM detection logic here
    # Placeholder implementation
    detected_image = image.copy()
    bounding_box = None  # Replace with actual bounding box from SAM

    # Example: Draw a dummy bounding box
    if click is not None:
        x, y = int(click['x']), int(click['y'])
        cv2.rectangle(detected_image, (x-50, y-50), (x+50, y+50), (0, 255, 0), 2)
        bounding_box = {'x_min': x-50, 'y_min': y-50, 'x_max': x+50, 'y_max': y+50}

    return detected_image, bounding_box

def detect_with_sam(image, click):
    if click is None:
        return image, "Please click on a point to detect an object."
    
    detected_image, bounding_box = sam_detect(image, click)
    
    if bounding_box:
        return detected_image, "Object detected."
    else:
        return image, "No object detected."

# === Launch ROS Services and Capture Output ===
ros_process = launch_ros_services()
output_queue = queue.Queue()

# Start threads to capture stdout and stderr
threading.Thread(target=enqueue_output, args=(ros_process.stdout, output_queue), daemon=True).start()
threading.Thread(target=enqueue_output, args=(ros_process.stderr, output_queue), daemon=True).start()

def get_ros_output():
    lines = []
    while not output_queue.empty():
        lines.append(output_queue.get())
    return "\n".join(lines)

# === Gradio Interface ===
with gr.Blocks() as demo:
    gr.Markdown("# Humanoid Robot Control Interface")
    
    with gr.Tab("Voice Control"):
        gr.Markdown("## Enter a Prompt to Generate Tone-Aware Speech")
        with gr.Row():
            input_text = gr.Textbox(lines=2, placeholder="Enter your prompt here...", label="Input Text")
            response_text = gr.Textbox(label="Generated Response with Tone")
        input_text.submit(fn=text_to_speech, inputs=input_text, outputs=response_text)
    
    with gr.Tab("Image Interaction"):
        gr.Markdown("## Click on the Image to Detect Objects")
        with gr.Row():
            image_input = gr.Image(type="numpy", label="Upload Image")
            image_output = gr.Image(label="Output Image")
            feedback = gr.Textbox(label="Feedback")
        image_output.click(fn=detect_with_sam, inputs=[image_input, gr.ClickData()], outputs=[image_output, feedback])
    
    with gr.Tab("ROS Service Output"):
        gr.Markdown("## Real-Time ROS Services Output")
        ros_output_textbox = gr.Textbox(label="ROS Output", lines=20, interactive=False)
        refresh_btn = gr.Button("Refresh ROS Output")
        refresh_btn.click(fn=get_ros_output, inputs=None, outputs=ros_output_textbox)
        # Auto-refresh every 2 seconds
        demo.load(fn=get_ros_output, outputs=ros_output_textbox, every=2.0)
    
    with gr.Tab("Shutdown"):
        gr.Markdown("## Gracefully Shutdown ROS Services")
        shutdown_btn = gr.Button("Shutdown ROS Services")
        shutdown_btn.click(fn=lambda: stop_ros_services(ros_process), inputs=None, outputs=None)

demo.launch()
