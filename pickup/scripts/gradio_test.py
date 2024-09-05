import gradio as gr
import numpy as np

def on_image_click(image, evt: gr.SelectData):
    # Extract x and y coordinates from the click event
    x, y = evt.index
    print(f"Clicked coordinates: x={x}, y={y}")
    return f"Clicked coordinates: x={x}, y={y}"

with gr.Blocks() as demo:
    image_input = gr.Image(label="Click on the image")
    click_output = gr.Textbox(label="Clicked Coordinates")
    
    image_input.select(
        on_image_click,
        inputs=[image_input],
        outputs=[click_output]
    )

demo.launch()