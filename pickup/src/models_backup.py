import argparse
import os
import re
import copy
import json

import torch
import torchvision
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import gradio as gr
import copy
# ChatGPT, BLIP
# import openai
# from openai import OpenAI

# client = OpenAI(api_key='')
from transformers import BlipProcessor, BlipForConditionalGeneration


# GrDINO, SAM
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import build_sam, SamPredictor 


def draw_candidate_boxes(image, det_list, output_dir, stepstr= 'targets', save=False):
    #assert stepstr in ['candidates', 'self', 'related', 'ref'], "stepstr must be one of ['self', 'related', 'ref']"
    #image_pil = Image.open(image_path).convert("RGB") 
    image_pil = copy.deepcopy(image) # add this deepcopy, otherwise in each inference, the previous inference result will also be painted on the current frame.
    image_pil = image_pil.convert("RGB")
    draw = ImageDraw.Draw(image_pil)
    boxes, labels =  det_list[0], det_list[1]

    for box, label in zip(boxes, labels):
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        # draw rectangle
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        draw.rectangle([x0, y0, x1, y1], outline=color, width=8)

        # draw textbox+text.
        #fontPath = "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf"
        fontPath = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
        sans16  =  ImageFont.truetype ( fontPath, 8 )
        font = sans16  #font = ImageFont.load_default()
        #label_txt = label[0]+"("+str(label[1])+")"
        label_txt = label
        if hasattr(font, "getbbox"):
            txtbox = draw.textbbox((x0, y0), label_txt, font)
        else:
            w, h = draw.textsize(label_txt, font)
            txtbox = (x0, y0, w + x0, y0 + h)

        draw.rectangle(txtbox, fill=color)
        draw.text((x0, y0), label_txt, fill="white", font=font)
        #mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=8)

    if save:
        image_pil.save(os.path.join(output_dir, stepstr+".jpg")) 
    
    return image_pil

class GroundedDetection:
    # GroundingDino
    def __init__(self, cfg):
        print(f"Initializing GroundingDINO to {cfg.device}")
        # self.model = build_model(SLConfig.fromfile('src/pickup/scripts/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py')) #debug with vs code
        self.model = build_model(SLConfig.fromfile('GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'))
        checkpoint = torch.load('/home/lab_cheem/claw_machine/src/pickup/scripts/GroundingDINO/weights/groundingdino_swint_ogc.pth', map_location="cpu")
        self.model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self.model.eval()
        self.processor = T.Compose([ 
                            T.RandomResize([800], max_size=1333),
                            T.ToTensor(),
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.cfg = cfg

    def inference(self, image_pil, caption, box_threshold, text_threshold, iou_threshold):
        self.model = self.model.to(self.cfg.device)

        # input: image, caption
        #image_pil = Image.open(image_path).convert("RGB")  # load image
        image_pil = image_pil.convert("RGB")
        image, _ = self.processor(image_pil, None) 
        image = image.to(self.cfg.device)

        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."

        # output: (boxes_filt, scores, pred_phrases)
        with torch.no_grad():
            outputs = self.model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]    # num_filt, 4
        logits_filt.shape[0]

        tokenlizer = self.model.tokenizer
        tokenized = tokenlizer(caption)
        pred_phrases = []
        scores = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            scores.append(logit.max().item())

        # postprocessing: norm2raw: xywh2xyxy
        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H]) 
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2  
            boxes_filt[i][2:] += boxes_filt[i][:2]         
        boxes_filt = boxes_filt.cpu()  # norm2raw: xywh2xyxy

        print(f"Before NMS: {boxes_filt.shape[0]} boxes")
        nms_idx = torchvision.ops.nms(boxes_filt, torch.tensor(scores), iou_threshold).numpy().tolist() 
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]
        print(f"After NMS:  {boxes_filt.shape[0]} boxes")

        return boxes_filt, pred_phrases

class DetPromptedSegmentation:
    # SAM
    def __init__(self, cfg):
        self.predictor = SamPredictor(build_sam(checkpoint='/home/lab_cheem/claw_machine/src/pickup/scripts/sam_vit_h_4b8939.pth'))
        self.cfg = cfg

    @staticmethod
    def save_mask_json(output_dir, mask_list, box_list, label_list, caption=''):
        value = 0  # 0 for white background

        mask_img = torch.zeros(mask_list.shape[-2:])
        for idx, mask in enumerate(mask_list):
            mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1

        plt.figure(figsize=(10, 10))
        plt.imshow(mask_img.numpy())
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

        json_data = {
            'caption': caption,
            'mask':[{
                'value': value,
                'label': 'background'
            }]
        }
        for label, box in zip(label_list, box_list):
            value += 1
            name, logit = label.split('(')
            logit = logit[:-1] # the last is ')'
            json_data['mask'].append({
                'value': value,
                'label': name,
                'logit': float(logit),
                'box': [int(x) for x in box.numpy().tolist()],
            })
        with open(os.path.join(output_dir, 'label.json'), 'w') as f:
            json.dump(json_data, f)
        return json_data

    def inference(self, image_pil, prompt_boxes, pred_phrases, save_dir, save_json=False):
        image = np.array(image_pil)
        self.predictor.set_image(image)

        transformed_boxes = self.predictor.transform.apply_boxes_torch(prompt_boxes, image.shape[:2])
        masks, _, _ = self.predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False) # masks [n, 1, H, W], boxes_filt [n, 4]

        # plot_raw_boxes_masks(image, boxes_filt, masks, pred_phrases)
        if save_json == True:
            DetPromptedSegmentation.save_mask_json(save_dir, masks, prompt_boxes, pred_phrases)

        return masks

class GPT4Reasoning: 
    def __init__(self):
        self.llms = ["gpt-3.5-turbo", "gpt-4"]
        self.split=','  
        self.max_tokens=100 
        self.temperature=0.2
        self.prompt = [{ 'role': 'system', 'content': ''}]

    def extract_unique_nouns(self, request): 
        self.prompt[0]['content'] = 'Extract the unique objects in the caption. Remove all the adjectives.' + \
                        f'List the nouns in singular form. Split them by "{self.split} ". ' + \
                        f'Caption: {request}.'

        response = client.chat.completions.create(model=self.llms[0], messages=self.prompt, temperature=self.temperature, max_tokens=self.max_tokens)
        reply = response.choices[0].message.content
        unique_nouns = reply.split(':')[-1].strip() # sometimes return with "noun: xxx, xxx, xxx"
        return unique_nouns
    
    def GroundedSAM_json_asPrompt(self, request):
        with open(os.path.join("outputs/", request, 'label.json'), 'r') as file:
            data = json.load(file)

        self.prompt[0]['content'] = 'Given the human request and the candidate objects, ' + \
                'locate the target objects. the output should be a tuple (tensor(n, 4), list(n strings)),' + \
                'following the style ( [[538, 622, 1082, 1237], [53, 62, 12, 37]], [candidate1 (0.43),  candidate2 (0.3)] ' + \
                f'Human request: {request}. ' + \
                f'JSON data: {data}. '

        response = client.chat.completions.create(model=self.llms[1], messages=self.prompt, temperature=self.temperature, max_tokens=self.max_tokens)
        reply = response.choices[0].message.content
        return reply

def runGroundingDino(image_pil, request):
    output_dir = os.path.join("outputs/" , request)
    os.makedirs(output_dir,exist_ok=True)

    detector = GroundedDetection(cfg)
    results = detector.inference(image_pil, request, cfg.box_threshold, cfg.text_threshold, cfg.iou_threshold)
    # results: tensor[n, 4], list[n]
    dino_pil = draw_candidate_boxes(image_pil, results, output_dir, stepstr='nouns', save=True)
    #image_pil.show()

    segmenter = DetPromptedSegmentation(cfg)
    mask = segmenter.inference(image_pil, results[0], results[1], output_dir, save_json=True)
    # mask_pil = Image.open(os.path.join(output_dir, 'mask.jpg')).convert("RGB") 
    return dino_pil, mask

def LLMsforRef(image_pil, request):
    reply = GPT4Reasoning().GroundedSAM_json_asPrompt(request)
    import ast
    targets = ast.literal_eval(reply)
    results = (np.array(targets[0]), targets[1]) # (array([[ 538,  622, 1082, 1237]]), [['the cup (0.39)']])

    output_dir = os.path.join("outputs/" , request)
    ref_pil = draw_candidate_boxes(image_pil, results, output_dir, stepstr='llmRef', save=True)
    return ref_pil

# def _trt_compile(): #TODO
    

if __name__ == "__main__":

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

    parser = argparse.ArgumentParser("GroundingDINO", add_help=True)
    parser.add_argument("--debug", action="store_true", help="using debug mode")
    parser.add_argument("--share", action="store_true", help="share the app")
    parser.add_argument("--box_threshold", type=float, default=0.25, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.2, help="text threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold")
    parser.add_argument("--visualize", default=False, help="visualize intermediate data mode")
    parser.add_argument("--device", type=str, default="cuda:0", help="run on: 'cuda:0' for GPU 0 or 'cpu' for CPU. Default GPU 0.")
    cfg = parser.parse_args()    
    
    '''
    input_image = "image/table.jpg"
    request = "give me the cup on the left side of the pot"
    image_pil = Image.open(input_image).convert("RGB") 
    LLMsforRef(image_pil, request) 
    '''

    css = """
    #mkd {
        height: 500px; 
        overflow: auto; 
        border: 1px solid #ccc; 
    }
    """
    
    block = gr.Blocks(css=css).queue()
    with block:
        gr.Markdown("<h1><center> Referring in Robotics: VLMs chained by LLMs. <h1><center>")
        
        with gr.Row():
            with gr.Column(scale=0.5):
                input_image = gr.Image(sources=['upload'], type="pil", label="image")
                request = gr.Textbox(label="User request", placeholder="Give me the cup on the left side of the pot.")
                run_button = gr.Button(value="Run GroundingDino")
                run_button2 = gr.Button(value="Run LLMsRef")
                with gr.Accordion("Advanced options", open=False):
                    box_threshold = gr.Slider(
                        label="Box Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.001)
                    text_threshold = gr.Slider(
                        label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.001)

            with gr.Column():
                #gallery = gr.Image(type="pil", label="GroundingDINO")
                gallery = gr.Gallery(label="GroundingDINO-SAM", show_label=True, elem_id="gallery", 
                                     columns=[1], rows=[2], object_fit="contain", height="auto")
                

            with gr.Column():
                gallery2 = gr.Image(type="pil", label="LLMsRef")
    


        run_button.click(fn=runGroundingDino, inputs=[input_image, request], outputs=[gallery])
        run_button2.click(fn=LLMsforRef, inputs=[input_image, request], outputs=[gallery2])
        
        
        '''
        gr.Examples(
          [["image/table.jpg", "coffee cup", 0.25, 0.25]],
          inputs = [input_image, request],
          outputs = [gallery],
          fn=runGroundingDino,
          cache_examples=True,
          label='Try this example input!'
        )
        '''

    block.launch(share=True, show_api=False, show_error=True)
