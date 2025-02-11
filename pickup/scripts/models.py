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
import openai
import base64
from transformers import BlipProcessor, BlipForConditionalGeneration
# GrDINO, SAM
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import build_sam, SamPredictor 
from nanoowl.owl_predictor import OwlPredictor
import azure.cognitiveservices.speech as speechsdk
# import FastSAM to use blended ultralytics
import sys
sys.path.append('/home/lab_cheem/claw_machine/src/pickup/scripts/FastSAM')
from FastSAM.fastsam import FastSAM, FastSAMPrompt

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
        mpath = os.path.join(os.path.expanduser("~"), 'claw_machine/src/pickup/scripts/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py')
        cpath = os.path.join(os.path.expanduser("~"), 'PromptCraft-Robotics/chatgpt_airsim/GroundingDINO/weights/groundingdino_swint_ogc.pth')
        self.model = build_model(SLConfig.fromfile(mpath)) #run on real
        checkpoint = torch.load(cpath, map_location=cfg.device)
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

        # plt.figure(figsize=(10, 10))
        # plt.imshow(mask_img.numpy())
        # plt.axis('off')
        # plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

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

    def inference(self, image_pil, prompt_boxes, pred_phrases, save_dir, point_coords=None, point_labels=None, save_json=False):
        image = np.array(image_pil)
        self.predictor.set_image(image)

        transformed_boxes = self.predictor.transform.apply_boxes_torch(prompt_boxes, image.shape[:2])
        masks, _, _ = self.predictor.predict_torch(
            point_coords = point_coords,
            point_labels = point_labels,
            boxes = transformed_boxes,
            multimask_output = False) # masks [n, 1, H, W], boxes_filt [n, 4]

        # plot_raw_boxes_masks(image, boxes_filt, masks, pred_phrases)
        if save_json == True:
            DetPromptedSegmentation.save_mask_json(save_dir, masks, prompt_boxes, pred_phrases)
        return masks
    
    def inference_point(self, image_pil, point_coords, point_labels, save_dir=None, save_json=False):
        image = np.array(image_pil)
        self.predictor.set_image(image)
        masks, _, _ = self.predictor.predict(
                                    point_coords=point_coords,
                                    point_labels=point_labels,
                                    multimask_output=False,
                                )
        # # plot_raw_boxes_masks(image, boxes_filt, masks, pred_phrases)
        # if save_json == True:
        #     DetPromptedSegmentation.save_mask_json(save_dir, masks, prompt_boxes, pred_phrases)
        return masks
    
    def get_image(self, image_pil, mask, mask_color=(0, 0, 255), alpha=0.5, save=False):
        # Convert the PIL image to a NumPy array
        image_np = np.array(image_pil)
        # Convert the mask to a binary mask and overlay it
        colored_mask = np.zeros_like(image_np, dtype=np.uint8)
        # Create a colored mask (blue color)
        colored_mask[mask > 0] = mask_color  # Blue color (0, 0, 255)
        # Overlay the colored mask on the image with transparency
        image_np = np.where(mask[..., None] > 0, 
                            (image_np * (1 - alpha) + colored_mask * alpha).astype(np.uint8), 
                            image_np)
        # Convert the result back to a PIL image
        masked_img = Image.fromarray(image_np)
        if save:
            masked_img.save(os.path.join(os.path.expanduser("~"), 'claw_machine/src/pickup/scripts/cache/view.png'))
        return masked_img
    
class FastSAMSegment:
    def __init__(self, model_pth = '/home/lab_cheem/claw_machine/src/pickup/scripts/FastSAM/weights/FastSAM.pt'):
        self.model = FastSAM(model_pth)
        self.device = 'cuda:0'

    def predict_prompt(self, image, prompt):
        predict_result = self.model.predict(image, device=self.device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
        prompt_process = FastSAMPrompt(image, predict_result, device=self.device)
        # text prompt
        ann = prompt_process.text_prompt(text=prompt)
        return ann

    def predict_box(self, image, boxes):
        predict_result = self.model.predict(image, device=self.device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
        prompt_process = FastSAMPrompt(image, predict_result, device=self.device)
        # bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
        # ann = prompt_process.box_prompt(bboxes=[[200, 200, 300, 300]])
        ann = prompt_process.box_prompt(bboxes=boxes)
        return ann

    def predict_point(self, image, points, pointlabel):
        predict_result = self.model.predict(image, device=self.device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
        prompt_process = FastSAMPrompt(image, predict_result, device=self.device)
        # point prompt
        # points default [[0,0]] [[x1,y1],[x2,y2]]
        # point_label default [0] [1,0] 0:background, 1:foreground
        # ann = prompt_process.point_prompt(points=[[620, 360]], pointlabel=[1])
        ann = prompt_process.point_prompt(points=points, pointlabel=pointlabel)
        return ann
    
    def get_image(self, image_pil, mask, mask_color=(0, 0, 255), alpha=0.5, save=False):
        # Convert the PIL image to a NumPy array
        image_np = np.array(image_pil)
        # Convert the mask to a binary mask and overlay it
        colored_mask = np.zeros_like(image_np, dtype=np.uint8)
        # Create a colored mask (blue color)
        colored_mask[mask > 0] = mask_color  # Blue color (0, 0, 255)
        # Overlay the colored mask on the image with transparency
        image_np = np.where(mask[..., None] > 0, 
                            (image_np * (1 - alpha) + colored_mask * alpha).astype(np.uint8), 
                            image_np)
        # Convert the result back to a PIL image
        masked_img = Image.fromarray(image_np)
        if save:
            masked_img.save(os.path.join(os.path.expanduser("~"), 'claw_machine/src/pickup/scripts/cache/view.png'))
        return masked_img
    
class OpenOWLDetection:
     # SAM
    def __init__(self):
        self.predictor =  OwlPredictor(
                            "google/owlvit-base-patch32",
                            image_encoder_engine="/home/lab_cheem/claw_machine/src/pickup/scripts/nanoowl/data/owl_image_encoder_patch32.engine"
                        )
        
    def inference(self, image_pil, prompt=["a red ball", "a purple ball"]): # prompt should be a list of descriptions
        output = self.predictor.predict(image=image_pil, text=prompt, text_encodings=None, threshold=0.1)
        return output
    
    @staticmethod
    def save_mask_json(output_dir, output, prompt=["a red ball", "a purple ball"], caption='Ball detection results'):
        json_data = {
            'caption': caption,
            'boxes':[]
        }
        # detech
        boxes = output.boxes.cpu().tolist()
        labels = output.labels.cpu().tolist()
        scores = output.scores.cpu().tolist()
        label_texts = [prompt[label] for label in labels]
        for box, label, score in zip(boxes, label_texts, scores):
            json_data['boxes'].append({
                'label': label,
                'score': float(score),
                'box': [int(x) for x in box],
            })
        with open(os.path.join(output_dir, 'boxes.json'), 'w') as f:
            json.dump(json_data, f)
        return boxes, label_texts, scores
    
    def draw_boxes(self, image, boxes, labels):
        det = [boxes, labels]
        marked_img = draw_candidate_boxes(image, det, None, save=False)
        return marked_img
    
class GPT4Reasoning: 
    def __init__(self, config = '/home/lab_cheem/claw_machine/src/pickup/scripts/config.json'):
        self.llms = ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]
        self.split = ','  
        self.max_tokens = 100
        self.temperature = 0.2
        self.prompt = [{ 'role': 'system', 'content': ''}]
        
        # Azure OpenAI Service specific configurations
        with open(config, "r") as f:
            config = json.load(f)
        
        # Set up the Azure client configuration
        openai.api_type = "azure"
        openai.api_version = config["AZURE_OPENAI_VERSION"]
        openai.api_key = config["AZURE_OPENAI_API_KEY"]
        openai.api_base = config["AZURE_OPENAI_ENDPOINT"]

    def extract_unique_nouns(self, request, model="gpt-3.5-turbo"): 
        self.prompt[0]['content'] = 'Extract the unique objects in the caption. Remove all the adjectives.' + \
                        f'List the nouns in singular form. Split them by "{self.split} ". ' + \
                        f'Caption: {request}.'

        if model in ["gpt-3.5-turbo", "gpt-4"]:
            response = openai.ChatCompletion.create(
                engine=self.deployment_name,  # Azure uses "engine" instead of "model"
                messages=self.prompt, 
                temperature=self.temperature, 
                max_tokens=self.max_tokens
            )
            reply = response.choices[0].message['content']
            unique_nouns = reply.split(':')[-1].strip()  # sometimes return with "noun: xxx, xxx, xxx"
            return unique_nouns
        elif model == "gpt-4o":
            # Additional handling for GPT-4o model if needed
            # Placeholder for handling image and json processing, if required
            pass

    def GroundedSAM_json_asPrompt(self, request, image=None, model="gpt-3.5-turbo"):
        with open(os.path.join(os.path.expanduser("~"), "claw_machine/src/pickup/scripts/cache/boxes.json"), 'r') as file:
            data = json.load(file)

        if model in ["gpt-3.5-turbo", "gpt-4"]:
            self.prompt[0]['content'] = 'Given the human request and the candidate objects, ' + \
                    'locate the target objects. the output should be a tuple (tensor(n, 4), list(n strings)),' + \
                    'following the style ( [[538, 622, 1082, 1237], [53, 62, 12, 37]], [candidate1 (0.43),  candidate2 (0.3)] ' + \
                    f'Human request: {request}. ' + \
                    f'JSON data: {data}. '
        elif model == "gpt-4o":
            if image is not None:
                img = base64.b64encode(image).decode('utf-8')
                with open(os.path.join("outputs/", request, 'label.json'), 'r') as file:
                    data = json.load(file)
                self.prompt[0]['content'] = [
                        {"type": "text", "text": 'Given the human request and the candidate objects, ' + \
                                                'locate the target object. the output should be a tuple (tensor(n, 4), list(n strings)),' + \
                                                'following the style ( [[538, 622, 1082, 1237], [53, 62, 12, 37]], [candidate1 (0.43),  candidate2 (0.3)] ' + \
                                                f'Human request: {request}. ' + \
                                                f'JSON data: {data}. '},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{img}"}
                        }
                    ]
            else:
                raise ValueError("Image input is required for model gpt-4o.")

        response = openai.ChatCompletion.create(
            model=model,
            deployment_id="gpt35turbo0125",
            messages=self.prompt, 
            temperature=self.temperature, 
            max_tokens=self.max_tokens        
        )
        reply = response.choices[0].message['content']
        return reply
    
    def parse_output(self, reply):
        boxes, objects = eval(reply)
        # Initialize two empty lists to hold object names and confidence scores
        object_names = []
        confidence_scores = []
        # Iterate over each object in the list
        for obj in objects:
            # Split the object string into name and confidence
            name, confidence = obj.rsplit(' (', 1)
            # Remove the closing parenthesis and convert the confidence to a float
            confidence = float(confidence.rstrip(')'))
            # Append the name and confidence to their respective lists
            object_names.append(name)
            confidence_scores.append(confidence)
        return boxes, object_names, confidence_scores, objects
    
    def draw_boxes(image, boxes, labels):
        det = [boxes, labels]
        marked_img = draw_candidate_boxes(image, det, None, save=False)
        return marked_img
    
class SpeechTextTrans:
    def __init__(self, config = '/home/lab_cheem/claw_machine/src/pickup/scripts/config.json'):
        # Azure OpenAI Service specific configurations
        with open(config, "r") as f:
            config = json.load(f)
        # Azure Speech Configuration
        speech_key = config["AZURE_SPEECH_API_KEY"]
        speech_region = config["AZURE_SPEECH_REGION"]
        # Initialize the Speech SDK for speech-to-text and text-to-speech
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
        audio_config = speechsdk.AudioConfig(use_default_microphone=True)
        self.speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        self.speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    def speech_to_text(self):
        print("Speak into your microphone.")
        result = self.speech_recognizer.recognize_once()
        # Check the result
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print(f"Recognized: {result.text}")
            return result.text
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("No speech could be recognized")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print(f"Speech Recognition canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print(f"Error details: {cancellation_details.error_details}")
        return ""

    def text_to_speech(self, text):
        # Synthesize spoken output
        result = self.speech_synthesizer.speak_text_async(text).get()
        # Check result
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print("Speech synthesized.")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print(f"Speech synthesis canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print(f"Error details: {cancellation_details.error_details}")
        

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

class GroundingDINOWrapper(torch.nn.Module):
    def __init__(self, model, processor, device):
        super(GroundingDINOWrapper, self).__init__()
        self.model = model
        self.processor = processor
        self.device = device
        self.tokenizer = self.model.tokenizer  # Assuming your model has a tokenizer

    def forward(self, image, caption):
        # forward pass
        outputs = self.model(image[None], captions=[caption])
        return outputs

def trt_compile():
    mpath = os.path.join(os.path.expanduser("~"), 'claw_machine/src/pickup/scripts/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py')
    cpath = os.path.join(os.path.expanduser("~"), 'PromptCraft-Robotics/chatgpt_airsim/GroundingDINO/weights/groundingdino_swint_ogc.pth')
    
    model = build_model(SLConfig.fromfile(mpath))
    checkpoint = torch.load(cpath, map_location=cfg.device)
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()

    processor = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model_wrapper = GroundingDINOWrapper(model, processor, cfg.device)

    # Prepare a dummy image (replace this with actual image size based on the model requirements)
    dummy_image_pil = Image.new('RGB', (640, 480))  # Create a dummy image of the required size

    # Preprocess the image into a tensor
    image_tensor, _ = processor(dummy_image_pil.convert("RGB"), None)
    image_tensor = image_tensor.to(cfg.device)  # Send to the correct device

    # Prepare a dummy caption
    dummy_caption = "A dummy caption."  # This is just a placeholder, replace it with an appropriate caption.

    model_wrapper = model_wrapper.to(cfg.device)

    # Export the model to ONNX format
    torch.onnx.export(
        model_wrapper,                                    # Model to export
        (image_tensor, dummy_caption),                    # Inputs: image tensor and caption string
        "model.onnx",                                     # Output file path
        export_params=True,                               # Store trained parameters
        opset_version=11,                                 # ONNX version to export with
        do_constant_folding=True,                         # Constant folding optimization
        input_names=['image_tensor', 'caption'],          # Name the inputs for ONNX
        output_names=['boxes_filt', 'scores', 'pred_phrases'],  # Name the outputs
        dynamic_axes={'image_tensor': {0: 'batch_size'},  # Allow dynamic batching for image
                      'caption': {0: 'batch_size'},       # Allow dynamic batching for caption
                      'boxes_filt': {0: 'batch_size'},    # Dynamic output
                      'scores': {0: 'batch_size'},
                      'pred_phrases': {0: 'batch_size'}}
    )
    print("Model has been successfully exported to ONNX format.")
    
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
    
    trt_compile()