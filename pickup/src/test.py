#!/usr/bin/env python3
# main.py

import cv2
import numpy as np
from models import runGroundingDino, GroundedDetection, DetPromptedSegmentation, draw_candidate_boxes
import argparse

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

from PIL import Image

parser = argparse.ArgumentParser("GroundingDINO", add_help=True)
parser.add_argument("--debug", action="store_true", help="using debug mode")
parser.add_argument("--share", action="store_true", help="share the app")
parser.add_argument("--box_threshold", type=float, default=0.25, help="box threshold")
parser.add_argument("--text_threshold", type=float, default=0.2, help="text threshold")
parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold")
parser.add_argument("--visualize", default=False, help="visualize intermediate data mode")
#parser.add_argument("--device", type=str, default="cuda:0", help="run on: 'cuda:0' for GPU 0 or 'cpu' for CPU. Default GPU 0.")
parser.add_argument("--device", type=str, default="cpu", help="run on: 'cuda:0' for GPU 0 or 'cpu' for CPU. Default GPU 0.")
cfg = parser.parse_args()    

class Test:
    def __init__(self, image):
        # Initialize components
        self.detector = GroundedDetection(cfg)
        self.segmenter = DetPromptedSegmentation(cfg)
        self.latest_image = image

    def process_image(self, user_request):
        # Use GroundingDINO to find the ball
        output_dir = os.path.join("outputs/" , user_request)
        os.makedirs(output_dir,exist_ok=True)

        results = self.detector.inference(self.latest_image, user_request, cfg.box_threshold, cfg.text_threshold, cfg.iou_threshold)
        # print(type(results))
        # results: tensor[n, 4], list[n]
        # only save the first element and ignore the others
        results_ = []
        results_.append(results[0][0].unsqueeze(0))
        results_.append([results[1][0]])
        
        dino_pil = draw_candidate_boxes(self.latest_image, results_, output_dir, stepstr='nouns', save=True)
        #image_pil.show()

        mask = self.segmenter.inference(self.latest_image, results_[0], results_[1], output_dir, save_json=True)
        # boxes, phrases = self.grounding_dino.detect(self.latest_image, user_request)

if __name__ == "__main__":
    #load image 
    im = Image.open("/home/lab_cheem/claw_machine/GroundingDINO/test.jpg")  
    im.show()
    test = Test(im)
    test.process_image('detect the upper football')