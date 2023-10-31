import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

class SAM:
    def __init__(self,model_type,sam_checkpoint,device):
        self.device = device
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        self.predictor = SamPredictor(self.sam)

    def run_sam(self,image,points):
        self.sam.to(device=self.device)
        embedding = self.predictor.set_image(image)
        input_point = points['point']
        input_label = points['label']
        masks, _, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        return masks


