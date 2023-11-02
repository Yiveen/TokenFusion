import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

class SAM:
    def __init__(self,model_type,sam_checkpoint):
        # self.device = device
        self.device = torch.device("cuda:1")
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=self.device)
        self.sam = torch.nn.DataParallel(self.sam, device_ids=[1])
        self.predictor = SamPredictor(self.sam.module)
        # self.predictor = torch.nn.DataParallel(self.predictor, device_ids=[1])

    def run_sam(self,image,points):
        print('images',image.shape)
        self.predictor.set_image(image)
        input_point = points['point']
        input_label = points['label']
        # masks, _, _ = self.predictor.predict(
        #     point_coords=input_point,
        #     point_labels=input_label,
        #     multimask_output=False,
        # )
        masks, _, _ = self.predictor.predict_torch(  # 这个函数也不一样  # masks(BxCxHxW)
            point_coords=input_point,
            point_labels=input_label,
            boxes=None,  # 形式 [[x1,y1,x2,y2], [x1,y1,x2,y2],...] (n,4)
            multimask_output=False,)
        return masks


