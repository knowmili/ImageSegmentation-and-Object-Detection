"""
Implementation of Yolo Loss Function from the original YOLOv1 paper

"""

import torch
from torch import nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self,S=7,B=2,C=20):
        super(YoloLoss,self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        """
        S is split size or grid size of image (in paper 7),
        B is number of bounding boxes (in paper 2),
        C is number of classes (in paper and VOC dataset is 20),
        """
        self.S = S
        self.B = B
        self.C = C
        # As per YOLOv1 Paper
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self,predictions,target):
        # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
        predictions = predictions.reshape(-1,self.S,self.S,self.C+self.B*5)

        # Calculate IoU for the two predicted bounding boxes with target bbox

        # For Bounding Box 1
        # (N, S, S, 30) for ... in Tensors
        iou_b1 = intersection_over_union(predictions[...,21:25],target[...,21:25])
        # For Bounding Box 2 
        iou_b2 = intersection_over_union(predictions[...,26:30],target[...,21:25])
        ious = torch.cat([iou_b1.unsqueeze(0),iou_b2.unsqueeze(0)],dim=0)
        
        # Select the Bounding Box with the highest IOU
        # Take the box with highest IoU out of the two prediction
        # Note that bestbox will be indices of 0, 1 for which bbox was best
        iou_maxes,bestbox = torch.max(ious,dim=0) # the result tuple of two output tensors (max values, max_indices(0 or 1))
        exists_box = target[...,20].unsqueeze(3) # Iobj_i 
        # We can also use target[...,20:21] instead of unsqueeze(3) to get the same result and preserve the shape of the tensor
        
        # ======================== #
        #  FOR BOX COORDINATES     #
        # ======================== #
        
        # Set boxes with no object in them to 0. We only take out one of the two 
        # predictions, which is the one with highest Iou calculated previously.
        box_predictions = exists_box * (
            (
                bestbox * predictions[...,26:30] 
                + (1-bestbox) * predictions[...,21:25]
             )
        )    

        box_targets = exists_box * target[...,21:25]

        # Take Absolute value of the predictions because due to models random 
        # initialization the predictions might be negative and then take the square root
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6) # Added Small Value for Numerical Stability
        )

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # (N,S,S,4) -> (N*S*S,4) as we need to calculate loss for each grid cell and for each grid bounding box
        box_loss = self.mse(
            torch.flatten(box_predictions,end_dim=-2), # Flatten the Tensor from 0 to second last dimension 
            torch.flatten(box_targets,end_dim=-2)
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #
        
        pred_box = (
            bestbox * predictions[...,25:26] + (1-bestbox) * predictions[...,20:21]
        )

        # N*S*S
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[...,20:21])
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        # (N,S,S,1) -> (N,S*S)
        no_object_loss = self.mse(
            torch.flatten((1-exists_box) * predictions[...,20:21],start_dim=1),
            torch.flatten((1-exists_box) * target[...,20:21],start_dim=1)
        )

        no_object_loss += self.mse(
            torch.flatten((1-exists_box) * predictions[...,25:26],start_dim=1),
            torch.flatten((1-exists_box) * target[...,20:21],start_dim=1)
        )
        
        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        # (N,S,S,20) -> (N*S*S,20) - We are viewing each cell as separate example
        class_loss = self.mse(
            torch.flatten(exists_box* predictions[...,:20],end_dim=-2),
            torch.flatten(exists_box* target[...,:20],end_dim=-2)
        )

        # TOTAL LOSS
        loss = (
            self.lambda_coord * box_loss # Coordination Loss
            + object_loss # Object Loss
            + self.lambda_noobj * no_object_loss # No Object Loss
            + class_loss # Class Loss
        )

        return loss

