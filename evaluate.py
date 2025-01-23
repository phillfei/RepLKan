import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from torch import nn 

class IoU_binary(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU_binary, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # inputs = torch.sigmoid(inputs)
                     
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth) / (union + smooth)
                
        return IoU

class IoU_multiple(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU_multiple, self).__init__()
        self.sfx = nn.Softmax(dim=1)
        self.binary = IoU_binary()
    def forward(self, inputs, targets, num_classes, smooth=1):
        sum_iou = 0.0
        inputs = torch.argmax(self.sfx(inputs), dim=1).squeeze(0)
        pred = inputs.view(-1)
        target = targets.view(-1)

        # Mask to ignore classes out of range (useful for edge cases)
        valid_mask = (target >= 0) & (target < num_classes) & (pred >= 0) & (pred < num_classes)
    
        # Filter out invalid entries
        pred = pred[valid_mask]
        target = target[valid_mask]
    
        # Calculate the confusion matrix using a more efficient method
        indices = target * num_classes + pred
        conf_matrix = torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)
    
        # Extract true positives, false positives, and false negatives for each class
        tp = torch.diag(conf_matrix)  # True Positives
        fp = conf_matrix.sum(dim=0) - tp  # False Positives
        fn = conf_matrix.sum(dim=1) - tp  # False Negatives
        tp = tp.float()
        fp = fp.float()
        fn = fn.float()
    
        # Compute IoU for each class
        union = tp + fp + fn
        iou_list = torch.where(union == 0, torch.full_like(union, float('nan')), tp / union)
    
        # Calculate mean IoU (excluding NaN)
        miou = torch.nanmean(iou_list)
        return miou

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    ious = 0
    iou_calculate = IoU_multiple()
    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch[0], batch[1]
            # print(image.shape)
            
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                # ious += iou_calculate(mask_pred,mask_true,net.n_classes)
                # ious = calculate_iou(mask_pred.argmax(dim=1)[0].long().squeeze().cpu().numpy(), mask_true.cpu().numpy(), net.n_classes)
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
 
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)
