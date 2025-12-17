import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from train_util import *

def compute_semantic_pos_loss(prob_in, labxy_feat,  pos_weight = 0.003,  kernel_size=16):
    # this wrt the slic paper who used sqrt of (mse)

    # rgbxy1_feat: B*50+2*H*W
    # output : B*9*H*w
    # NOTE: this loss is only designed for one level structure

    # todo: currently we assume the downsize scale in x,y direction are always same
    S = kernel_size
    m = pos_weight
    prob = prob_in.clone()

    b, c, h, w = labxy_feat.shape
    pooled_labxy = poolfeat(labxy_feat, prob, kernel_size, kernel_size)
    reconstr_feat = upfeat(pooled_labxy, prob, kernel_size, kernel_size)

    loss_map = reconstr_feat[:,-2:,:,:] - labxy_feat[:,-2:,:,:]

    # self def cross entropy  -- the official one combined softmax
    logit = torch.log(reconstr_feat[:, :-2, :, :] + 1e-8)
    loss_sem = - torch.sum(logit * labxy_feat[:, :-2, :, :]) / b
    loss_pos = torch.norm(loss_map, p=2, dim=1).sum() / b * m / S

    # empirically we find timing 0.005 tend to better performance
    loss_sum =  0.005 * (loss_sem + loss_pos)
    loss_sem_sum =  0.005 * loss_sem
    loss_pos_sum = 0.005 * loss_pos

    return loss_sum, loss_sem_sum,  loss_pos_sum


def cross_entropy_loss_edge(prediction, label): 
    label = label.long()
    mask = label.float()
    
    num_positive = torch.sum((mask==1).float()).float()
    num_negative = torch.sum((mask==0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0
    cost = nn.BCEWithLogitsLoss(weight=mask)(prediction.float(),label.float())
#    print(cost.shape)
    return cost * 100.

def focal_cross_entropy_loss_edge(prediction, label):
    gamma = 2.0 #1
    prob_fl = 0.85 #0.75-0.95
    prob_wce = 1 - prob_fl

    label = label.long()
    mask = label.float()

    num_positive = torch.sum((mask == 1).float()).float()
    num_negative = torch.sum((mask == 0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0
    # cost1 = nn.BCEWithLogitsLoss(weight=mask)(prediction.float(), label.float())
    # bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(prediction.float(), torch.clamp_max(label.float(), 1))
    bce_loss = F.binary_cross_entropy_with_logits(prediction.float(), torch.clamp_max(label.float(), 1), reduction='none')
    # cost2 = (mask * bce_loss).mean()
    with torch.no_grad():
        pt = torch.exp(-bce_loss)
    focal_loss = (prob_fl * (1 - pt) ** gamma) * (mask * bce_loss) + prob_wce * (mask * bce_loss)
    # cost = torch.sum(focal_loss)
    cost = focal_loss.mean()

    return cost * 100.

def dice_loss(inputs, targets, smooth=1e-8):
    # inputs = torch.sigmoid(inputs)
    targets[targets == 2] = 0
    intersection = torch.sum(inputs * targets)
    union = torch.sum(inputs) + torch.sum(targets)
    dice = (2. * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice
    return dice_loss