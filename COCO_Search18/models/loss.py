import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from numpy import pi
import math

epsilon = 1e-7 #regularization value in Keras

def CrossEntropyLoss(input, gt, mask):
    batch, time_scale, action = input.size()
    input = F.softmax(input, dim=-1)
    loss = -(gt * torch.log(input + epsilon) * mask.unsqueeze(-1)).sum() / mask.sum()
    return loss

def DurationSmoothL1Loss(input, gt, mask):
    batch, time_scale = input.size()
    loss = F.smooth_l1_loss(input * mask, gt * mask, reduction='sum') / mask.sum()
    return loss

def MLPRayleighDistribution(Rayleigh_sigma2, gt, mask):
    batch, time_scale = Rayleigh_sigma2.size()
    logpdf = torch.log(gt / Rayleigh_sigma2 + epsilon) + (- gt ** 2 / (2 * Rayleigh_sigma2))
    loss = (logpdf[mask == 1]).sum() / mask.sum()
    return -loss

def MLPLogNormalDistribution(log_normal_mu, log_normal_sigma2, gt, mask):
    batch, time_scale = log_normal_mu.size()
    logpdf = torch.log(1 / (gt + epsilon) * 1 / (torch.sqrt(2 * math.pi * log_normal_sigma2))) \
             + (- (torch.log(gt + epsilon) - log_normal_mu) ** 2 / (2 * log_normal_sigma2))
    loss = (logpdf[mask == 1]).sum() / mask.sum()
    return -loss

def LogAction(input, mask):
    batch, time_scale = input.size()
    action_logprobs = (torch.log(input + epsilon) * mask).sum(dim=-1) / mask.sum()
    return action_logprobs

def LogDuration(input, log_normal_mu, log_normal_sigma2, mask):
    batch, time_scale = input.size()
    duration_logprob_items = torch.log(1 / (input + epsilon) * 1 / (torch.sqrt(2 * math.pi * log_normal_sigma2))) \
                             + (- (torch.log(input + epsilon) - log_normal_mu) ** 2 / (2 * log_normal_sigma2))

    duration_logprobs = (duration_logprob_items * mask).sum(dim=-1) / mask.sum()
    return duration_logprobs

def NSS(input, fixation):
    # Normalized Scanpath Saliency (NSS)
    inputs = input.view(input.shape[0], -1)
    inputs = inputs / (inputs.max(-1, keepdim=True)[0] + epsilon)
    fixations = fixation.view(fixation.shape[0], -1)
    inputs = (inputs - inputs.mean(-1, keepdim=True)) / (inputs.std(-1, keepdim=True) + epsilon)
    loss = ((inputs * fixations).sum(-1) / (fixations.sum(-1) + epsilon)).mean()

    return loss

def CC(input, salmap):
    # Linear Correlation Coefficient (CC)
    input = input.reshape(input.shape[0], -1)
    salmap = salmap.reshape(salmap.shape[0], -1)
    input_normalized = input / (input.sum(-1, keepdim=True) + epsilon)
    salmap_normalized = salmap / (salmap.sum(-1, keepdim=True) + epsilon)

    input_centered = input_normalized - input_normalized.mean(-1, keepdim=True)
    salmap_centered = salmap_normalized - salmap_normalized.mean(-1, keepdim=True)

    cov_xy = (input_centered * salmap_centered).sum(-1)
    sigma_x = torch.sqrt((input_centered ** 2).sum(-1))
    sigma_y = torch.sqrt((salmap_centered ** 2).sum(-1))

    loss = (cov_xy / (sigma_x * sigma_y + epsilon)).mean()

    return loss

def CC_terms(input, salmap, good_duration_masks, poor_duration_masks):
    # Linear Correlation Coefficient (CC)
    paired_masks = (good_duration_masks.sum(-1) > 0) * (poor_duration_masks.sum(-1) > 0)
    if paired_masks.sum() > 0:
        input = input[paired_masks]
        salmap = salmap[paired_masks]
        input = input.reshape(input.shape[0], -1)
        salmap = salmap.reshape(salmap.shape[0], -1)
        input_normalized = input / (input.sum(-1, keepdim=True) + epsilon)
        salmap_normalized = salmap / (salmap.sum(-1, keepdim=True) + epsilon)

        input_centered = input_normalized - input_normalized.mean(-1, keepdim=True)
        salmap_centered = salmap_normalized - salmap_normalized.mean(-1, keepdim=True)

        cov_xy = (input_centered * salmap_centered).sum(-1)
        sigma_x = torch.sqrt((input_centered ** 2).sum(-1))
        sigma_y = torch.sqrt((salmap_centered ** 2).sum(-1))

        loss = (cov_xy / (sigma_x * sigma_y + epsilon))
    else:
        loss = paired_masks.sum().float() * 0
    return loss

    return loss

def CC_MatchLoss(gt_CC, pre_CC):
    loss = (torch.abs(gt_CC - pre_CC)).mean()
    return loss

def KLD(input, salmap):
    # Kullback-Leibler Divergence (KL-Div)
    input = input.view(input.shape[0], -1)
    salmap = salmap.view(salmap.shape[0], -1)
    inputs_prob_normalized = input / (input.sum(-1, keepdim=True) + epsilon)
    salmaps_prob_normalized = salmap / (salmap.sum(-1, keepdim=True) + epsilon)

    loss = (salmaps_prob_normalized *
            torch.log(salmaps_prob_normalized / (inputs_prob_normalized + epsilon) + epsilon)).sum(-1).mean()

    return loss