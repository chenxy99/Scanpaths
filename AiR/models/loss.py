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

def KLD_items(input, salmap):
    # Kullback-Leibler Divergence (KL-Div)
    input = input.view(input.shape[0], -1)
    salmap = salmap.view(salmap.shape[0], -1)
    inputs_prob_normalized = input / (input.sum(-1, keepdim=True) + epsilon)
    salmaps_prob_normalized = salmap / (salmap.sum(-1, keepdim=True) + epsilon)

    loss = (salmaps_prob_normalized *
            torch.log(salmaps_prob_normalized / (inputs_prob_normalized + epsilon) + epsilon)).sum(-1)

    return loss

def KLD_visual_linguistic_alignment(input, question_objects_pos, question_objects_masks,
                                    fullAnswer_objects_pos, fullAnswer_objects_masks):
    batch, channel, height, width = input.shape
    gt_visual_attention = (question_objects_pos * question_objects_masks.unsqueeze(1).unsqueeze(1)).sum(-1) + \
                          (fullAnswer_objects_pos * fullAnswer_objects_masks.unsqueeze(1).unsqueeze(1)).sum(-1)
    gt_visual_attention = (gt_visual_attention > 0) * 1.0
    gt_visual_attention /= gt_visual_attention.view(batch, -1).sum(-1).unsqueeze(-1).unsqueeze(-1)

    input = F.softmax(input.view(batch, -1), -1).view(batch, height, width)
    kld_loss = KLD(input.squeeze(1), gt_visual_attention)

    return kld_loss

def KLD_question_aligment(input, question_objects_pos, question_objects_masks, duration_masks):

    batch, channel, height, width = input.shape
    input = F.softmax(input.view(batch * channel, -1), -1).view(batch, channel, height, width)

    # collect_kld = list()
    # for index in range(question_objects_masks.shape[1]):
    #     question_objects_pos_extract = question_objects_pos[:, :, :, index].unsqueeze(1)
    #     question_objects_pos_extract = question_objects_pos_extract.expand(-1, channel, -1, -1).contiguous()
    #     kld_items = KLD_items(input.view(-1, height, width), question_objects_pos_extract.view(-1, height, width))\
    #         .reshape(batch, channel)
    #     collect_kld.append(kld_items.unsqueeze(-1))
    #
    # collect_kld = torch.cat(collect_kld, -1)
    # collect_kld[duration_masks == 0] = float('inf')
    # min_kld = (collect_kld).min(axis=1)[0]
    # KLD_aligment = (min_kld * question_objects_masks).sum() / question_objects_masks.sum()
    KLD_aligment = list()
    for index in range(batch):
        for col in range(question_objects_masks.shape[1]):
            if question_objects_masks[index, col] == 0:
                break
            else:
                question_objects_pos_extract = question_objects_pos[index, :, :, col].unsqueeze(0)
                question_objects_pos_extract = question_objects_pos_extract.expand(channel, -1, -1).contiguous()
                kld_items = KLD_items(input[index], question_objects_pos_extract)
                kld_items[duration_masks[index] == 0] = float('inf')
                KLD_aligment.append(torch.min(kld_items).unsqueeze(0))
    KLD_aligment = torch.cat(KLD_aligment).mean()

    return KLD_aligment



