import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import sys
import numpy as np
import matplotlib.pyplot as plt

from mmcv.cnn import (xavier_init, constant_init, kaiming_init, normal_init)
from models.resnet import resnet50
epsilon = 1e-7


class ConvLSTM(nn.Module):
    def __init__(self, embed_size=512):
        super(ConvLSTM, self).__init__()
        #LSTM gates
        self.input_x = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)
        self.forget_x = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)
        self.output_x = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)
        self.memory_x = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)
        self.input_h = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)
        self.forget_h = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)
        self.output_h = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)
        self.memory_h = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)

        self.input_pos = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)
        self.forget_pos = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)
        self.output_pos = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)
        self.input_neg = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)
        self.forget_neg = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)
        self.output_neg = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)

        self.init_weights()

    def forward(self, x, state, spatial_pos, spatial_neg, semantic_pos, semantic_neg):
        batch, channel, col, row = x.size()

        spatial_semantic_pos = spatial_pos.unsqueeze(1) * semantic_pos.unsqueeze(-1).unsqueeze(-1)
        spatial_semantic_neg = spatial_neg.unsqueeze(1) * semantic_neg.unsqueeze(-1).unsqueeze(-1)

        h, c = state[0], state[1]
        i = torch.sigmoid(self.input_x(x) + self.input_h(h) + self.input_pos(spatial_semantic_pos)
                          + self.input_neg(spatial_semantic_neg))
        f = torch.sigmoid(self.forget_x(x) + self.forget_h(h) + self.forget_pos(spatial_semantic_pos)
                          + self.forget_neg(spatial_semantic_neg))
        o = torch.sigmoid(self.output_x(x) + self.output_h(h) + self.output_pos(spatial_semantic_pos)
                          + self.output_neg(spatial_semantic_neg))
        g = torch.tanh(self.memory_x(x) + self.memory_h(h))

        next_c = f * c + i * g
        h = o * next_c
        state = (h, next_c)

        return h, state

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.01)


class semantic_att(nn.Module):
    def __init__(self, embed_size=512):
        super(semantic_att, self).__init__()
        self.semantic_lists = nn.Linear(embed_size, embed_size, bias=True)
        self.semantic_cur = nn.Linear(embed_size, embed_size, bias=True)
        self.semantic_attention = nn.Linear(embed_size, 1, bias=True)

        self.init_weights()

    def forward(self, visual_lists, visual_cur):
        '''
        visual_lists [N, T, E]
        visual_cur [N, E]
        '''
        semantic_visual_lists = self.semantic_lists(visual_lists)
        semantic_visual_cur = self.semantic_cur(visual_cur)
        semantic_attention = F.softmax(
            self.semantic_attention(semantic_visual_lists + semantic_visual_cur.unsqueeze(1)), 1)
        semantic = (visual_lists * semantic_attention).sum(1)

        return semantic

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.01)


class spatial_att(nn.Module):
    def __init__(self, map_width=40, map_height=30):
        super(spatial_att, self).__init__()
        self.spatial_lists = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1, bias=True)
        self.spatial_cur = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1, bias=True)
        self.spatial_attention = nn.Conv2d(1, 1, kernel_size=(30, 40), padding=0, stride=1, bias=True)
        self.map_width = map_width
        self.map_height = map_height

        self.init_weights()

    def forward(self, visual_lists, visual_cur):
        '''
        visual_lists [N, C, H, W]
        visual_cur [N, 1, H, W]
        '''
        batch, T, height, width = visual_lists.shape
        spatial_visual_lists = self.spatial_lists(visual_lists.view(-1, 1, height, width))
        spatial_visual_cur = self.spatial_cur(visual_cur)
        semantic_attention = F.softmax(
            self.spatial_attention((spatial_visual_lists.view(batch, T, height, width) + spatial_visual_cur)
                                   .view(-1, 1, height, width)).view(batch, T, 1, 1), 1)
        semantic = (visual_lists * semantic_attention).sum(1)

        return semantic

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.01)


class predict_head(nn.Module):
    def __init__(self, convLSTM_length):
        super(predict_head, self).__init__()
        self.convLSTM_length = convLSTM_length
        self.sal_layer_2 = nn.Conv2d(512, 1, kernel_size=1, padding=0, stride=1, bias=True)
        self.sal_layer_3 = nn.Conv2d(512, 1, kernel_size=1, padding=0, stride=1, bias=True)
        self.global_avg = nn.AvgPool2d(kernel_size=(30, 40))

        self.drt_layer_1 = nn.Conv2d(512, 1, kernel_size=7, padding=2, stride=5, bias=True)
        self.drt_layer_2 = nn.Conv2d(1, 2, kernel_size=(6, 8), padding=0, stride=1, bias=True)

        self.init_weights()

    def forward(self, features):
        batch = features.shape[0]
        x = features
        y = self.sal_layer_2(x).squeeze(1)
        y = self.global_avg(y)
        t = F.relu(self.drt_layer_1(x))
        t = self.drt_layer_2(t)
        log_normal_mu = t[:, 0].view(batch, -1)
        log_normal_sigma2 = torch.exp(t[:, 1]).view(batch, -1)
        x = F.relu(self.sal_layer_3(x))
        z = torch.cat([y, x.view(batch, 1, -1)], dim=-1)

        if self.training == False:
            z = F.softmax(z, -1)

        predicts = {}
        # [N, T, A] A = H * W + 1
        predicts['actions'] = z
        # [N, T]
        predicts['log_normal_mu'] = log_normal_mu
        # [N, T]
        predicts['log_normal_sigma2'] = log_normal_sigma2
        # [N, T, H, W]
        predicts["action_map"] = x

        return predicts

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m)
            elif isinstance(m, nn.Conv3d):
                xavier_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.01)

class baseline(nn.Module):
    def __init__(self, embed_size=512, convLSTM_length=16, min_length=1, ratio=4,
                  map_width=40, map_height=30):
        super(baseline, self).__init__()
        self.embed_size = embed_size
        self.ratio = ratio
        self.convLSTM_length = convLSTM_length
        self.min_length = min_length
        self.downsampling_rate = 8
        self.map_width = map_width
        self.map_height = map_height
        self.performance_situation = ["False", "True"]
        self.int2performance = {i: self.performance_situation[i] for i in range(len(self.performance_situation))}

        self.resnet = resnet50(pretrained=True)
        self.dilate_resnet(self.resnet)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        self.sal_conv = nn.Conv2d(2048, 512, kernel_size=3, padding=1, stride=1, bias=True)
        self.lstm = ConvLSTM(self.embed_size)

        self.semantic_embed = nn.Linear(512, embed_size)
        self.spatial_embed = nn.Linear(1200, 1200, bias=True)
        self.semantic_att = semantic_att(embed_size=512)
        self.spatial_att = spatial_att(map_width, map_height)

        self.performance_sal_layer = nn.ModuleDict(
            {self.performance_situation[i]: nn.Conv2d(512, 512, kernel_size=5, padding=2, stride=1,
                                                      bias=True) for
             i in range(len(self.performance_situation))})
        self.object_head = predict_head(convLSTM_length)

        self.init_weights()


    def init_hidden(self, x): #initializing hidden state as all zero
        h = torch.zeros_like(x)
        c = torch.zeros_like(x)
        return (h, c)

    def dilate_resnet(self, resnet):    #modifying resnet as in SAM paper
        resnet.layer2[0].conv1.stride = 1
        resnet.layer2[0].downsample[0].stride = 1
        resnet.layer4[0].conv1.stride = 1
        resnet.layer4[0].downsample[0].stride = 1

        for block in resnet.layer3:
            block.conv2.dilation = 2
            block.conv2.padding = 2

        for block in resnet.layer4:
            block.conv2.dilation = 4
            block.conv2.padding = 4

    def get_spatial_semantic(self, action_map, visual_feature):
        semantic_feature = action_map.expand_as(visual_feature) * visual_feature
        semantic_feature = semantic_feature.mean(1, keepdims=True)

        return semantic_feature

    def get_channel_semantic(self, action_map, visual_feature):
        semantic_feature = action_map.expand_as(visual_feature) * visual_feature
        semantic_feature = semantic_feature.view(visual_feature.shape[0], visual_feature.shape[1], -1).mean(-1)

        return semantic_feature


    def forward(self, images, attention_maps, performances=None):
        # scanpath is used for the extract embedding feature to the ConvLSTM modules  (We do not use it at this model)
        # durations is used in the ConvLSTM modules (We do not use it at this model)
        # active_scanpath_temporal_masks is used for training the saliency map and obtained from duration_masks

        if self.training:
            predicts = self.training_process(images, attention_maps, performances)
        else:
            predicts = self.inference(images, attention_maps)

        return predicts

    def training_process(self, images, attention_maps, performances):
        # img = img.unsqueeze(0)
        batch, _, height, width = images.size()# build a one-hot performance embedding

        x = self.resnet(images)
        visual_feature = F.relu(self.sal_conv(x)) #change filter size

        spatial_lists_pos = list()
        spatial_lists_neg = list()
        semantic_lists_pos = list()
        semantic_lists_neg = list()

        spatial_feature_pos = F.relu(self.get_spatial_semantic(attention_maps, visual_feature))
        spatial_feature_neg = F.relu(self.get_spatial_semantic(attention_maps, visual_feature))
        spatial_feature_pos = self.spatial_embed(spatial_feature_pos.view(batch, 1, -1)).view(batch, 1, 30, 40)
        spatial_feature_neg = self.spatial_embed(spatial_feature_neg.view(batch, 1, -1)).view(batch, 1, 30, 40)
        spatial_lists_pos.append(spatial_feature_pos)
        spatial_lists_neg.append(spatial_feature_neg)
        semantic_feature_pos = F.relu(self.get_channel_semantic(attention_maps, visual_feature))
        semantic_feature_neg = F.relu(self.get_channel_semantic(attention_maps, visual_feature))
        semantic_feature_pos = self.semantic_embed(semantic_feature_pos)
        semantic_feature_neg = self.semantic_embed(semantic_feature_neg)
        semantic_lists_pos.append(semantic_feature_pos)
        semantic_lists_neg.append(semantic_feature_neg)

        spatial_mem_pos = self.spatial_att(torch.cat([_ for _ in spatial_lists_pos], 1), spatial_feature_pos)
        spatial_mem_neg = self.spatial_att(torch.cat([_ for _ in spatial_lists_neg], 1), spatial_feature_neg)

        semantic_mem_pos = self.semantic_att(torch.cat([_.unsqueeze(1) for _ in semantic_lists_pos], 1),
                                             semantic_feature_pos)
        semantic_mem_neg = self.semantic_att(torch.cat([_.unsqueeze(1) for _ in semantic_lists_neg], 1),
                                             semantic_feature_neg)

        state = self.init_hidden(visual_feature)  # initialize hidden state as zeros

        #sequential model
        good_predict_alls = list()
        poor_predict_alls = list()
        for i in range(self.convLSTM_length):
            output, state = self.lstm(visual_feature, state, spatial_mem_pos, spatial_mem_neg,
                                      semantic_mem_pos, semantic_mem_neg)
            good_feature = self.performance_sal_layer["True"](output)
            good_predict_head_rlts = self.object_head(good_feature)
            poor_feature = self.performance_sal_layer["False"](output)
            poor_predict_head_rlts = self.object_head(poor_feature)

            good_predict_alls.append(good_predict_head_rlts)
            poor_predict_alls.append(poor_predict_head_rlts)

            good_predict_action_map = good_predict_head_rlts["action_map"]
            poor_predict_action_map = poor_predict_head_rlts["action_map"]

            spatial_feature_pos = F.relu(self.get_spatial_semantic(good_predict_action_map, visual_feature))
            spatial_feature_neg = F.relu(self.get_spatial_semantic(poor_predict_action_map, visual_feature))
            spatial_feature_pos = self.spatial_embed(spatial_feature_pos.view(batch, 1, -1)).view(batch, 1, 30, 40)
            spatial_feature_neg = self.spatial_embed(spatial_feature_neg.view(batch, 1, -1)).view(batch, 1, 30, 40)
            spatial_lists_pos.append(spatial_feature_pos)
            spatial_lists_neg.append(spatial_feature_neg)
            semantic_feature_pos = F.relu(self.get_channel_semantic(good_predict_action_map, visual_feature))
            semantic_feature_neg = F.relu(self.get_channel_semantic(poor_predict_action_map, visual_feature))
            semantic_feature_pos = self.semantic_embed(semantic_feature_pos)
            semantic_feature_neg = self.semantic_embed(semantic_feature_neg)
            semantic_lists_pos.append(semantic_feature_pos)
            semantic_lists_neg.append(semantic_feature_neg)

            spatial_mem_pos = self.spatial_att(torch.cat([_ for _ in spatial_lists_pos], 1), spatial_feature_pos)
            spatial_mem_neg = self.spatial_att(torch.cat([_ for _ in spatial_lists_neg], 1), spatial_feature_neg)

            semantic_mem_pos = self.semantic_att(torch.cat([_.unsqueeze(1) for _ in semantic_lists_pos], 1),
                                                 semantic_feature_pos)
            semantic_mem_neg = self.semantic_att(torch.cat([_.unsqueeze(1) for _ in semantic_lists_neg], 1),
                                                 semantic_feature_neg)

        good_predict = dict()
        poor_predict = dict()
        for predicts, save_predict in zip([good_predict_alls, poor_predict_alls], [good_predict, poor_predict]):
            actions_pools = list()
            log_normal_mu_pools = list()
            log_normal_sigma2_pools = list()
            action_map_pools = list()
            for i in range(self.convLSTM_length):
                actions_pools.append(predicts[i]["actions"])
                log_normal_mu_pools.append(predicts[i]["log_normal_mu"])
                log_normal_sigma2_pools.append(predicts[i]["log_normal_sigma2"])
                action_map_pools.append(predicts[i]["action_map"])
            save_predict["actions"] = torch.cat(actions_pools, axis=1)
            save_predict["log_normal_mu"] = torch.cat(log_normal_mu_pools, axis=1)
            save_predict["log_normal_sigma2"] = torch.cat(log_normal_sigma2_pools, axis=1)
            save_predict["action_map"] = torch.cat(action_map_pools, axis=1)

        predict_head_rlts = dict()
        actions_pools = list()
        log_normal_mu_pools = list()
        log_normal_sigma2_pools = list()
        action_map_pools = list()
        for index in range(batch):
            if performances[index]:
                actions_pools.append(good_predict["actions"][index].unsqueeze(0))
                log_normal_mu_pools.append(good_predict["log_normal_mu"][index].unsqueeze(0))
                log_normal_sigma2_pools.append(good_predict["log_normal_sigma2"][index].unsqueeze(0))
                action_map_pools.append(good_predict["action_map"][index].unsqueeze(0))
            else:
                actions_pools.append(poor_predict["actions"][index].unsqueeze(0))
                log_normal_mu_pools.append(poor_predict["log_normal_mu"][index].unsqueeze(0))
                log_normal_sigma2_pools.append(poor_predict["log_normal_sigma2"][index].unsqueeze(0))
                action_map_pools.append(poor_predict["action_map"][index].unsqueeze(0))
        predict_head_rlts["actions"] = torch.cat(actions_pools, axis=0)
        predict_head_rlts["log_normal_mu"] = torch.cat(log_normal_mu_pools, axis=0)
        predict_head_rlts["log_normal_sigma2"] = torch.cat(log_normal_sigma2_pools, axis=0)
        predict_head_rlts["action_map"] = torch.cat(action_map_pools, axis=0)

        predicts = {}
        # [N, T, A] A = H * W + 1
        predicts['all_actions_prob'] = predict_head_rlts["actions"]
        # [N, T]
        predicts['log_normal_mu'] = predict_head_rlts["log_normal_mu"]
        # [N, T]
        predicts['log_normal_sigma2'] = predict_head_rlts["log_normal_sigma2"]
        return predicts

    def inference(self, images, attention_maps):
        # img = img.unsqueeze(0)
        batch, _, height, width = images.size()  # build a one-hot performance embedding

        x = self.resnet(images)
        visual_feature = F.relu(self.sal_conv(x))  # change filter size

        spatial_lists_pos = list()
        spatial_lists_neg = list()
        semantic_lists_pos = list()
        semantic_lists_neg = list()

        spatial_feature_pos = F.relu(self.get_spatial_semantic(attention_maps, visual_feature))
        spatial_feature_neg = F.relu(self.get_spatial_semantic(attention_maps, visual_feature))
        spatial_feature_pos = self.spatial_embed(spatial_feature_pos.view(batch, 1, -1)).view(batch, 1, 30, 40)
        spatial_feature_neg = self.spatial_embed(spatial_feature_neg.view(batch, 1, -1)).view(batch, 1, 30, 40)
        spatial_lists_pos.append(spatial_feature_pos)
        spatial_lists_neg.append(spatial_feature_neg)
        semantic_feature_pos = F.relu(self.get_channel_semantic(attention_maps, visual_feature))
        semantic_feature_neg = F.relu(self.get_channel_semantic(attention_maps, visual_feature))
        semantic_feature_pos = self.semantic_embed(semantic_feature_pos)
        semantic_feature_neg = self.semantic_embed(semantic_feature_neg)
        semantic_lists_pos.append(semantic_feature_pos)
        semantic_lists_neg.append(semantic_feature_neg)

        spatial_mem_pos = self.spatial_att(torch.cat([_ for _ in spatial_lists_pos], 1), spatial_feature_pos)
        spatial_mem_neg = self.spatial_att(torch.cat([_ for _ in spatial_lists_neg], 1), spatial_feature_neg)

        semantic_mem_pos = self.semantic_att(torch.cat([_.unsqueeze(1) for _ in semantic_lists_pos], 1),
                                             semantic_feature_pos)
        semantic_mem_neg = self.semantic_att(torch.cat([_.unsqueeze(1) for _ in semantic_lists_neg], 1),
                                             semantic_feature_neg)

        state = self.init_hidden(visual_feature)  # initialize hidden state as zeros

        # sequential model
        good_predict_alls = list()
        poor_predict_alls = list()
        for i in range(self.convLSTM_length):
            output, state = self.lstm(visual_feature, state, spatial_mem_pos, spatial_mem_neg,
                                      semantic_mem_pos, semantic_mem_neg)
            good_feature = self.performance_sal_layer["True"](output)
            good_predict_head_rlts = self.object_head(good_feature)
            poor_feature = self.performance_sal_layer["False"](output)
            poor_predict_head_rlts = self.object_head(poor_feature)

            good_predict_alls.append(good_predict_head_rlts)
            poor_predict_alls.append(poor_predict_head_rlts)

            good_predict_action_map = good_predict_head_rlts["action_map"]
            poor_predict_action_map = poor_predict_head_rlts["action_map"]
            spatial_feature_pos = F.relu(self.get_spatial_semantic(good_predict_action_map, visual_feature))
            spatial_feature_neg = F.relu(self.get_spatial_semantic(poor_predict_action_map, visual_feature))
            spatial_feature_pos = self.spatial_embed(spatial_feature_pos.view(batch, 1, -1)).view(batch, 1, 30, 40)
            spatial_feature_neg = self.spatial_embed(spatial_feature_neg.view(batch, 1, -1)).view(batch, 1, 30, 40)
            spatial_lists_pos.append(spatial_feature_pos)
            spatial_lists_neg.append(spatial_feature_neg)
            semantic_feature_pos = F.relu(self.get_channel_semantic(good_predict_action_map, visual_feature))
            semantic_feature_neg = F.relu(self.get_channel_semantic(poor_predict_action_map, visual_feature))
            semantic_feature_pos = self.semantic_embed(semantic_feature_pos)
            semantic_feature_neg = self.semantic_embed(semantic_feature_neg)
            semantic_lists_pos.append(semantic_feature_pos)
            semantic_lists_neg.append(semantic_feature_neg)

            spatial_mem_pos = self.spatial_att(torch.cat([_ for _ in spatial_lists_pos], 1), spatial_feature_pos)
            spatial_mem_neg = self.spatial_att(torch.cat([_ for _ in spatial_lists_neg], 1), spatial_feature_neg)

            semantic_mem_pos = self.semantic_att(torch.cat([_.unsqueeze(1) for _ in semantic_lists_pos], 1),
                                                 semantic_feature_pos)
            semantic_mem_neg = self.semantic_att(torch.cat([_.unsqueeze(1) for _ in semantic_lists_neg], 1),
                                                 semantic_feature_neg)

        good_predict = dict()
        poor_predict = dict()
        for predicts, save_predict in zip([good_predict_alls, poor_predict_alls], [good_predict, poor_predict]):
            actions_pools = list()
            log_normal_mu_pools = list()
            log_normal_sigma2_pools = list()
            action_map_pools = list()
            for i in range(self.convLSTM_length):
                actions_pools.append(predicts[i]["actions"])
                log_normal_mu_pools.append(predicts[i]["log_normal_mu"])
                log_normal_sigma2_pools.append(predicts[i]["log_normal_sigma2"])
                action_map_pools.append(predicts[i]["action_map"])
            save_predict["actions"] = torch.cat(actions_pools, axis=1)
            save_predict["log_normal_mu"] = torch.cat(log_normal_mu_pools, axis=1)
            save_predict["log_normal_sigma2"] = torch.cat(log_normal_sigma2_pools, axis=1)
            save_predict["action_map"] = torch.cat(action_map_pools, axis=1)


        predicts = {}
        # [N, T, A] A = H * W + 1
        predicts["good_all_actions_prob"] = good_predict["actions"]
        # [N, T]
        predicts["good_log_normal_mu"] = good_predict["log_normal_mu"]
        # [N, T]
        predicts["good_log_normal_sigma2"] = good_predict["log_normal_sigma2"]
        # [N, T, H, W]
        predicts["good_action_map"] = good_predict["action_map"]
        # [N, T, A] A = H * W + 1
        predicts["poor_all_actions_prob"] = poor_predict["actions"]
        # [N, T]
        predicts["poor_log_normal_mu"] = poor_predict["log_normal_mu"]
        # [N, T]
        predicts["poor_log_normal_sigma2"] = poor_predict["log_normal_sigma2"]
        # [N, T, H, W]
        predicts["poor_action_map"] = poor_predict["action_map"]

        return predicts

    def init_weights(self):
        for modules in [self.sal_conv.modules(), self.performance_sal_layer.modules(),
                        self.semantic_embed.modules(), self.spatial_embed.modules()]:
            for m in modules:
                if isinstance(m, nn.Conv2d):
                    xavier_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)

    def last_zero_init(self, m):
        if isinstance(m, nn.Sequential):
            constant_init(m[-1], val=0)
        else:
            constant_init(m, val=0)
