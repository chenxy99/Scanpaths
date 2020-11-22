import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import sys
import numpy as np

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

        self.input = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)
        self.forget = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)
        self.output = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)

        self.init_weights()

    def forward(self, x, state, spatial, semantic):
        batch, channel, col, row = x.size()

        spatial_semantic = spatial.unsqueeze(1) * semantic.unsqueeze(-1).unsqueeze(-1)

        h, c = state[0], state[1]
        i = torch.sigmoid(self.input_x(x) + self.input_h(h) + self.input(spatial_semantic))
        f = torch.sigmoid(self.forget_x(x) + self.forget_h(h) + self.forget(spatial_semantic))
        o = torch.sigmoid(self.output_x(x) + self.output_h(h) + self.output(spatial_semantic))
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
                  map_width=40, map_height=30, projected_label_length=18):
        super(baseline, self).__init__()
        self.embed_size = embed_size
        self.ratio = ratio
        self.convLSTM_length = convLSTM_length
        self.min_length = min_length
        self.downsampling_rate = 8
        self.map_width = map_width
        self.map_height = map_height

        self.resnet = resnet50(pretrained=True)
        self.dilate_resnet(self.resnet)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        self.sal_conv = nn.Conv2d(2048, 512, kernel_size=3, padding=1, stride=1, bias=True)
        self.lstm = ConvLSTM(self.embed_size)

        self.semantic_embed = nn.Linear(512, embed_size)
        self.spatial_embed = nn.Linear(1200, 1200, bias=True)
        self.semantic_att = semantic_att(embed_size=512)
        self.spatial_att = spatial_att(map_width, map_height)

        self.performance_sal_layer = nn.Conv2d(512, 512, kernel_size=5, padding=2, stride=1, bias=True)
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


    def forward(self, images):
        # scanpath is used for the extract embedding feature to the ConvLSTM modules  (We do not use it at this model)
        # durations is used in the ConvLSTM modules (We do not use it at this model)
        # active_scanpath_temporal_masks is used for training the saliency map and obtained from duration_masks

        if self.training:
            predicts = self.training_process(images)
        else:
            predicts = self.inference(images)

        return predicts

    def training_process(self, images):
        # img = img.unsqueeze(0)
        batch, _, height, width = images.size()# build a one-hot performance embedding

        x = self.resnet(images)
        visual_feature = F.relu(self.sal_conv(x)) #change filter size

        spatial_lists = list()
        semantic_lists = list()

        attention_maps = images.new_zeros((batch, 1, self.map_height, self.map_width))

        spatial_feature = F.relu(self.get_spatial_semantic(attention_maps, visual_feature))
        spatial_feature = self.spatial_embed(spatial_feature.view(batch, 1, -1)).view(batch, 1, 30, 40)
        spatial_lists.append(spatial_feature)
        semantic_feature = F.relu(self.get_channel_semantic(attention_maps, visual_feature))
        semantic_feature = self.semantic_embed(semantic_feature)
        semantic_lists.append(semantic_feature)

        spatial_mem = self.spatial_att(torch.cat([_ for _ in spatial_lists], 1), spatial_feature)
        semantic_mem = self.semantic_att(torch.cat([_.unsqueeze(1) for _ in semantic_lists], 1), semantic_feature)

        state = self.init_hidden(visual_feature)  # initialize hidden state as zeros

        #sequential model
        predict_alls = list()
        for i in range(self.convLSTM_length):
            output, state = self.lstm(visual_feature, state, spatial_mem, semantic_mem)

            feature = self.performance_sal_layer(output)
            predict_head_rlts = self.object_head(feature)

            predict_alls.append(predict_head_rlts)

            predict_action_map = predict_head_rlts["action_map"]

            spatial_feature = F.relu(self.get_spatial_semantic(predict_action_map, visual_feature))
            spatial_feature = self.spatial_embed(spatial_feature.view(batch, 1, -1)).view(batch, 1, 30, 40)
            spatial_lists.append(spatial_feature)
            semantic_feature = F.relu(self.get_channel_semantic(predict_action_map, visual_feature))
            semantic_feature = self.semantic_embed(semantic_feature)
            semantic_lists.append(semantic_feature)

            spatial_mem = self.spatial_att(torch.cat([_ for _ in spatial_lists], 1), spatial_feature)
            semantic_mem = self.semantic_att(torch.cat([_.unsqueeze(1) for _ in semantic_lists], 1), semantic_feature)


        predict = dict()
        for predicts, save_predict in zip([predict_alls], [predict]):
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
        predicts['actions'] = predict["actions"]
        # [N, T]
        predicts['log_normal_mu'] = predict["log_normal_mu"]
        # [N, T]
        predicts['log_normal_sigma2'] = predict["log_normal_sigma2"]
        return predicts

    def inference(self, images):
        # img = img.unsqueeze(0)
        batch, _, height, width = images.size()  # build a one-hot performance embedding

        x = self.resnet(images)
        visual_feature = F.relu(self.sal_conv(x))  # change filter size

        spatial_lists = list()
        semantic_lists = list()

        attention_maps = images.new_zeros((batch, 1, self.map_height, self.map_width))

        spatial_feature = F.relu(self.get_spatial_semantic(attention_maps, visual_feature))
        spatial_feature = self.spatial_embed(spatial_feature.view(batch, 1, -1)).view(batch, 1, 30, 40)
        spatial_lists.append(spatial_feature)
        semantic_feature = F.relu(self.get_channel_semantic(attention_maps, visual_feature))
        semantic_feature = self.semantic_embed(semantic_feature)
        semantic_lists.append(semantic_feature)

        spatial_mem = self.spatial_att(torch.cat([_ for _ in spatial_lists], 1), spatial_feature)
        semantic_mem = self.semantic_att(torch.cat([_.unsqueeze(1) for _ in semantic_lists], 1), semantic_feature)

        state = self.init_hidden(visual_feature)  # initialize hidden state as zeros

        # sequential model
        predict_alls = list()
        for i in range(self.convLSTM_length):
            output, state = self.lstm(visual_feature, state, spatial_mem, semantic_mem)

            feature = self.performance_sal_layer(output)
            predict_head_rlts = self.object_head(feature)

            predict_alls.append(predict_head_rlts)

            predict_action_map = predict_head_rlts["action_map"]

            spatial_feature = F.relu(self.get_spatial_semantic(predict_action_map, visual_feature))
            spatial_feature = self.spatial_embed(spatial_feature.view(batch, 1, -1)).view(batch, 1, 30, 40)
            spatial_lists.append(spatial_feature)
            semantic_feature = F.relu(self.get_channel_semantic(predict_action_map, visual_feature))
            semantic_feature = self.semantic_embed(semantic_feature)

            semantic_lists.append(semantic_feature)

            spatial_mem = self.spatial_att(torch.cat([_ for _ in spatial_lists], 1), spatial_feature)
            semantic_mem = self.semantic_att(torch.cat([_.unsqueeze(1) for _ in semantic_lists], 1), semantic_feature)

        predict = dict()
        for predicts, save_predict in zip([predict_alls], [predict]):
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
        predicts["all_actions_prob"] = predict["actions"]
        # [N, T]
        predicts["log_normal_mu"] = predict["log_normal_mu"]
        # [N, T]
        predicts["log_normal_sigma2"] = predict["log_normal_sigma2"]
        # [N, T, H, W]
        predicts["action_map"] = predict["action_map"]

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
