from PIL import Image
import requests
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
import time
import math
import numpy as np
# from vit_pytorch import ViT
from .transformer_origin import Transformer

class DETR(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """
    def __init__(self, hidden_dim=64, nheads=8,
                 num_encoder_layers=1, num_decoder_layers=6):
        super().__init__()

        # create ResNet-50 backbone

        # create conversion layer
        self.conv = nn.Conv2d(64, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        self.transformer = Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        # self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        # self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.queries_num = 16*16
        conv1d_num = 240*16
        # self.query_pos = nn.Parameter(torch.rand(self.queries_num, hidden_dim))
        self.temperature = 10000
        self.num_pos_feats = hidden_dim

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        # self.row_embed = nn.Parameter(torch.rand(40, hidden_dim // 2))
        # self.col_embed = nn.Parameter(torch.rand(40, hidden_dim // 2))
        # self.row_embed = nn.Parameter(torch.rand(40, hidden_dim))
        # self.col_embed = nn.Parameter(torch.rand(40, hidden_dim))
        # self.row_embed_2 = nn.Parameter(torch.rand(40, hidden_dim))
        # self.col_embed_2 = nn.Parameter(torch.rand(40, hidden_dim))

        self.pos_encoding()
        
        self.down_sample = nn.Conv2d(64, 64, kernel_size=2,stride=2, padding=0, bias=True)
        self.conv1d = nn.Sequential(nn.Conv1d(self.queries_num, 64, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv1d(64, conv1d_num, kernel_size=3, padding=1))
        self.conv1d_decoding = nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv1d(hidden_dim, 2, kernel_size=3, padding=1))
        
        # self.up_sample = nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.up_sample = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)
        # self.up_sample = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)
        # self.conv2 = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim//2, kernel_size=3, padding=1, bias=True),
        #                              nn.ReLU(inplace=True),
        #                              nn.Conv2d(hidden_dim//2, 2, kernel_size=1, stride=1, padding=0, bias=True))

        mask = self.generate_mask(1, 48, 80)
        self.src_mask = mask
        self.tgt_mask = mask[:self.queries_num,:self.queries_num]

    def pos_encoding(self):
        

        # print(self.row_embed.shape)
        num = 40*4
        batch_size = 48
        
        not_mask = torch.ones([batch_size, num, num], dtype=torch.float32).cuda()
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * math.pi
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * math.pi

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=not_mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        self.row_embed = pos_x+nn.Parameter(torch.rand(batch_size, num, num, self.num_pos_feats)).cuda()
        self.col_embed = pos_y+nn.Parameter(torch.rand(batch_size, num, num, self.num_pos_feats)).cuda()
        # self.row_embed = pos_x
        # self.col_embed = pos_y

        self.row_query = pos_x
        self.col_query = pos_y
        self.para_query = nn.Parameter(torch.rand(self.queries_num, 48, self.num_pos_feats)).cuda()

    def generate_mask(self, batch_size, H, W):
        seq_length = H*W
        length = 5
        mask = torch.zeros([batch_size, seq_length, seq_length], dtype=torch.float32).cuda()
        mask = torch.zeros([batch_size, H, W, H, W], dtype=torch.float32).cuda()
        for i in range(H):
            for j in range(W):
                mask[:,i,j,i:(i+3*length),max(0, j-length):j+length] = 1
        # mask = mask.reshape(batch_size, seq_length, seq_length)
        mask = mask.reshape(seq_length, seq_length)
        # print(mask)
        # new = mask.cpu().numpy().reshape(seq_length, seq_length)*255
        # new = new.reshape(H, W, H, W)[10,10,:,:]

        # # new = (new-np.min(new))/(np.max(new)-np.min(new))
        # print(new)
        # im = Image.fromarray(np.uint8(new))
        # im = im.convert('RGB')
        # im.save('/media/zhouyunsong/Toshiba/SenseTimeResearch/pod_ad/Mono3D_Git/GUPNet-main/code/lib/models/'+'1.png')
        # input()
        return mask


    def window_transformer(self, inputs):

        split_num_h = 1
        split_num_w = 2
        input_h, input_w = inputs.shape[-2:]
        # print(inputs.shape)
        input_h, input_w = input_h//split_num_h, input_w//split_num_w
        for i in range(split_num_h):
            for j in range(split_num_w):
                # time1 = time.time()
            
                x = inputs[:,:,i*input_h:(i+1)*input_h, j*input_w:(j+1)*input_w]
                # x = inputs[:,:,:12, :20]

                # convert from 2048 to 256 feature planes for the transformer
                h = self.conv(x)
                # time2 = time.time()

                # construct positional encodings
                H, W = h.shape[-2:]
                # print(h.shape)
                # pos = torch.cat([
                #     self.col_embed_2[:W].unsqueeze(0).repeat(H, 1, 1),
                #     self.row_embed_2[:H].unsqueeze(1).repeat(1, W, 1),
                # ], dim=-1).flatten(0, 1).unsqueeze(1)
                # # print(pos.shape)
                pos = torch.cat([self.col_embed, self.row_embed], dim=3)[:h.shape[0],:H,:W,:h.shape[1]].permute(0, 3, 1, 2).reshape(-1,H*W,h.shape[1]).permute(1,0,2).cuda()

                self.query_pos = torch.cat([self.col_query, self.row_query], dim=3)[:h.shape[0],:H,:W,:h.shape[1]].permute(0, 3, 1, 2).reshape(-1,H*W,h.shape[1]).permute(1,0,2)
                # print(self.query_pos.shape)
                self.query_pos = self.query_pos[0::(pos.shape[0]//self.queries_num),:,:].cuda()+self.para_query[:,:pos.shape[1],:pos.shape[2]].cuda()
                # self.query_pos = self.query_pos[0::(pos.shape[0]//self.queries_num),:,:].cuda()


                # print(pos.shape)
                # print(h.flatten(2).permute(2, 0, 1).shape)
                # input()


                # propagate through the transformer
                # print(pos.shape, self.query_pos.unsqueeze(1).repeat(h.shape[0], axis=1).shape, h.flatten(2).permute(2, 0, 1).shape)
                # h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                #                     self.query_pos.unsqueeze(1).repeat(1, h.shape[0], 1)).reshape(h.shape)
                # print(pos)
                # input()
                # print(h.shape)
                # input()
                # h_1 = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                #                     self.query_pos.unsqueeze(1).repeat(1, h.shape[0], 1))
                # print(pos.shape, self.query_pos.shape)
                # input()
                h_1 = self.transformer(pos + h.flatten(2).permute(2, 0, 1),
                                     self.query_pos,self.src_mask.cuda(),self.tgt_mask.cuda())
                # print(h.shape)
                # print(h_1.shape)
                h_1 = self.conv1d(h_1.permute(1,0,2))
                # print(h_1.shape)
                h_1 = self.conv1d_decoding(h_1.permute(0,2,1)).reshape(h.shape[0], -1, H, W)
                # print(h_1.shape)
                # input()
                h = h_1
                # time3 = time.time()
                # print(time3-time2)
                if j == 0:
                    res_roll = h 
                else:
                    res_roll = torch.cat((res_roll, h), -1)
                # time4 = time.time()

                # print(time4-time3, time3-time2, time2-time1)
            # time1 = time.time()   
            if i == 0:
                res = res_roll 
            else:
                res = torch.cat((res, res_roll), -2)
            # time2 = time.time()
            # print(time2-time1)

        # print(res.shape)
        
        # finally project transformer outputs to class labels and bounding boxes
        # return {'pred_logits': self.linear_class(h), 
        #         'pred_boxes': self.linear_bbox(h).sigmoid()}
        return res


    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        # x = self.backbone.conv1(inputs)
        # x = self.backbone.bn1(x)
        # x = self.backbone.relu(x)
        # x = self.backbone.maxpool(x)

        # x = self.backbone.layer1(x)
        # x = self.backbone.layer2(x)
        # x = self.backbone.layer3(x)
        # x = self.backbone.layer4(x)
        # print(inputs.shape)
        inputs = self.down_sample(inputs)
        
        # time1 = time.time()
        x = self.window_transformer(inputs)
        # x = self.shifted_window_transformer(x)
        # time2 = time.time()
        x = self.up_sample(x)
        # x = self.conv2(x)
        # time3 = time.time()
        # print(time3-time2, time2-time1)
        
        return x
        

def detect(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled