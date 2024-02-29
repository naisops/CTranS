# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 15:03:50 2022

@author: naisops
"""

import torch
from torch import nn
import torch.nn.functional as F
import math
import einops
import copy

device = torch.device('cuda:0') 
drop_out=0.1

class PositionalEncoder(nn.Module):
    # output shape: (bs, seq_length, (p * p))
    def __init__(self, d_model, seq_len, dropout = drop_out):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x is in the shape of (bs, seq_length, (p * p))
        x = x + self.pos_embed
        return self.dropout(x)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = drop_out):
        super().__init__()
        
        self.d_model = d_model
        self.d_head = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
        
    
    def forward(self, q, k, v):
        bs = q.size(0)
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_head)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_head)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_head)
        
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        scores, sim = attention(q, k, v, self.d_head, self.dropout)
        # scores shape: (bs x h x sl x d_k)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        
        output = self.out(concat)
        # output shape: (bs x sl x d_model)
    
        return output, sim
    
# class MultiHeadAttention_crf(nn.Module):
#     def __init__(self, heads, d_model, n, p, c, seperate=False, dropout = drop_out):
#         super().__init__()
        
#         # heads = p*p
#         # d_head = channel
#         # d_model = channel*p*p
        
#         self.d_head = d_model//heads
#         self.n = n
#         self.p = p
#         self.c = c
#         # self.h = heads
#         self.n_head = int(heads**0.5)
        
#         self.q_linear = nn.Linear(d_model, d_model)
#         self.v_linear = nn.Linear(d_model, d_model)
#         self.k_linear = nn.Linear(d_model, d_model)

#         self.dropout = nn.Dropout(dropout)
#         self.out = nn.Linear(d_model, d_model)
#         self.seperate = seperate
    
#     def forward(self, q, k, v):
        
#         k = self.k_linear(k)
#         q = self.q_linear(q)
#         v = self.v_linear(v)
        
#         if not self.seperate:
#             k = einops.rearrange(k, 'b (n1 n2) (c p1 p2) -> b n1 p1 n2 p2 c', n1=self.n,c=self.c,p1=self.p)
#             k = einops.rearrange(k, 'b n1 (n_head1 p_head1) n2 (n_head p_head) c -> b (n_head1 n_head) (n1 n2) (p_head1 p_head c)'
#                                  ,n_head=self.n_head, n_head1=self.n_head)
#             q = einops.rearrange(q, 'b (n1 n2) (c p1 p2) -> b n1 p1 n2 p2 c', n1=self.n,c=self.c,p1=self.p)
#             q = einops.rearrange(q, 'b n1 (n_head1 p_head1) n2 (n_head p_head) c -> b (n_head1 n_head) (n1 n2) (p_head1 p_head c)'
#                                  ,n_head=self.n_head, n_head1=self.n_head)
#             v = einops.rearrange(v, 'b (n1 n2) (c p1 p2) -> b n1 p1 n2 p2 c', n1=self.n,c=self.c,p1=self.p)
#             v = einops.rearrange(v, 'b n1 (n_head1 p_head1) n2 (n_head p_head) c -> b (n_head1 n_head) (n1 n2) (p_head1 p_head c)'
#                                  ,n_head=self.n_head, n_head1=self.n_head)
#             scores, sim = attention(q, k, v, self.d_head, self.dropout)
#             scores = einops.rearrange(scores, 'b (n_head1 n_head) (n1 n2) (p_head1 p_head c) -> b n1 (n_head1 p_head1) n2 (n_head p_head) c'
#                                       ,n1=self.n,c=self.c,n_head=self.n_head, p_head1=self.p//self.n_head)
#             scores = einops.rearrange(scores, 'b n1 p1 n2 p2 c -> b (n1 n2) (c p1 p2)')
#         else:
#             k = einops.rearrange(k, 'b n (channel p_len) -> b p_len n channel', channel = self.d_head)
#             q = einops.rearrange(q, 'b n (channel p_len) -> b p_len n channel', channel = self.d_head)
#             v = einops.rearrange(v, 'b n (channel p_len) -> b p_len n channel', channel = self.d_head)
#             scores, sim = attention(q, k, v, self.d_head, self.dropout)
#             scores = einops.rearrange(scores, 'b head n dh -> b n (dh head)')

#         scores = self.out(scores)
    
#         return scores, sim

class MultiHeadAttention_crf(nn.Module):
    def __init__(self, heads, d_model, n, p, c, dropout = drop_out):
        super().__init__()
        
        # heads = p*p
        # d_head = channel
        # d_model = channel*p*p
        
        self.d_head = d_model//heads
        self.n = n
        self.p = p
        self.c = c
        # self.h = heads
        self.n_head = int(heads**0.5)
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v):
        
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)
        

        k = einops.rearrange(k, 'b n (h channel) -> b h n channel', channel = self.d_head)
        q = einops.rearrange(q, 'b n (h channel) -> b h n channel', channel = self.d_head)
        v = einops.rearrange(v, 'b n (h channel) -> b h n channel', channel = self.d_head)
        scores, sim = attention(q, k, v, self.d_head, self.dropout)
        scores = einops.rearrange(scores, 'b head n dh -> b n (head dh)')

        scores = self.out(scores)
    
        return scores, sim
       
def attention(q, k, v, d_k, dropout=None):  
    # q,k,v shape: (bs x h x sl x d_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    # output shape: (bs x h x sl x d_k)
    return output, scores

class FeedForward(nn.Module):
    
    def __init__(self, d_model, dropout = drop_out):
        super().__init__() 
        self.linear_1 = nn.Linear(d_model, d_model*2)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_model*2, d_model)
        
    def forward(self, x):
        x = self.dropout(self.act(self.linear_1(x)))
        x = self.linear_2(x)
        
        return x
    
class Norm(nn.Module):
    
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        norm = self.norm(x)
        
        return norm
    
class EncoderLayer(nn.Module):
    # inpuut: (bs,C,H,W)
    
    def __init__(self, d_model, heads, dropout = drop_out):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x2 = self.norm_1(x)
        temp, sim = self.attn(x2,x2,x2)
        x = x + self.dropout_1(temp)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x, sim

class EncoderLayer_crf(nn.Module):
    # inpuut: (bs,C,H,W)
    
    def __init__(self, d_model, heads, n, p, c, dropout = drop_out):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention_crf(heads, d_model, n, p, c)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x2 = self.norm_1(x)
        temp, sim = self.attn(x2,x2,x2)
        x = x + self.dropout_1(temp)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x, sim


class transBlock(nn.Module):
    def __init__(self, d, heads, n_trans):
        super().__init__()
        layers = []
        self.n_trans = n_trans
        for i in range(n_trans):
            layers.append(EncoderLayer(d, heads))
             
        self.sequential = nn.Sequential(*layers)
    
    def forward(self, x):
        for i in range(self.n_trans):
            x, sim = self.sequential[i](x)
            
        return x, sim
    
    
class ConvBlock(nn.Module):
    # input (bs, C_in, D_in, H_in, W_in)
    # output (bs, C_out, D_out, H_out, W_out)
    
    def __init__(self, in_channels, out_channels, padding = (1,1), kernel_size = 3, dropout=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding = padding)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()
        self.dropout = dropout
        if dropout:
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding = padding)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()
        self.conv_skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding = padding),
            nn.BatchNorm2d(out_channels),
        )

        
    def forward(self, x):
        skip = self.conv_skip(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        if self.dropout:
            x = self.dropout1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        if self.dropout:
            x = self.dropout2(x)

        return x+skip
    
class CTS(nn.Module):

    def __init__(self, ini_channel, base_channel, convN, im_size, n_trans, num_class,c0):
        super().__init__()
        self.num_class = num_class
        self.im_size = im_size
        self.convN = convN
        self.n = im_size // (2**convN)
        p = 2**convN
        d = base_channel*(2**(convN))
        self.p = p

        self.conv = []
        self.conv.append(ConvBlock(ini_channel, base_channel))
        for i in range(convN):
            self.conv.append(nn.MaxPool2d(2))
            self.conv.append(ConvBlock(base_channel*(2**(i)), base_channel*(2**(i+1))))
        self.conv = nn.Sequential(*self.conv)
        
        self.linear = []
        for i in range(convN):
            self.linear.append(nn.Linear(base_channel*(2**(i)),base_channel*(2**(i))//c0))
        self.linear = nn.Sequential(*self.linear)
        
        self.pe = PositionalEncoder(d, (im_size//p)**2)
        self.trans = []
        self.trans0 = transBlock(d, d//(base_channel), n_trans)
        # self.trans0 = transBlock(d, 1, n_trans)

        for i in range(convN):
            if i == 0:
                self.trans.append(nn.Linear(d,d*(2**(i+1))//c0))
            else:
                self.trans.append(nn.Linear(d*(2**i)//c0,d*(2**(i+1))//c0))
            self.trans.append(transBlock(d*(2**(i+1))//c0, d*(2**(i+1))//(c0*base_channel), n_trans)) 
            # self.trans.append(transBlock(d*(2**(i+1))//c0, (2**(i+1))**2, n_trans))
        self.trans = nn.Sequential(*self.trans)
        
        # self.final_norm = Norm((d*p)//c0)
        # self.final_att = MultiHeadAttention(p*p, (d*p)//c0)

        self.seg_head = nn.Conv2d(base_channel//c0, num_class, (1,1))
        if num_class < 2:
            self.out = nn.Sigmoid()
        
    def forward(self, x, convn=True): 
        cnn_out = []
        for i,l in enumerate(self.conv):
            x = l(x)
            if i%2 == 0 and i != self.convN*2:
                x0 = einops.rearrange(x, 'b c h w -> b h w c')
                x0 = self.linear[i//2](x0)
                x0 = einops.rearrange(x0, 'b (n1 p1) (n2 p2) c -> b (c p1 p2) n1 n2', n1=self.n, n2=self.n)
                x0 = einops.rearrange(x0, 'b c h w -> b (h w) c')
                cnn_out.append(x0)
        
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        x = self.pe(x)
        x,sim = self.trans0(x)

        for i,l in enumerate(self.trans):
            if i % 2 == 0:
                x = l(x)
                x = x + (cnn_out[self.convN - 1 - i//2])
            else:
                x, sim = l(x)
        
        # temp = self.final_norm(x)
        # _, sim = self.final_att(temp,temp,temp)
        
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=self.n, w=self.n)
        x = einops.rearrange(x, 'b (c p1 p2) n1 n2 -> b c (n1 p1) (n2 p2)', p1=2**self.convN, p2=2**self.convN)
        x = self.seg_head(x)
        if self.num_class < 2:
            x = self.out(x)
        return x
    
class CTS_NoDSL(nn.Module):
    
    def __init__(self, ini_channel, base_channel, convN, im_size, n_trans, num_class):
        super().__init__()
        self.num_class = num_class
        self.im_size = im_size
        self.convN = convN
        self.n = im_size // (2**convN)
        p = 2**convN
        d = base_channel*(2**(convN))
        self.p = p

        self.conv = []
        self.conv.append(ConvBlock(ini_channel, base_channel))
        for i in range(convN):
            self.conv.append(nn.MaxPool2d(2))
            self.conv.append(ConvBlock(base_channel*(2**(i)), base_channel*(2**(i+1))))
        self.conv = nn.Sequential(*self.conv)
        
        self.pe = PositionalEncoder(d, (im_size//p)**2)
        self.trans = []
        self.trans0 = transBlock(d, d//base_channel, n_trans)
        for i in range(convN):
            self.trans.append(nn.Linear(d*(2**i),d*(2**(i+1))))
            self.trans.append(transBlock(d*(2**(i+1)), d*(2**(i+1))//base_channel, n_trans))       
        self.trans = nn.Sequential(*self.trans)

        self.seg_head = nn.Conv2d(base_channel, num_class, (1,1))
        self.out = nn.Sigmoid()
        
    def forward(self, x, convn=True):     
        cnn_out = []
        for i,l in enumerate(self.conv):
            x = l(x)
            if i%2 == 0 and i != self.convN*2:
                x0 = einops.rearrange(x, 'b c (n1 p1) (n2 p2) -> b (c p1 p2) n1 n2', n1=self.n, n2=self.n)
                x0 = einops.rearrange(x0, 'b c h w -> b (h w) c')
                cnn_out.append(x0)
                
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        x = self.pe(x)
        x,sim = self.trans0(x)

        for i,l in enumerate(self.trans):
            if i % 2 == 0:
                x = l(x)
                x = x + cnn_out[self.convN - 1 - i//2]
            else:
                x,sim = l(x)
        
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=self.n, w=self.n)
        x = einops.rearrange(x, 'b (c p1 p2) n1 n2 -> b c (n1 p1) (n2 p2)', p1=2**self.convN, p2=2**self.convN)
        x = self.seg_head(x)
        x = self.out(x)
        return x,sim
    
class CTS_NoSkip(nn.Module):
    
    def __init__(self, ini_channel, base_channel, convN, im_size, n_trans, num_class,c0):
        super().__init__()
        self.num_class = num_class
        self.im_size = im_size
        self.convN = convN
        self.n = im_size // (2**convN)
        p = 2**convN
        d = base_channel*(2**(convN))
        self.p = p

        self.conv = []
        self.conv.append(ConvBlock(ini_channel, base_channel))
        for i in range(convN):
            self.conv.append(nn.MaxPool2d(2))
            self.conv.append(ConvBlock(base_channel*(2**(i)), base_channel*(2**(i+1))))
        self.conv = nn.Sequential(*self.conv)
        
        self.pe = PositionalEncoder(d, (im_size//p)**2)
        self.trans = []
        self.trans0 = transBlock(d, d//base_channel, n_trans)
        for i in range(convN):
            if i == 0:
                self.trans.append(nn.Linear(d,d*(2**(i+1))//c0))
            else:
                self.trans.append(nn.Linear(d*(2**i)//c0,d*(2**(i+1))//c0))
            self.trans.append(transBlock(d*(2**(i+1))//c0, d*(2**(i+1))//(c0*base_channel), n_trans))       
        self.trans = nn.Sequential(*self.trans)

        self.seg_head = nn.Conv2d(base_channel//c0, num_class, (1,1))
        self.out = nn.Sigmoid()
        
    def forward(self, x, convn=True): 

        x = self.conv(x)
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        x = self.pe(x)
        x,sim = self.trans0(x)
        
        for i,l in enumerate(self.trans):
            if i % 2 == 0:
                x = l(x)
            else:
                x,sim = l(x)
        
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=self.n, w=self.n)
        x = einops.rearrange(x, 'b (c p1 p2) n1 n2 -> b c (n1 p1) (n2 p2)', p1=2**self.convN, p2=2**self.convN)
        x = self.seg_head(x)
        x = self.out(x)
        return x,sim

class transBlock_v2(nn.Module):
    def __init__(self, d, heads, n_trans):
        super().__init__()
        layers = []
        self.n_trans = n_trans
        for i in range(n_trans):
            layers.append(EncoderLayer(d, heads))
             
        self.sequential = nn.Sequential(*layers)
    
    def forward(self, x, pos):
        for i in range(self.n_trans):
            x, sim = self.sequential[i](x+pos)
            
        return x, sim
    
class CTS_v2(nn.Module):

    def __init__(self, ini_channel, base_channel, convN, im_size, n_trans, num_class,c0):
        super().__init__()
        self.num_class = num_class
        self.im_size = im_size
        self.convN = convN
        self.n = im_size // (2**convN)
        p = 2**convN
        d = base_channel*(2**(convN))//c0
        self.p = p

        self.conv = []
        self.conv.append(ConvBlock(ini_channel, base_channel))
        for i in range(convN):
            self.conv.append(nn.MaxPool2d(2))
            self.conv.append(ConvBlock(base_channel*(2**(i)), base_channel*(2**(i+1))))
        self.conv = nn.Sequential(*self.conv)
        
        self.linear = []
        for i in range(convN+1):
            self.linear.append(nn.Linear(base_channel*(2**(i)),base_channel*(2**(i))//c0))
        self.linear = nn.Sequential(*self.linear)
        
        self.pe = nn.Parameter(torch.randn(1, (im_size//p)**2, d))
        
        self.trans = []
        for i in range(convN+1):
            if i == convN:
                self.trans.append(transBlock_v2(d*(2**i), d*(2**i)//base_channel, n_trans))
            else:
                self.trans.append(transBlock_v2(d*(2**i), d*(2**i)//base_channel, n_trans))
                self.trans.append(nn.Linear(d*(2**i),d*(2**(i+1))))
        self.trans = nn.Sequential(*self.trans)

        self.seg_head = nn.Conv2d(base_channel//c0, num_class, (1,1))
        self.out = nn.Sigmoid()
        
    def forward(self, x, convn=True): 
        cnn_out = []
        for i,l in enumerate(self.conv):
            x = l(x)
            if i%2 == 0:
                x0 = einops.rearrange(x, 'b c h w -> b h w c')
                x0 = self.linear[i//2](x0)
                x0 = einops.rearrange(x0, 'b (n1 p1) (n2 p2) c -> b (c p1 p2) n1 n2', n1=self.n, n2=self.n)
                x0 = einops.rearrange(x0, 'b c h w -> b (h w) c')
                cnn_out.append(x0)
        
        x = cnn_out[-1]
        pos = self.pe

        for i,l in enumerate(self.trans):
            if i % 2 == 0:
                x,sim = l(x,pos)
            else:
                x = l(x)
                pos = l(pos)
                x = x + (cnn_out[self.convN - 1 - i//2])
        
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=self.n, w=self.n)
        x = einops.rearrange(x, 'b (c p1 p2) n1 n2 -> b c (n1 p1) (n2 p2)', p1=2**self.convN, p2=2**self.convN)
        x = self.seg_head(x)
        x = self.out(x)
        return x, sim
    
    
class attention_mat(nn.Module):
    def __init__(self, heads, d_model):
        super().__init__()
        self.d_head = d_model//heads
        self.norm = nn.LayerNorm(d_model)
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        x0 = self.norm(x)
        k = self.k_linear(x0)
        q = self.q_linear(x0)
        k = einops.rearrange(k, 'b n (h channel) -> b h n channel', channel = self.d_head)
        q = einops.rearrange(q, 'b n (h channel) -> b h n channel', channel = self.d_head)
        att = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(self.d_head)
        att = F.softmax(att, dim=-1)
        
        return att
    
def comp_mask(pred):
    # pred size: (bs,h,ns,dh)
    pred[torch.where(pred < 0.5)] = pred[torch.where(pred < 0.5)]-1
    comp = -torch.matmul(pred, pred.transpose(-2, -1))
    
    return comp

class CTS_CRFMF(nn.Module):
    def __init__(self, ini_channel, base_channel, convN, im_size, crf_heads, n_trans, num_class,c0):
        super().__init__()
        self.num_class = num_class
        self.im_size = im_size
        self.convN = convN
        self.n = im_size // (2**convN)
        p = 2**convN
        d = base_channel*(2**(convN))
        self.p = p
        self.crf_heads = crf_heads

        self.conv = []
        self.conv.append(ConvBlock(ini_channel, base_channel))
        for i in range(convN):
            self.conv.append(nn.MaxPool2d(2))
            self.conv.append(ConvBlock(base_channel*(2**(i)), base_channel*(2**(i+1))))
        self.conv = nn.Sequential(*self.conv)
        
        self.linear = []
        for i in range(convN):
            self.linear.append(nn.Linear(base_channel*(2**(i)),base_channel*(2**(i))//c0))
        self.linear = nn.Sequential(*self.linear)
        
        self.pe = PositionalEncoder(d, (im_size//p)**2)
        self.trans = []
        self.trans0 = transBlock(d, d//(base_channel), n_trans)

        for i in range(convN):
            if i == 0:
                self.trans.append(nn.Linear(d,d*(2**(i+1))//c0))
            else:
                self.trans.append(nn.Linear(d*(2**i)//c0,d*(2**(i+1))//c0))
            if i < convN-1:
                self.trans.append(transBlock(d*(2**(i+1))//c0, d*(2**(i+1))//(c0*base_channel), n_trans)) 
            else:
                self.trans.append(transBlock(d*(2**(i+1))//c0, d*(2**(i+1))//(c0*base_channel), n_trans-1))
                self.trans.append(EncoderLayer_crf(d*(2**(i+1))//c0, crf_heads, self.n, self.p, base_channel//c0))

        self.trans = nn.Sequential(*self.trans)
        self.seg_head = nn.Conv2d(base_channel//c0, num_class, (1,1))
    
        if num_class>1:
            self.out = nn.Softmax(dim=1)
        else:
            self.out = nn.Sigmoid()
            
        comp_weight = (1 - torch.eye(num_class)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.mf_att = attention_mat(crf_heads, d*(2**convN)//c0)
        self.mf_com = nn.Conv3d(num_class, num_class, [1,1,1])
        self.mf_com.weight = nn.Parameter(comp_weight.type(torch.FloatTensor))
        if self.mf_com.bias is not None:
            self.mf_com.bias.data.zero_()

        self.mf_linear = FeedForward(int(p**2//crf_heads))
        
    def forward(self, x, mf_ite=10, MF=True, convn=True): 
        cnn_out = []
        for i,l in enumerate(self.conv):
            x = l(x)
            if i%2 == 0 and i != self.convN*2:
                x0 = einops.rearrange(x, 'b c h w -> b h w c')
                x0 = self.linear[i//2](x0)
                x0 = einops.rearrange(x0, 'b (n1 p1) (n2 p2) c -> b (c p1 p2) n1 n2', n1=self.n, n2=self.n)
                x0 = einops.rearrange(x0, 'b c h w -> b (h w) c')
                cnn_out.append(x0)
        
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        x = self.pe(x)
        x,sim = self.trans0(x)

        for i,l in enumerate(self.trans):
            if i % 2 == 0 and i != len(self.trans)-1:
                x = l(x)
                x = x + (cnn_out[self.convN - 1 - i//2])
            else:
                x, sim = l(x)
        
        sim_mf = self.mf_att(x)
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=self.n, w=self.n)
        x = einops.rearrange(x, 'b (c p1 p2) n1 n2 -> b c (n1 p1) (n2 p2)', p1=2**self.convN, p2=2**self.convN)
        x = self.seg_head(x)
        out = self.out(x)
        
        if not MF:
            return out, sim
        
        u = einops.rearrange(x, 'b c (n1 p1) (n2 p2) -> b c (n1 n2) (p1 p2)  ', p1=2**self.convN, p2=2**self.convN)
        u = einops.rearrange(u, 'b c n (h channel) -> b c h n channel', h = self.crf_heads)
        q = einops.rearrange(out, 'b c (n1 p1) (n2 p2) -> b c (n1 n2) (p1 p2)  ', p1=2**self.convN, p2=2**self.convN)
        q = einops.rearrange(q, 'b c n (h channel) -> b c h n channel', h = self.crf_heads)
        
        # sim_mf = sim
        # sim_mag = torch.max(torch.linalg.vector_norm(sim,dim=-1,keepdim=True),torch.tensor(1e-8))
        # att = torch.matmul(sim, sim.transpose(-2, -1))/torch.matmul(sim_mag,sim_mag.transpose(-2, -1))
        diagonal_mask = (1-torch.eye(sim_mf.size(-1))).to(sim.device)
        att = sim_mf*diagonal_mask
        
        for i in range(mf_ite):
            temp = []
            for j in range(q.size(1)):
                temp.append(torch.matmul(att,q[:,j]))
            q = torch.stack(temp,dim=1)
            q = self.mf_linear(q)
            q = self.mf_com(q)
            q = u - q
            q = self.out(q)
            
        q = einops.rearrange(q, 'b c h n channel -> b c n (h channel)', h = self.crf_heads)
        q = einops.rearrange(q, 'b c (n1 n2) (p1 p2) -> b c (n1 p1) (n2 p2)', p1=2**self.convN, n1=self.n)
            
        return q,sim_mf

    
class CTS_crfLoss(nn.Module):

    def __init__(self, ini_channel, base_channel, convN, im_size, crf_heads, n_trans, num_class,c0):
        super().__init__()
        self.num_class = num_class
        self.im_size = im_size
        self.convN = convN
        self.n = im_size // (2**convN)
        p = 2**convN
        d = base_channel*(2**(convN))
        self.p = p

        self.conv = []
        self.conv.append(ConvBlock(ini_channel, base_channel))
        for i in range(convN):
            self.conv.append(nn.MaxPool2d(2))
            self.conv.append(ConvBlock(base_channel*(2**(i)), base_channel*(2**(i+1))))
        self.conv = nn.Sequential(*self.conv)
        
        self.linear = []
        for i in range(convN):
            self.linear.append(nn.Linear(base_channel*(2**(i)),base_channel*(2**(i))//c0))
        self.linear = nn.Sequential(*self.linear)
        
        self.pe = PositionalEncoder(d, (im_size//p)**2)
        self.trans = []
        self.trans0 = transBlock(d, d//(base_channel), n_trans)

        for i in range(convN):
            if i == 0:
                self.trans.append(nn.Linear(d,d*(2**(i+1))//c0))
            else:
                self.trans.append(nn.Linear(d*(2**i)//c0,d*(2**(i+1))//c0))
            if i < convN-1:
                self.trans.append(transBlock(d*(2**(i+1))//c0, d*(2**(i+1))//(c0*base_channel), n_trans)) 
            else:
                self.trans.append(transBlock(d*(2**(i+1))//c0, d*(2**(i+1))//(c0*base_channel), n_trans-1))
                self.trans.append(EncoderLayer_crf(d*(2**(i+1))//c0, crf_heads, self.n, self.p, base_channel//c0))

        self.trans = nn.Sequential(*self.trans)
        self.seg_head = nn.Conv2d(base_channel//c0, num_class, (1,1))
        self.out = nn.Sigmoid()
        
    def forward(self, x, convn=True): 
        cnn_out = []
        for i,l in enumerate(self.conv):
            x = l(x)
            if i%2 == 0 and i != self.convN*2:
                x0 = einops.rearrange(x, 'b c h w -> b h w c')
                x0 = self.linear[i//2](x0)
                x0 = einops.rearrange(x0, 'b (n1 p1) (n2 p2) c -> b (c p1 p2) n1 n2', n1=self.n, n2=self.n)
                x0 = einops.rearrange(x0, 'b c h w -> b (h w) c')
                cnn_out.append(x0)
        
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        x = self.pe(x)
        x,sim = self.trans0(x)

        for i,l in enumerate(self.trans):
            if i % 2 == 0 and i != len(self.trans)-1:
                x = l(x)
                x = x + (cnn_out[self.convN - 1 - i//2])
            else:
                x, sim = l(x)
        
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=self.n, w=self.n)
        x = einops.rearrange(x, 'b (c p1 p2) n1 n2 -> b c (n1 p1) (n2 p2)', p1=2**self.convN, p2=2**self.convN)
        x = self.seg_head(x)
        x = self.out(x)
        
        return x, sim
    
