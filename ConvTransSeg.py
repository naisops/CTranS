# -*- coding: utf-8 -*-
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
        # x (bs, seq_length, (p * p))
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

        scores = attention(q, k, v, self.d_head, self.dropout)
        # scores shape: (bs x h x sl x d_k)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        
        output = self.out(concat)
        # output shape: (bs x sl x d_model)
    
        return output
       
def attention(q, k, v, d_k, dropout=None):  
    # q,k,v shape: (bs x h x sl x d_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    # output shape: (bs x h x sl x d_k)
    return output

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
        temp = self.attn(x2,x2,x2)
        x = x + self.dropout_1(temp)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

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
            x = self.sequential[i](x)
            
        return x
    
    
class ConvBlock(nn.Module):
    
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

        for i in range(convN):
            if i == 0:
                self.trans.append(nn.Linear(d,d*(2**(i+1))//c0))
            else:
                self.trans.append(nn.Linear(d*(2**i)//c0,d*(2**(i+1))//c0))
            self.trans.append(transBlock(d*(2**(i+1))//c0, d*(2**(i+1))//(c0*base_channel), n_trans)) 
        self.trans = nn.Sequential(*self.trans)

        self.seg_head = nn.Conv2d(base_channel//c0, num_class, (1,1))
        if num_class < 2:
            self.out = nn.Sigmoid()
        
    def forward(self, x): 
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
        x = self.trans0(x)

        for i,l in enumerate(self.trans):
            if i % 2 == 0:
                x = l(x)
                x = x + (cnn_out[self.convN - 1 - i//2])
            else:
                x = l(x)
        
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=self.n, w=self.n)
        x = einops.rearrange(x, 'b (c p1 p2) n1 n2 -> b c (n1 p1) (n2 p2)', p1=2**self.convN, p2=2**self.convN)
        x = self.seg_head(x)
        if self.num_class < 2:
            x = self.out(x)
        return x
