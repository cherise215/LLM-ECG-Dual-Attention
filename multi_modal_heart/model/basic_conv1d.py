import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from fastai.layers import *
from fastai.core import *

##############################################################################################################################################
# utility functions

"""
a modularized deep neural network for 1-d signal data, pytorch version
 
Shenda Hong, Mar 2020
"""

import numpy as np
from collections import Counter
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from multi_modal_heart.model.custom_layers.fixable_dropout import BatchwiseDropout

class Bottleneck1d(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, kernel_size=3, downsample=None, act=nn.ReLU):
        super().__init__()
        
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=stride,
                               padding=(kernel_size-1)//2, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = act(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding

    input: (n_sample, in_channels, n_length)
    output: (n_sample, out_channels, (n_length+stride-1)//stride)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):
        
        net = x
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.conv(net)


def _conv1d(in_planes,out_planes,kernel_size=3, stride=1, dilation=1, act="relu", bn=True, drop_p=0):
    lst=[]
    if(drop_p>0):
        lst.append(nn.Dropout(drop_p))
    lst.append(nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, dilation=dilation, bias=not(bn)))
    if(bn):
        lst.append(nn.BatchNorm1d(out_planes))
    if(act=="relu"):
        lst.append(nn.ReLU(True))
    if(act=="elu"):
        lst.append(nn.ELU(True))
    if(act=="prelu"):
        lst.append(nn.PReLU(True))
    return nn.Sequential(*lst)

def _fc(in_planes,out_planes, act="relu", bn=True):
    lst = [nn.Linear(in_planes, out_planes, bias=not(bn))]
    if(bn):
        lst.append(nn.BatchNorm1d(out_planes))
    if(act=="relu"):
        lst.append(nn.ReLU(True))
    if(act=="elu"):
        lst.append(nn.ELU(True))
    if(act=="prelu"):
        lst.append(nn.PReLU(True))
    return nn.Sequential(*lst)

def cd_adaptiveconcatpool(relevant, irrelevant, module):
    mpr, mpi = module.mp.attrib(relevant,irrelevant)
    apr, api = module.ap.attrib(relevant,irrelevant)
    return torch.cat([mpr, apr], 1), torch.cat([mpi, api], 1)
def attrib_adaptiveconcatpool(self,relevant,irrelevant):
    return cd_adaptiveconcatpool(relevant,irrelevant,self)

class AdaptiveConcatPool1d(nn.Module):
    "Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`."
    def __init__(self, sz:Optional[int]=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or 1
        self.ap,self.mp = nn.AdaptiveAvgPool1d(sz), nn.AdaptiveMaxPool1d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)
    def attrib(self,relevant,irrelevant):
        return attrib_adaptiveconcatpool(self,relevant,irrelevant)
    
class SqueezeExcite1d(nn.Module):
    '''squeeze excite block as used for example in LSTM FCN'''
    def __init__(self,channels,reduction=16):
        super().__init__()
        channels_reduced = channels//reduction
        self.w1 = torch.nn.Parameter(torch.randn(channels_reduced,channels).unsqueeze(0))
        self.w2 = torch.nn.Parameter(torch.randn(channels, channels_reduced).unsqueeze(0))

    def forward(self, x):
        #input is bs,ch,seq
        z=torch.mean(x,dim=2,keepdim=True)#bs,ch
        intermed = F.relu(torch.matmul(self.w1,z))#(1,ch_red,ch * bs,ch,1) = (bs, ch_red, 1)
        s=F.sigmoid(torch.matmul(self.w2,intermed))#(1,ch,ch_red * bs, ch_red, 1=bs, ch, 1
        return s*x #bs,ch,seq * bs, ch,1 = bs,ch,seq

def weight_init(m):
    '''call weight initialization for model n via n.appy(weight_init)'''
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    if isinstance(m,SqueezeExcite1d):
        stdv1=math.sqrt(2./m.w1.size[0])
        nn.init.normal_(m.w1,0.,stdv1)
        stdv2=math.sqrt(1./m.w2.size[1])
        nn.init.normal_(m.w2,0.,stdv2)

def create_head1d(nf:int, nc:int, lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5, bn_final:bool=False, bn:bool=True, act="relu", concat_pooling=True):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes; added bn and act here"
    lin_ftrs = [2*nf if concat_pooling else nf, nc] if lin_ftrs is None else [2*nf if concat_pooling else nf] + lin_ftrs + [nc] #was [nf, 512,nc]
    ps = listify(ps)
    if len(ps)==1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [nn.ReLU(inplace=True) if act=="relu" else nn.ELU(inplace=True)] * (len(lin_ftrs)-2) + [None]
    layers = [AdaptiveConcatPool1d() if concat_pooling else nn.MaxPool1d(2), Flatten()]
    for ni,no,p,actn in zip(lin_ftrs[:-1],lin_ftrs[1:],ps,actns):
        layers += bn_drop_lin(ni,no,bn,p,actn)
    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    return nn.Sequential(*layers)
##############################################################################################################################################
# basic convolutional architecture

class basic_conv1d(nn.Sequential):
    '''basic conv1d'''
    def __init__(self, filters=[128,128,128,128],kernel_size=3, stride=2, dilation=1, pool=0, pool_stride=1, squeeze_excite_reduction=0, num_classes=2, input_channels=8, act="relu", bn=True, headless=False,split_first_layer=False,drop_p=0.,lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True):
        layers = []
        if(isinstance(kernel_size,int)):
            kernel_size = [kernel_size]*len(filters)
        for i in range(len(filters)):
            layers_tmp = []
            
            layers_tmp.append(_conv1d(input_channels if i==0 else filters[i-1],filters[i],kernel_size=kernel_size[i],stride=(1 if (split_first_layer is True and i==0) else stride),dilation=dilation,act="none" if ((headless is True and i==len(filters)-1) or (split_first_layer is True and i==0)) else act, bn=False if (headless is True and i==len(filters)-1) else bn,drop_p=(0. if i==0 else drop_p)))
            if((split_first_layer is True and i==0)):
                layers_tmp.append(_conv1d(filters[0],filters[0],kernel_size=1,stride=1,act=act, bn=bn,drop_p=0.))
                #layers_tmp.append(nn.Linear(filters[0],filters[0],bias=not(bn)))
                #layers_tmp.append(_fc(filters[0],filters[0],act=act,bn=bn))
            if(pool>0 and i<len(filters)-1):
                layers_tmp.append(nn.MaxPool1d(pool,stride=pool_stride,padding=(pool-1)//2))
            if(squeeze_excite_reduction>0):
                layers_tmp.append(SqueezeExcite1d(filters[i],squeeze_excite_reduction))
            layers.append(nn.Sequential(*layers_tmp))

        #head
        #layers.append(nn.AdaptiveAvgPool1d(1))    
        #layers.append(nn.Linear(filters[-1],num_classes))
        #head #inplace=True leads to a runtime error see ReLU+ dropout https://discuss.pytorch.org/t/relu-dropout-inplace/13467/5
        self.headless = headless
        if(headless is True):
            head = nn.Sequential(nn.AdaptiveAvgPool1d(1),Flatten())
        else:
            head=create_head1d(filters[-1], nc=num_classes, lin_ftrs=lin_ftrs_head, ps=ps_head, bn_final=bn_final_head, bn=bn_head, act=act_head, concat_pooling=concat_pooling)
        layers.append(head)
        
        super().__init__(*layers)
    
    def get_layer_groups(self):
        return (self[2],self[-1])

    def get_output_layer(self):
        if self.headless is False:
            return self[-1][-1]
        else:
            return None
    
    def set_output_layer(self,x):
        if self.headless is False:
            self[-1][-1] = x
 



class MyResidualBlock(nn.Module):
    def __init__(self,downsample=False,upsample=False,in_ch=256, out_ch =256, kernel_size = 9, norm=nn.BatchNorm1d,groups=1, act = nn.GELU()):
        super(MyResidualBlock,self).__init__()
        assert downsample != upsample
        self.downsample = downsample
        self.upsample = upsample
        self.stride = 2 if self.downsample else 1
        K = kernel_size
        P = (K-1)//2
        if self.upsample:
            self.idfunc_0 = nn.ConvTranspose1d(in_channels=in_ch,out_channels=out_ch,
                                            kernel_size=4,stride=2,padding=1,groups=groups)
            self.idfunc_1 = nn.Conv1d(in_channels=out_ch,
                                      out_channels=out_ch,
                                      kernel_size=1,
                                      bias=True,groups=groups)
            
            self.conv1 = nn.Conv1d(in_channels=out_ch,
                                out_channels=out_ch,
                                kernel_size=K,
                                stride=self.stride,
                                padding=P,
                                bias=False,groups=groups)
        else:
             self.conv1 = nn.Conv1d(in_channels=in_ch,
                                out_channels=out_ch,
                                kernel_size=K,
                                stride=self.stride,
                                padding=P,
                                bias=False,groups=groups)
        self.bn1 = norm(out_ch)

        self.conv2 = nn.Conv1d(in_channels=out_ch,
                               out_channels=out_ch,
                               kernel_size=K,
                               padding=P,
                               bias=False,groups=groups)
        self.bn2 = norm(out_ch)
        self.act = act

        if self.downsample:
            self.idfunc_0 = nn.MaxPool1d(kernel_size=2,stride=2)
            self.idfunc_1 = nn.Conv1d(in_channels=in_ch,
                                      out_channels=out_ch,
                                      kernel_size=1,
                                      bias=True, groups=groups)
        if not self.downsample and not self.upsample:
            self.idfunc_1 = nn.Conv1d(in_channels=in_ch,
                                      out_channels=out_ch,
                                      kernel_size=1,
                                      bias=True, groups=groups)
        

    def forward(self, x):
        identity = x
        if self.upsample:
            x = self.idfunc_0(x)
            x = self.idfunc_1(x)
            identity = x
       
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        if self.downsample:
            identity = self.idfunc_0(identity)
            identity = self.idfunc_1(identity)
        if not self.upsample and not self.downsample:
            identity = self.idfunc_1(identity)
        out = x+identity
        out = self.act(out) ## for ecg recon maybe not that useful...
        return out
    

class MyResidualBlock2D(nn.Module):
    """_summary_
    only use this for 2d data and conv over the second dimension only
    Args:
        nn (_type_): _description_
    """
    def __init__(self,downsample=False,upsample=False,in_ch=256, out_ch =256, kernel_size = 9, norm=nn.BatchNorm2d,groups=1, act = nn.GELU()):
        super(MyResidualBlock2D,self).__init__()
        assert downsample != upsample or (downsample is False and upsample is False)
        self.downsample = downsample
        self.upsample = upsample
        self.stride = 2 if self.downsample else 1
        K = kernel_size
        P = (K-1)//2
        if self.upsample:
            self.idfunc_0 = nn.ConvTranspose2d(in_channels=in_ch,out_channels=out_ch,kernel_size=(1,4),stride=(1,2),padding=(0,1),groups=groups)
            self.idfunc_1 = nn.Conv2d(in_channels=out_ch,
                                      out_channels=out_ch,
                                      kernel_size=1,
                                      bias=False,groups=groups)
            
            self.conv1 = nn.Conv2d(in_channels=out_ch,
                                out_channels=out_ch,
                                kernel_size=(1,K),
                                stride=(1,1),
                                padding=(0,P),
                                bias=False,groups=groups)
        else:
             self.conv1 = nn.Conv2d(in_channels=in_ch,
                                out_channels=out_ch,
                                kernel_size=(1,K),
                                stride=(1,self.stride),
                                padding=(0,P),
                                bias=False,groups=groups)
        self.bn1 = norm(out_ch)

        self.conv2 = nn.Conv2d(in_channels=out_ch,
                                out_channels=out_ch,
                                kernel_size=(1,K),
                                padding=(0,P),
                                bias=False,groups=groups)
        self.bn2 = norm(out_ch)
        self.act = act

        if self.downsample:
            self.idfunc_0 = nn.MaxPool2d(kernel_size=(1,2),stride=(1,2))
            self.idfunc_1 = nn.Conv2d(in_channels=in_ch,
                                      out_channels=out_ch,
                                      kernel_size=1,
                                      bias=True, groups=groups)
        if not self.downsample and not self.upsample:
            self.idfunc_1 = nn.Conv2d(in_channels=in_ch,
                                      out_channels=out_ch,
                                      kernel_size=1,
                                      bias=True, groups=groups)

    def forward(self, x):
        identity = x

        if self.upsample:
            x = self.idfunc_0(x)
            x = self.idfunc_1(x)
            identity = x
       
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        if self.downsample:
            identity = self.idfunc_0(identity)
            identity = self.idfunc_1(identity)
        if not self.upsample and not self.downsample:
            identity = self.idfunc_1(identity)
        out = x+identity
        # print ("output shape", x.shape)
        out = self.act(out) ## for ecg recon maybe not that useful...
        return out
    
class AttentionPool1d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None, dropout_rate: float = 0,apply_batchwise_dropout: bool =False):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim+ 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.dropout_p  = dropout_rate
        self.apply_batchwise_dropout = apply_batchwise_dropout

    def forward(self, x):
        x = x.permute(2, 0, 1)  # NCL -> (L)NC

        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (L)NC
        # print (x.shape)
        # print (self.positional_embedding[:, None, :].to(x.dtype).shape)
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        if not self.apply_batchwise_dropout:
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        else:
            x = BatchwiseDropout(p=self.dropout_p,batch_first=False)(x)


        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            dropout_p=0,
            add_zero_attn=False,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)
    

    
class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None, dropout_rate: float = 0):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=None,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)
    

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
      
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
# class Bottle2neck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, kernel_size=3,padding= 1,downsample=None, baseWidth=26, scale = 4, stype='normal'):
#         """ Constructor
#         Args:
#             inplanes: input channel dimensionality
#             planes: output channel dimensionality
#             stride: conv stride. Replaces pooling layer.
#             downsample: None when stride = 1
#             baseWidth: basic width of conv3x3
#             scale: number of scale.
#             type: 'normal': normal set. 'stage': first block of a new stage.
#         """
#         super(Bottle2neck, self).__init__()

#         width = int(math.floor(planes * (baseWidth/64.0)))
#         self.conv1 = nn.Conv1d(inplanes, width*scale, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(width*scale)
        
#         if scale == 1:
#           self.nums = 1
#         else:
#           self.nums = scale -1
#         if stype == 'stage':
#             self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride = stride, padding=padding)
#         convs = []
#         bns = []
#         for i in range(self.nums):
#           convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, stride = stride, padding=padding, bias=False))
#           bns.append(nn.BatchNorm1d(width))
#         self.convs = nn.ModuleList(convs)
#         self.bns = nn.ModuleList(bns)

#         self.conv3 = nn.Conv1d(width*scale, planes * self.expansion, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm1d(planes * self.expansion)

#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stype = stype
#         self.scale = scale
#         self.width  = width

#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         spx = torch.split(out, self.width, 1)
#         for i in range(self.nums):
#           if i==0 or self.stype=='stage':
#             sp = spx[i]
#           else:
#             sp = sp + spx[i]
#           sp = self.convs[i](sp)
#           sp = self.relu(self.bns[i](sp))
#           if i==0:
#             out = sp
#           else:
#             out = torch.cat((out, sp), 1)
#         if self.scale != 1 and self.stype=='normal':
#           out = torch.cat((out, spx[self.nums]),1)
#         elif self.scale != 1 and self.stype=='stage':
#           out = torch.cat((out, self.pool(spx[self.nums])),1)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)
#         return out
    
class Res2Net(nn.Module):
    ## reference: https://github.com/LeiJiangJNU/Res2Net/blob/master/res2net.py

    def __init__(self, block, layers, baseWidth = 26, scale = 4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample=downsample, 
                        stype='stage', baseWidth = self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth = self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x     
class MultiScaleBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dilation, groups=1, act = nn.GELU(),norm=nn.InstanceNorm1d):
        super().__init__()
        assert hidden_size % 4 == 0 and input_size%groups == 0 and input_size>=groups, "hidden_size must be divisible by 4 and input_size must be divisible by groups"
        self.input_conv = nn.Conv1d(input_size, hidden_size, 1, dilation=1, padding=0,groups=groups)
        self.filters = nn.ModuleList([
            nn.Conv1d(input_size, hidden_size, 3, dilation=dilation, padding=dilation,groups=groups),
            nn.Conv1d(input_size, hidden_size, 5, dilation=dilation, padding=2*dilation,groups=groups),
            nn.Conv1d(input_size, hidden_size, 9, dilation=dilation, padding=4*dilation,groups=groups),
            nn.Conv1d(input_size, hidden_size, 15, dilation=dilation, padding=7*dilation,groups=groups),
        ])
        
        self.conv_1 = nn.Conv1d(hidden_size*4, hidden_size*4, 9, padding=4,bias=False)
        self.norm = norm(hidden_size*2)
        self.conv_2 = nn.Conv1d(hidden_size*4, hidden_size, 1, padding=0,bias=True)
        self.act = act
    
    def forward(self, x):
        residual = self.input_conv(x)
        
        filts = []
        for layer in self.filters:
            filts.append(layer(x))
            
        filts = torch.cat(filts, dim=1)
        # print(filts.shape)
        
        nfilts, filts = self.conv_1(filts).chunk(2, dim=1)
        
        filts = self.act(torch.cat([self.norm(nfilts), filts], dim=1))
        filts = self.act(self.conv_2(filts))
        # print(filts.shape)
        return filts + residual
    
class MultiScaleBlock2d(nn.Module):
    def __init__(self, input_size, hidden_size, dilation,act = nn.GELU(),norm = nn.InstanceNorm2d):
        super().__init__()
        self.input_conv = nn.Conv2d(input_size, hidden_size, 1, dilation=1, padding=0,groups=1)
        self.filters = nn.ModuleList([
            nn.Conv2d(input_size, hidden_size//4, (1,3), dilation=(1,dilation), padding=(0,dilation)),
            nn.Conv2d(input_size, hidden_size//4, (1,5), dilation=(1,dilation), padding=(0,2*dilation)),
            nn.Conv2d(input_size, hidden_size//4, (1,9), dilation=(1,dilation), padding=(0,4*dilation)),
            nn.Conv2d(input_size, hidden_size//4, (1,15), dilation=(1,dilation), padding=(0,7*dilation)),
        ])
        
        self.conv_1 = nn.Conv2d(hidden_size, hidden_size, (1,9), padding=(0,4),bias=False)
        self.norm =nn.InstanceNorm2d(hidden_size//2)
        
        self.conv_2 = nn.Conv2d(hidden_size, hidden_size, (1,9), padding=(0,4),bias=True)
        self.act = act

        
    def forward(self, x):
        residual = self.input_conv(x)
        
        filts = []
        for layer in self.filters:
            filts.append(layer(x))
            
        filts = torch.cat(filts, dim=1)
        # print(filts.shape)
        
        nfilts, filts = self.conv_1(filts).chunk(2, dim=1)
        
        filts =self.act(torch.cat([self.norm(nfilts), filts], dim=1))
        filts =self.act(self.conv_2(filts))
        # print(filts.shape)
        return filts + residual
    
############################################################################################
# convenience functions for basic convolutional architectures
def fcn(filters=[128]*5,num_classes=2,input_channels=8):
    filters_in = filters + [num_classes]
    return basic_conv1d(filters=filters_in,kernel_size=3,stride=1,pool=2,pool_stride=2,input_channels=input_channels,act="relu",bn=True,headless=True)

def fcn_wang(num_classes=2,input_channels=8,lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True):
    return basic_conv1d(filters=[128,256,128],kernel_size=[8,5,3],stride=1,pool=0,pool_stride=2, num_classes=num_classes,input_channels=input_channels,act="relu",bn=True,lin_ftrs_head=lin_ftrs_head, ps_head=ps_head, bn_final_head=bn_final_head, bn_head=bn_head, act_head=act_head, concat_pooling=concat_pooling)

def schirrmeister(num_classes=2,input_channels=8,lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True):
    return basic_conv1d(filters=[25,50,100,200],kernel_size=10, stride=3, pool=3, pool_stride=1, num_classes=num_classes, input_channels=input_channels, act="relu", bn=True, headless=False,split_first_layer=True,drop_p=0.5,lin_ftrs_head=lin_ftrs_head, ps_head=ps_head, bn_final_head=bn_final_head, bn_head=bn_head, act_head=act_head, concat_pooling=concat_pooling)

def sen(filters=[128]*5,num_classes=2,input_channels=8,squeeze_excite_reduction=16,drop_p=0.,lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True):
    return basic_conv1d(filters=filters,kernel_size=3,stride=2,pool=0,pool_stride=0,input_channels=input_channels,act="relu",bn=True,num_classes=num_classes,squeeze_excite_reduction=squeeze_excite_reduction,drop_p=drop_p,lin_ftrs_head=lin_ftrs_head, ps_head=ps_head, bn_final_head=bn_final_head, bn_head=bn_head, act_head=act_head, concat_pooling=concat_pooling)

def basic1d(filters=[128]*5,kernel_size=3, stride=2, dilation=1, pool=0, pool_stride=1, squeeze_excite_reduction=0, num_classes=2, input_channels=8, act="relu", bn=True, headless=False,drop_p=0.,lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True):
    return basic_conv1d(filters=filters,kernel_size=kernel_size, stride=stride, dilation=dilation, pool=pool, pool_stride=pool_stride, squeeze_excite_reduction=squeeze_excite_reduction, num_classes=num_classes, input_channels=input_channels, act=act, bn=bn, headless=headless,drop_p=drop_p,lin_ftrs_head=lin_ftrs_head, ps_head=ps_head, bn_final_head=bn_final_head, bn_head=bn_head, act_head=act_head, concat_pooling=concat_pooling)
