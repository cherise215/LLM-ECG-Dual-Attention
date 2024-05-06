

import torch
import torch.nn.functional as F
import torch.nn as nn 
import math
import sys
sys.path.append(".")
import numpy as np
from multi_modal_heart.model.resnet1d import resnet1d101
from multi_modal_heart.model.xresnet1d import xresnet1d101
from multi_modal_heart.model.basic_conv1d import MyResidualBlock, MultiScaleBlock,AttentionPool1d,AttentionPool2d
from multi_modal_heart.model.custom_layers.pos_embed import SineActivation,get_2d_sincos_pos_embed_from_grid,get_1d_sincos_pos_embed_from_grid
from multi_modal_heart.model.d_linearnet import SeriesDecomp
from multi_modal_heart.model.custom_layers.concat_pooling import ConcatPooling1DLayer
from multi_modal_heart.model.ecg_net_attention import ECGDecoder
from multi_modal_heart.model.custom_layers.fixable_dropout import BatchwiseDropout
class ISIBrnoAIMTNet(nn.Module):
    def __init__(self,in_channels=12,nOUT=24, n_feature=256, norm= nn.BatchNorm2d):
        super(ISIBrnoAIMTNet,self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=n_feature,
                              kernel_size=(1,15),
                              padding=(0,7),
                              stride=(1,2),
                              bias=False)
        self.bn = nn.BatchNorm2d(n_feature)
        self.rb_0 = MyResidualBlock(downsample=True, in_ch=n_feature, out_ch=n_feature, norm=norm)
        self.rb_1 = MyResidualBlock(downsample=True,in_ch=n_feature, out_ch=n_feature,norm=norm)
        self.rb_2 = MyResidualBlock(downsample=True,in_ch=n_feature, out_ch=n_feature, norm=norm)
        self.rb_3 = MyResidualBlock(downsample=True,in_ch=n_feature, out_ch=n_feature, norm=norm)
        self.rb_4 = MyResidualBlock(downsample=True,in_ch=n_feature, out_ch=n_feature, norm=norm)

        self.mha = nn.MultiheadAttention(n_feature,num_heads = 8)
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.fc_1 = nn.Linear(n_feature + in_channels,nOUT)
        self.ch_fc1 = nn.Linear(nOUT,n_feature)
        self.ch_bn = nn.BatchNorm1d(n_feature)
        self.ch_fc2 = nn.Linear(n_feature,nOUT)

    def forward(self, x,l):
        x = F.leaky_relu(self.bn(self.conv(x)))

        x = self.rb_0(x)
        x = self.rb_1(x)
        x = self.rb_2(x)
        x = self.rb_3(x)
        x = self.rb_4(x)

        x = F.dropout(x,p=0.5,training=self.training)

        x = x.squeeze(2).permute(2,0,1)
        x,s = self.mha(x,x,x)
        x = x.permute(1,2,0)
        x = self.pool(x).squeeze(2)
        x = torch.cat((x,l),dim=1)

        x = self.fc_1(x)
        p = x.detach()
        p = F.leaky_relu(self.ch_bn(self.ch_fc1(p)))
        p = torch.sigmoid(self.ch_fc2(p))
        return x,p

class BenchmarkClassifier(nn.Module):
    '''
    this is the classifier used in the PTBXL benchmarking paper
    

    (2): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.25, inplace=False)
    (4): Linear(in_features=512, out_features=128, bias=True)
    (5): ReLU(inplace=True)
    (6): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): Dropout(p=0.5, inplace=False)
    (8): Linear(in_features=128, out_features=71, bias=True)
  '''
    def __init__(self, input_size=512, hidden_size=128, output_size=5, dropout_rate=0.5,act=nn.ReLU, last_act=None,batchwise_dropout=False):
        super(BenchmarkClassifier, self).__init__()
        # Create a list to hold the layers
        layers = []
        # Add the input layer
        layers.append(nn.BatchNorm1d(input_size))
        if batchwise_dropout:
            layers.append(BatchwiseDropout(p=0.25))
        else: layers.append(nn.Dropout(0.25))
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(act())
        layers.append(nn.BatchNorm1d(hidden_size))
        if hidden_size>100:
            if batchwise_dropout:
                layers.append(BatchwiseDropout(p=0.5))
            else: layers.append(nn.Dropout(0.5))
        else: layers.append(nn.Identity())
        layers.append(nn.Linear(hidden_size, output_size))
        if last_act is not None:
            layers.append(last_act())
        # Create the sequential model
        self.model = nn.Sequential(*layers)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.model[2].weight)
        nn.init.xavier_uniform_(self.model[6].weight)

    def forward(self,x):
        return self.model(x)
    
    def get_features(self,x):
        '''
        return the features before the last layer
        '''
        x = self.model[0](x)
        x = self.model[1](x)
        x = self.model[2](x)
        x = self.model[3](x)
        x = self.model[4](x)
        x = self.model[5](x)
        return x
         


class ClassifierMLP(nn.Module):
    def __init__(self, input_size=512, hidden_sizes=[256,256], output_size=2, dropout_rate=0.5,act=nn.GELU, last_act=None):
        '''
        input_size: int, input dim
        hidden_sizes: <list of int>, dim of intermediate layers
        output_size: int final,
        dropout_rate: float, [0,1), dropout rate
        '''
        super(ClassifierMLP, self).__init__()
        # Create a list to hold the layers
        layers = []
        # Add the input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(act())
        layers.append(nn.Dropout(dropout_rate))
        
        # Add the hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(act())
            layers.append(nn.Dropout(dropout_rate))
        
        # Add the output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        if last_act is not None:
            layers.append(last_act())
        # Create the sequential model
        self.model = nn.Sequential(*layers)
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    
    def forward(self, x):
        return self.model(x)





class ECG_encoder(nn.Module):
    def __init__(self,in_channels=12,ECG_length=1024,embedding_dim=256,output_dim=512,norm=nn.InstanceNorm1d, downsample_factor=5, if_VAE=False, mha=True, groups=1,downscale_feature=1,act=nn.GELU(), no_linear=False, last_act=None):
        super(ECG_encoder, self).__init__()
        spacial_dim = ECG_length//(2**downsample_factor)
        self.downsample_factor = downsample_factor
        assert ECG_length%(2**downsample_factor)==0, "ECG length should be divisible by 32"
        print (spacial_dim)
        self.spacial_dim = spacial_dim
        n_features = [in_channels,in_channels*2//downscale_feature,in_channels*4//downscale_feature,
                      in_channels*8//downscale_feature,256//downscale_feature,embedding_dim]
        n_features[0]= in_channels
        self.latent_code_dim=output_dim
        self.act = act
        self.groups = groups
        self.input_conv = nn.Sequential(
                    nn.Conv1d(in_channels,in_channels,kernel_size=5, stride=1, padding=2, bias=False, groups=self.groups),
                    nn.InstanceNorm1d(in_channels),
                    self.act,  
                    MultiScaleBlock(in_channels,in_channels,dilation =2,norm = nn.InstanceNorm1d, act=self.act,groups=self.groups),
                    MultiScaleBlock(in_channels,in_channels,dilation =2,norm = nn.InstanceNorm1d, act=self.act,groups=self.groups),
                    )
        list_encoder_layers = []
        for i in range(downsample_factor):
            list_encoder_layers.append(
                MyResidualBlock(downsample=True,in_ch=n_features[i],out_ch= n_features[i+1],norm=nn.BatchNorm1d,kernel_size=5,act = act))
          
        assert n_features[i+1]==embedding_dim, "the last layer should be the embedding dim"
        self.encoder = nn.Sequential(*list_encoder_layers)
        if mha:
            print ('multihead attention pooling is applied')
            self.pool = nn.Sequential(
                    # nn.Dropout(0.5),
                    AttentionPool1d(spacial_dim=spacial_dim, embed_dim=embedding_dim,num_heads=8, output_dim=output_dim,dropout_rate=0.2)
                    )
        else: 
            ## use adaptive pooling instead
            self.pool = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Conv1d(embedding_dim,output_dim,kernel_size=1, stride=1, padding=0, bias=True),
                    nn.AdaptiveAvgPool1d(output_size=1),
                    nn.Flatten()
                    )
        self.if_VAE = if_VAE
        ## if VAE:
        self.no_linear = no_linear
        if not no_linear:
            self.linear_1 = nn.Linear(output_dim,output_dim)
                            

            if self.if_VAE:
                self.encFC1 = nn.Linear(output_dim, output_dim) ## mean 
                self.encFC2 = nn.Linear(output_dim, output_dim) ## variance
            else:
                self.linear = nn.Linear(output_dim,output_dim)
        else:
            self.linear_1 = nn.Identity()
            self.linear = nn.Identity()
            self.encFC1 = nn.Identity()
            self.encFC2 = nn.Identity()
        self.last_act = last_act
        if last_act is not None:
            self.last_act = last_act()
        else:
            self.last_act = nn.Identity()
    
    def forward(self,x, mask=None):
        # x = self.conv_down(x)
        x = self.encoder(x)
        x = self.pool(x)
        x = self.linear_1(x)
        ## 
        if self.if_VAE:
            x_mean = self.encFC1(x)
            x_logvar = self.encFC2(x)
            return x_mean, x_logvar
        else:
            x = self.linear(x)
            x = self.last_act(x)
        return x
    
    def get_features_after_pooling(self,x, mask=None):
        ## before linear layer
        x = self.encoder(x)
        ## embedding dim 256
        x_avg =nn.AdaptiveAvgPool1d(output_size=1)(x)
        x_avg = torch.squeeze(x_avg)
        x = self.pool(x)
        x = torch.cat((x,x_avg),dim=1)
        ## add average pool and max pool
        return x

    
## encoder v2 with layer norm
class ECG_encoder_MLP(nn.Module):
    def __init__(self,in_channels=12,ECG_length=1024,embedding_dim=256,output_dim=512,norm=nn.InstanceNorm1d, downsample_factor=5, if_VAE=False, mha=True, groups=1,downscale_feature=1,act=nn.GELU(), mlp_layers=2):
        super(ECG_encoder_MLP, self).__init__()
        spacial_dim = ECG_length//(2**downsample_factor)
        assert ECG_length%(2**downsample_factor)==0, "ECG length should be divisible by 32"
        print (spacial_dim)
        self.spacial_dim = spacial_dim
        n_features = [in_channels,in_channels*2//downscale_feature,in_channels*4//downscale_feature,
                      in_channels*8//downscale_feature,256//downscale_feature,embedding_dim]
        n_features[0]= in_channels
        self.latent_code_dim=output_dim
    
        self.act = act
        
        self.groups = groups
      
        self.input_conv = nn.Sequential(
                        nn.Conv1d(in_channels,in_channels,kernel_size=5, stride=1, padding=2, bias=False, groups=self.groups),                     
                        nn.InstanceNorm1d(in_channels),
                        self.act,  
                        MultiScaleBlock(in_channels,in_channels,dilation =2,norm = nn.InstanceNorm1d, act=self.act,groups=self.groups),
                        MultiScaleBlock(in_channels,in_channels,dilation =2,norm = nn.InstanceNorm1d, act=self.act,groups=self.groups),
                        )
      

        list_encoder_layers = []
        for i in range(downsample_factor):
            list_encoder_layers.append(
                MyResidualBlock(downsample=True,in_ch=n_features[i],out_ch= n_features[i+1],norm=nn.BatchNorm1d,kernel_size=5,act = act))
          
        assert n_features[i+1]==embedding_dim, "the last layer should be the embedding dim"
        self.encoder = nn.Sequential(*list_encoder_layers)
        if mha:
            print ('multihead attention pooling is applied')
            self.pool = nn.Sequential(
                    AttentionPool1d(spacial_dim=spacial_dim, embed_dim=embedding_dim,num_heads=8, output_dim=output_dim,dropout_rate=0.2)
                    )
        else: 
            ## use adaptive pooling instead
            self.pool = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Conv1d(embedding_dim,output_dim,kernel_size=1, stride=1, padding=0, bias=True),
                    nn.AdaptiveAvgPool1d(output_size=1),
                    nn.Flatten()
                    )
        self.if_VAE = if_VAE
        ## if VAE:
        self.linear_1 = nn.Sequential(nn.Linear(output_dim , output_dim, bias=False),
                          nn.LayerNorm(output_dim),
                          nn.GELU(),
                          nn.Dropout (0.15))
        self.mlp = nn.ModuleList ([
                          nn.Sequential(nn.Linear(output_dim , output_dim, bias=False),
                          nn.LayerNorm(output_dim),
                          nn.GELU(),
                          nn.Dropout (0.15)) for _ in  range (mlp_layers)]
                          )
    
        if self.if_VAE:
            self.encFC1 = nn.Linear(output_dim, output_dim) ## mean 
            self.encFC2 = nn.Linear(output_dim, output_dim) ## variance
        else:
            self.linear = nn.Linear(output_dim,output_dim)

    def forward(self,x):
        # x = self.conv_down(x)
        x = self.encoder(x)
        x = self.pool(x)
        x= self.linear_1(x)
        ## residual MLP module
        residual = x
        for  res_block  in range(len(self.mlp)):
            x = self.mlp[res_block ](x)
            x +=  residual
            residual = x
        ## 
        if self.if_VAE:
            x_mean = self.encFC1(x)
            x_logvar = self.encFC2(x)
            return x_mean, x_logvar
        else:
            x = self.linear(x)
        return x



class ECG_ResNetencoder(nn.Module):
    def __init__(self,in_channels=12,ECG_length=1024,embedding_dim=256,output_dim=512,norm=nn.InstanceNorm1d, downsample_factor=5, 
                 group_conv=False, if_VAE=False, mha=False, no_linear=True, last_act=None, apply_batchwise_dropout=False):
        super(ECG_ResNetencoder, self).__init__()
        spacial_dim = ECG_length//(2**downsample_factor)
        assert ECG_length%(2**(downsample_factor))==0, "ECG length should be divisible by 32"
        print (spacial_dim)
        self.spacial_dim = spacial_dim
        self.input_channels = in_channels
        self.latent_code_dim=output_dim
        self.group_conv = group_conv
        self.last_act = last_act
        if group_conv:
             self.input_conv = nn.Sequential(
                    nn.Conv1d(in_channels,in_channels,kernel_size=5, stride=1, padding=2, bias=False, groups=in_channels),
                    nn.LayerNorm([in_channels,ECG_length]),
                    # nn.GELU(),  
                    # MultiScaleBlock(in_channels,in_channels,dilation =2,norm = nn.InstanceNorm1d, act= nn.GELU(),groups=in_channels),
                    # MultiScaleBlock(in_channels,in_channels,dilation =2,norm = nn.InstanceNorm1d, act= nn.GELU(),groups=in_channels),
                    )
        else:
            self.input_conv=nn.Identity()
        self.mha = mha
        if self.mha:
            self.encoder =xresnet1d101(num_classes=output_dim,input_channels=in_channels,kernel_size=5,ps_head=0.5,lin_ftrs_head=[128], create_head=False)
            self.pool  = AttentionPool1d(spacial_dim=spacial_dim, embed_dim=embedding_dim,num_heads=8, output_dim=embedding_dim,dropout_rate=0.2,apply_batchwise_dropout=apply_batchwise_dropout)

        else: 
            self.encoder = nn.Sequential(
            xresnet1d101(num_classes=output_dim,input_channels=in_channels,kernel_size=5,ps_head=0.5,lin_ftrs_head=[128], create_head=False),
            ConcatPooling1DLayer(),
            nn.Flatten()
            )
        self.if_VAE = if_VAE
       

        ## if VAE:
        if not no_linear:
            if self.if_VAE:
                self.encFC1 = nn.Linear(embedding_dim*2, output_dim) ## mean 
                self.encFC2 = nn.Linear(embedding_dim*2, output_dim) ## variance
            else:
                self.linear = nn.Linear(embedding_dim*2,output_dim)
        else:
            self.linear = nn.Identity()
            self.encFC1 = nn.Identity()
            self.encFC2 = nn.Identity()
        
        if last_act is not None:
            self.last_act = last_act()
        else:
            self.last_act = nn.Identity()


    def get_features_after_pooling(self,x,mask=None):
        ## before linear layer
        if self.mha:
            x = self.input_conv(x)
            x = self.encoder(x)
            attention_pool = self.pool(x)
            max_pool = nn.AdaptiveMaxPool1d(output_size=1)(x)
            max_pool = torch.squeeze(max_pool)
            x = torch.cat((attention_pool,max_pool),dim=1)
        else:
            x = self.input_conv(x)
            x = self.encoder(x)
        return x
    def forward(self,x, mask=None):
        x = self.get_features_after_pooling(x, mask)
        if self.if_VAE:
            x_mean = self.encFC1(x)
            x_logvar = self.encFC2(x)
            return x_mean, x_logvar
        else:
            x = self.linear(x)
            x = self.last_act(x)
        return x
    
class ECG_decoder(nn.Module):
    def __init__(self,in_channels=64,base_spatial = 32, output_dim=12,norm=nn.BatchNorm1d,upsample_factor=5,
                    n_features = [12,64,32,32,16,16],groups=1,act=nn.GELU()):
        super(ECG_decoder, self).__init__()
        # n_features = [output_dim]*(upsample_factor+1)
        self.linear = nn.Sequential(
                nn.Linear(in_channels,in_channels*base_spatial), 
                # nn.GELU(),
        )
        self.base_spatial = base_spatial
        list_decoder_layers = []
        list_decoder_layers.append(
                            MyResidualBlock(upsample=True,in_ch=in_channels,out_ch=n_features[0],norm=nn.InstanceNorm1d,kernel_size=5,act =act))
       
        for i in range(upsample_factor-1):
            list_decoder_layers.append(
                            MyResidualBlock(upsample=True,in_ch=n_features[i],out_ch=n_features[i+1],norm=nn.InstanceNorm1d,kernel_size=5,act=act))          
        self.decoder = nn.Sequential(*list_decoder_layers)
        self.final_conv = nn.Sequential(
                    nn.Conv1d(n_features[i+1],output_dim,kernel_size=1, stride=1, padding=0, bias=False),
                    nn.InstanceNorm1d(output_dim),
                    nn.GELU(),
                    nn.Conv1d(output_dim,output_dim,kernel_size=1, stride=1, padding=0, bias=True, groups=groups),
                    nn.InstanceNorm1d(output_dim),
                    )


    def forward(self,x):
        x = self.linear(x)
        x= x.view(x.shape[0],-1,self.base_spatial)
        x = self.decoder(x)
        x = self.final_conv(x)
        return x
       
class LinearDecoder(nn.Module):
    def __init__(self, input_feature_dim, output_length, out_ch=1,decompose=False):
        super().__init__()

        self.out_ch = out_ch
        self.decompose = decompose
        self.linear1 = nn.ModuleList()
        self.linear2 = nn.ModuleList()
        if decompose:
            assert input_feature_dim%2==0
            input_feature_dim = input_feature_dim//2
            for i in range(self.out_ch):
                self.linear1.append(nn.Linear(input_feature_dim, output_length))
                self.linear2.append(nn.Linear(input_feature_dim, output_length))
        else:
            for i in range(self.out_ch):
                self.linear1.append(nn.Linear(input_feature_dim, output_length))

        ## sample code
        # input_data = torch.randn(5,64)
        # model = LinearDecoder(input_feature_dim =64, output_length=1024,out_ch=12,decompose=True)
        # output = model(input_data)
        # print (output.shape)
            
    def forward(self, x):
        '''
        x: [Batch, feature_dim]
        return: [Batch, out_ch, output_length]
        '''
        output = []
        size = x.size(1)
        for i in range(self.out_ch):
            if self.decompose: 
                out = self.linear1[i](x[:,:size//2])
                output_2 = self.linear2[i](x[:,size//2:])
                out = out + output_2
            else:
                out = self.linear1[i](x)
            output.append(out)
        output = torch.stack(output, dim=1)
        return output


class ECGAE(nn.Module):
    def __init__(self,
                    ## ECG encoder config:
                    #------ encoder config--------#
                    encoder_type="ms_resnet", # optional canditate = "ms",
                    decoder_type ="ms_resnet",
                    in_channels=12,
                    ECG_length=1024,embedding_dim = 64,
                    encoder_down_sample_layers= 5,
                    latent_code_dim = 512,
                    encoder_mha = False,
                    if_VAE =False,
                    groups =1,
                    no_linear = True,
                    downscale_feature=1,
                    encoder_lact_act =None,
                    apply_batchwise_dropout=False,
                    # ------ time2vec config--------#
                    add_time = False,
                    time_dim = 4,
                    apply_method = "time2vec", ## None, or "time2vec" or "fixed_positional_encoding"
                    #------ decoder config--------#
                    decoder_outdim= 24,
                    decoder_upsample_layers= 5,
                    base_feature_dim=4,

                    ## general config:
                    norm=nn.BatchNorm1d, 
                    act = nn.GELU(),
                    with_positional_encoding=False):
        super(ECGAE, self).__init__()
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.encoder_lact_act = encoder_lact_act
        self.ECG_length = ECG_length
        self.base_feature_dim = base_feature_dim
        self.no_linear = no_linear
        self.add_time = add_time
        self.apply_method = apply_method
        self.lead_channels = in_channels
        self.time_dim = time_dim if  self.apply_method is not None else 2 ## [12, 1024, 2], first dim is lead, second dim is time
        self.latent_code_dim =latent_code_dim
        self.embedding_dim  =embedding_dim
        self.norm = norm
        self.groups = groups
        self.downscale_feature =downscale_feature
        self.encoder_down_sample_layers = encoder_down_sample_layers
        self.encoder_mha = encoder_mha
        self.decoder_outdim = decoder_outdim
        self.decoder_upsample_layers = decoder_upsample_layers
        self.if_VAE = if_VAE
        self.with_positional_encoding = with_positional_encoding
        self.apply_batchwise_dropout =apply_batchwise_dropout
        
        if add_time:
            if self.apply_method == "time2vec" or self.apply_method == "FPE2vec":
                ## learnable time encoder
                self.time_encoder= SineActivation(2,time_dim)
                time_total  =time_dim*in_channels
            elif self.apply_method == "fixed_positional_encoding":
                self.time_encoder =None
                time_total = time_dim*in_channels
            elif self.apply_method == "LPE": ## learnable time position
                self.time_encoder =None
                self.time_positional_embedding = nn.Parameter(torch.randn(self.lead_channels,self.ECG_length,self.time_dim)*0.01)
                time_total = time_dim

            else:
                self.time_encoder = None
                self.time_dim = 2
                time_total = 2*in_channels
            in_channels +=time_total
            print ("input dim is increased to {}, with {} dim for time".format(in_channels,time_total))
        self.in_channels = in_channels
        self.act = act
        self.create_network()

    def create_network(self):
        if self.encoder_type =="ms_resnet":
            self.encoder =ECG_encoder(in_channels=self.in_channels,ECG_length=self.ECG_length,
                                        embedding_dim=self.embedding_dim,output_dim=self.latent_code_dim,groups=self.groups,
                                        norm=self.norm, downsample_factor=self.encoder_down_sample_layers, if_VAE=self.if_VAE, mha=self.encoder_mha,downscale_feature=self.downscale_feature,act=self.act, 
                                        last_act=self.encoder_lact_act,apply_batchwise_dropout=self.apply_batchwise_dropout)
        elif self.encoder_type =="resnet1d101":
            self.encoder =ECG_ResNetencoder(in_channels=self.in_channels,ECG_length=self.ECG_length,
                                        embedding_dim=self.embedding_dim,output_dim=self.latent_code_dim,group_conv=self.groups>1,
                                        norm=self.norm, downsample_factor=self.encoder_down_sample_layers, if_VAE=self.if_VAE, mha=self.encoder_mha,
                                        last_act=self.encoder_lact_act,no_linear=self.no_linear,apply_batchwise_dropout=self.apply_batchwise_dropout)
        else:
            raise NotImplementedError("encoder type {} is not implemented".format(self.encoder_type))
        base_spatial = self.encoder.spacial_dim
        ## create decoder
        if self.decoder_type =="ms_resnet" or self.decoder_type =="ms_resnet1d":
            self.decoder = ECG_decoder(in_channels=self.latent_code_dim,base_spatial=base_spatial,output_dim=self.decoder_outdim,norm=self.norm,
                                    upsample_factor=self.decoder_upsample_layers,groups=self.groups,act=self.act)
           
        elif self.decoder_type =="linear":
            self.decoder = LinearDecoder(input_feature_dim=self.latent_code_dim,output_length=self.ECG_length,out_ch=self.decoder_outdim,decompose=False)
        elif self.decoder_type=="attention_decoder":
            self.decoder = ECGDecoder(in_channels=self.latent_code_dim, base_spatial = self.ECG_length//(2**(self.encoder_down_sample_layers)), 
                                  num_leads=self.in_channels,output_dim=self.in_channels,norm=nn.InstanceNorm1d,upsample_factor=self.decoder_upsample_layers,
                                  base_feature=self.base_feature_dim,num_linear_in_D=2)
        else:
            raise NotImplementedError("decoder type {} is not implemented".format(self.decoder_type))
        self.with_positional_encoding = self.with_positional_encoding
        if self.with_positional_encoding:
            raise NotImplementedError("positional encoding is not implemented yet")            
            # self.positional_embedding = nn.Parameter(torch.randn(16,256))

        
        # ## classifier branch
        # self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        # self.fc_1 = nn.Linear(embedding_dim, embedding_dim)
        # self.ch_bn_1 = nn.BatchNorm1d(embedding_dim)
        # self.linear_drop = nn.Dropout(p=0.25)
        # self.fc_2 = nn.Linear(embedding_dim,64)
        # self.ch_bn_2 = nn.BatchNorm1d(64)
        # self.linear_drop_2 = nn.Dropout(p=0.5)
        
        # self.ch_fc1 = nn.Linear(256, 64)
        # self.ch_bn = nn.BatchNorm1d(64)
        # self.ch_fc2 = nn.Linear(64, nOUT)

        ## decoder branch for signal reconstruction and recovering
        # self.decoder_conv1 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(1,16), stride=(1,1), padding=(0,0),bias=False)
        # self.decoder_bn1 = norm(256)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d) :
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        ## init attention module
        try:
            if isinstance(self.encoder.pool, AttentionPool1d) or isinstance(self.encoder.pool,AttentionPool2d):
                    std = self.encoder.pool.c_proj.in_features ** -0.5
                    nn.init.normal_( self.encoder.pool.q_proj.weight, std=std)
                    nn.init.normal_(self.encoder.pool.k_proj.weight, std=std)
                    nn.init.normal_(self.encoder.pool.v_proj.weight, std=std)
                    nn.init.normal_(self.encoder.pool.c_proj.weight, std=std)
        except Exception as e:
            pass

    def forward(self, x, mask = None, ecg_position=None, auto_pad_input=True):
        '''
        x: signal: [N,12,F] where F is the number of input signal points, N is batch size
        missing_input_mask: if there is any lead originally missing, shape [N,12,F],
        ecg_position: [N,12,F,2], where the last dim is the position of the signal, first dim is lead, second dim is time

        '''
        z = self.encodeECG(x, mask = mask, ecg_position=ecg_position,auto_pad_input=auto_pad_input)
        if self.if_VAE:
            mu, log_var = z[0], z[1]
            z = self.reparameterize(mu, log_var)
            self.z_mu = mu
            self.z_log_var = log_var
        self.z = z
        x = self.decodeECG(z)
        return x
    
    def encodeECG(self,x, mask = None, ecg_position=None,auto_pad_input=True):
        '''
        x: signal: [N,12,F] where F is the number of input signal points, N is batch size
        '''
        if x.size(2)<self.ECG_length:
            if auto_pad_input:
                pad_zero_n = self.ECG_length - x.size(2)
                x = torch.nn.functional.pad(x, pad=(pad_zero_n//2,pad_zero_n//2), mode='constant') 
            else:
                raise ValueError("input signal length should be at least {} points".format(self.ECG_length))
        # x = self.input_conv(x)
        if self.add_time:
                ## 12*1024*2
            if self.apply_method == "time2vec":
                if ecg_position is None:
                    time_position = torch.arange(0, self.ECG_length).unsqueeze(0)
                    lead_position = torch.arange(0, self.lead_channels).unsqueeze(1)
                    ecg_position = torch.cat([lead_position.repeat(1,self.ECG_length).unsqueeze(2),time_position.repeat(self.lead_channels,1).unsqueeze(2)],dim=2)
                    ecg_position = ecg_position.float()
                if x.is_cuda:
                    ecg_position = ecg_position.cuda()
                ## scale it down like the positional encoding
                ecg_position /= self.time_dim / 2.
                ecg_position = 1. / 10000**ecg_position  # (D/2,)
                time_vec = self.time_encoder(ecg_position)
                ## N*L*T
                # print ('time vec shape', time_vec.shape)
            elif self.apply_method == "LPE": ## learnable time position
                time_vec = self.time_positional_embedding
            elif self.apply_method == "FPE2vec":
               time_pos = get_1d_sincos_pos_embed_from_grid(2,np.arange(0,self.ECG_length*self.lead_channels)) ##ECG_length*time_dim
               time_enc = np.reshape(time_pos, (self.ECG_length,self.lead_channels,2)) # only 2dim works good
               time_enc = np.transpose(time_enc, (1,0,2)) ### lead, L, T
               time_enc = torch.from_numpy(time_enc).float().to(x.device)
               time_vec = self.time_encoder(time_enc)

            elif self.apply_method == "fixed_positional_encoding":
               time_pos = get_1d_sincos_pos_embed_from_grid(self.time_dim,np.arange(0,self.ECG_length*self.lead_channels)) ##ECG_length*time_dim
               time_enc = np.reshape(time_pos, (self.ECG_length,self.lead_channels,self.time_dim))
               time_enc = np.transpose(time_enc, (1,0,2))
               time_vec = torch.from_numpy(time_enc).float()
               time_vec = time_vec.to(x.device)

            else:
                if ecg_position is None:
                    time_position = torch.arange(1, self.ECG_length+1).unsqueeze(0)
                    lead_position = torch.arange(1, self.lead_channels+1).unsqueeze(1)
                    time_position = time_position/self.ECG_length ## scale it down
                    lead_position = lead_position/self.lead_channels
                    ecg_position = torch.cat([lead_position.repeat(1,self.ECG_length).unsqueeze(2),time_position.repeat(self.lead_channels,1).unsqueeze(2)],dim=2)
                    ecg_position = ecg_position.float()
                if x.is_cuda:
                    ecg_position = ecg_position.cuda()
                time_vec = ecg_position
                # print (time_vec.shape)
            ## timevec
            batch_timevec = time_vec.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(-1)
                batch_timevec = mask*batch_timevec ## bs, lead, L, T
            else:
                mask = torch.ones_like(x)
                mask = mask.unsqueeze(-1)
                batch_timevec = mask*batch_timevec
            ## ns, 12, 1024,t 
            x = x.unsqueeze(-1)
            x = torch.cat([x,batch_timevec],dim=-1) ## N*12*1024*(T+1)
            x = x.permute(0,3,1,2) ## N*T*(12+dim)*1024
            x = x.reshape(x.size(0),-1,self.ECG_length) ## N*T*(12+dim*1024)
        z = self.encoder(x)
        return z
    
    def decodeECG(self,z):
        '''
        z: latent code: [N,F]
        return x: [N,12/24,F]
        '''
        x = self.decoder(z)
        return x
    
    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def compute_VAE_loss(self, beta = 0.00025):
        assert self.if_VAE, "VAE loss is only applicable for VAE model"
        assert  self.z_log_var, "z_log_var is not defined"
        assert  self.z_mu, "z_mu is not defined"
       
        kld_loss = torch.mean(-0.5 * torch.sum(1 + self.z_log_var - self.z_mu ** 2 - self.z_log_var.exp(), dim = 1), dim = 0)
       
        return kld_loss*beta

    def sample(self,
               num_samples:int,
               device=torch.device('cuda')):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,self.latent_code_dim).to(device)
        samples = self.decodeECG(z)
        return samples

class doubleECGNet(nn.Module):
    def __init__(self, encoder_type="ms_resnet", decoder_type="ms_resnet", in_channels=12, ECG_length=1024, embedding_dim=64, encoder_down_sample_layers=5, latent_code_dim=512, encoder_mha=False, if_VAE=False, groups=1, add_time=False, time_dim=4, apply_method="time2vec", decoder_outdim=24, decoder_upsample_layers=5, norm=nn.BatchNorm1d, with_positional_encoding=False,act=nn.GELU()):
        super(doubleECGNet, self).__init__()
        self.mean_EAE = ECGAE(encoder_type=encoder_type, decoder_type=decoder_type, in_channels=in_channels, 
                                ECG_length=ECG_length, embedding_dim=embedding_dim,downscale_feature=1, 
                                encoder_down_sample_layers=encoder_down_sample_layers, latent_code_dim=latent_code_dim, encoder_mha=encoder_mha, if_VAE=if_VAE, groups=groups, add_time=add_time, time_dim=time_dim, apply_method=apply_method, decoder_outdim=decoder_outdim, decoder_upsample_layers=decoder_upsample_layers, norm=norm, with_positional_encoding=with_positional_encoding,act=act)
        self.res_EAE = ECGAE(encoder_type=encoder_type, decoder_type=decoder_type, in_channels=in_channels,embedding_dim=embedding_dim,downscale_feature=1, 
                              ECG_length=ECG_length, encoder_down_sample_layers=encoder_down_sample_layers, latent_code_dim=latent_code_dim, encoder_mha=encoder_mha, if_VAE=if_VAE, groups=groups, add_time=add_time, time_dim=time_dim, apply_method=apply_method, decoder_outdim=decoder_outdim, decoder_upsample_layers=decoder_upsample_layers, norm=norm, with_positional_encoding=with_positional_encoding,act=act)
    def forward(self,x, mask = None, ecg_position=None,auto_pad_input=True):
        res,average = SeriesDecomp(kernel_size=25)(x)
        res = self.res_EAE(res)
        average = self.mean_EAE(average)
        return res+average

if __name__ =="__main__":
    ## test the model
    batch_size = 16
    num_lead = 12
    n_classes = 24
    # Generate sample input
    x = torch.rand(batch_size,num_lead,1,256)
    mdl = ECGNet(in_channels=12,recon_output_channels=12,with_positional_encoding=True,with_mha=True)
    mdl.eval()
    embedding, decoded = mdl(x)

    
    print("Predictions shape:", embedding.shape)
    print("Decoded shape:", decoded.shape)
    print(torch.__version__)