import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import numpy as np
sys.path.append(".")
from multi_modal_heart.model.basic_conv1d import AttentionPool1d,MyResidualBlock2D,MultiScaleBlock,AttentionPool2d
from multi_modal_heart.model.custom_layers.pos_embed import get_1d_sincos_pos_embed_from_grid
from multi_modal_heart.model.custom_layers.fixable_dropout import BatchwiseDropout

class ECGAttentionAE(nn.Module):
    def __init__(self, num_leads=12, time_steps=1024, z_dims=64,downsample_factor=5, if_VAE=False,base_feature_dim=1, num_linear_in_D =2,linear_out=64,use_attention_pool=False, no_linear_in_E=False, add_time=False,upsample_factor = 5,
                 no_lead_attention=False, no_time_attention=False,apply_subsequent_attention=False,apply_lead_mask=False, apply_batchwise_dropout=False):
        super(ECGAttentionAE, self).__init__()
       
        self.encoder = ECGEncoder(num_leads, time_steps,downsample_factor=downsample_factor, out_latent_code_dim=z_dims,if_VAE=if_VAE, linear_out=linear_out,
                                  base_feature_dim=base_feature_dim,use_attention_pool=use_attention_pool,no_linear=no_linear_in_E,add_time=add_time,
                                  no_lead_attention=no_lead_attention, no_time_attention=no_time_attention,apply_subsequent_attention=apply_subsequent_attention,apply_lead_mask=apply_lead_mask,apply_batchwise_dropout=apply_batchwise_dropout)
        if no_linear_in_E:
            linear_out = z_dims
        self.decoder = ECGDecoder(in_channels=linear_out, base_spatial = time_steps//(2**(downsample_factor)), 
                                  num_leads=num_leads,output_dim=num_leads,norm=nn.InstanceNorm1d,upsample_factor=upsample_factor,
                                  base_feature=base_feature_dim,num_linear_in_D=num_linear_in_D)
        self.if_VAE = if_VAE ## declare this for pytorch lightning solver
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        return sample
    
    def forward(self, ecg, mask=None):
        '''
        Input: N*L*T
        return: N*L*T
        '''
        ## mask is not used in this model, but for the sake of compatibility with other models
        if self.if_VAE:
            ecg_mean, ecg_std = self.encoder(ecg,mask)
            z = self.reparameterize(ecg_mean, ecg_std)
            self.z_log_var = ecg_std    ## declare this for pytorch lightning solver
            self.z_mu = ecg_mean  ## declare this for pytorch lightning solver
            self.z = z
        else:
            z = self.encoder(ecg,mask)
            self.z  = z
        ecg_preds = self.decoder(z)
        return ecg_preds


class ECGEncoder(nn.Module):
    def __init__(self, num_leads=12, time_steps=1024, z_dims=64,act = nn.GELU(),
                 downsample_factor=5, out_latent_code_dim =64,if_VAE=True,base_feature_dim=1,use_attention_pool=True, linear_out=64, 
                 no_linear=False, add_time=False,
                 no_lead_attention=False, no_time_attention=False, apply_subsequent_attention=False,
                 apply_lead_mask=False,apply_batchwise_dropout=False):
        assert time_steps % 2 == 0, "Time steps must be even"
        super(ECGEncoder, self).__init__()
        self.num_leads = num_leads
        self.time_steps = time_steps
        self.if_VAE =if_VAE
        self.base_feature_dim = base_feature_dim
        self.add_time = add_time
        self.no_lead_attention = no_lead_attention
        self.no_time_attention = no_time_attention
        self.apply_subsequent_attention = apply_subsequent_attention ## first time and then lead attention
        self.apply_batchwise_dropout = apply_batchwise_dropout
        self.apply_lead_mask = apply_lead_mask
      
        self.input_conv = nn.Sequential(
                    nn.Conv1d(num_leads,num_leads*base_feature_dim,kernel_size=5, stride=1, padding=2, bias=True, groups=num_leads),
                    nn.GroupNorm(num_leads,num_leads*base_feature_dim),
                    act,  
                    
                    nn.Conv1d(num_leads*base_feature_dim,num_leads*base_feature_dim,kernel_size=5, stride=1, padding=2, bias=True),
                    nn.GroupNorm(num_leads,num_leads*base_feature_dim),
                    act, 
                    # MultiScaleBlock(num_leads*base_feature_dim,num_leads*base_feature_dim,dilation =2,norm = nn.InstanceNorm1d, act=act,groups=num_leads),
                    )
        list_encoder_layers = []
        ## change feature dimension here
        n_features = [1*self.base_feature_dim,2*self.base_feature_dim,4*self.base_feature_dim,4*self.base_feature_dim,
                      4*self.base_feature_dim,
                      4*self.base_feature_dim]
        if self.add_time:
            n_features[0] = 2*self.base_feature_dim
        for i in range(downsample_factor):
            list_encoder_layers.append(
                MyResidualBlock2D(downsample=True,in_ch=n_features[i],out_ch= n_features[i+1],norm=nn.BatchNorm2d,kernel_size=5,act = act)
                )
        
        self.encoder = nn.Sequential(*list_encoder_layers)
        ## add attention module
        self.attention = STAttentionModel(n_feature=n_features[i+1], num_leads = num_leads, signal_length=time_steps//(2**downsample_factor),
                                     out_feature = out_latent_code_dim, no_lead_attention=no_lead_attention, no_time_attention=no_time_attention,
                                     apply_subsequent_attention=apply_subsequent_attention,apply_lead_mask=apply_lead_mask,apply_batchwise_dropout=apply_batchwise_dropout)
        self.no_linear = no_linear
        self.z = None
 
        if self.no_linear:
            self.linear =nn.Identity()
            print ('no linear layer')
        else:
            if self.if_VAE:
                self.dense_mean = nn.Linear(out_latent_code_dim,linear_out)
                self.dense_std = nn.Linear(out_latent_code_dim,linear_out)
            else:
                self.linear = nn.Linear(out_latent_code_dim,linear_out)
        self.init_weights()

    def get_attention(self):
        return self.attention.lead_attention,self.attention.time_attention
    
    def get_features(z):
        return self.z
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv1d) or  isinstance(m, nn.ConvTranspose1d):
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
            if isinstance(self.attention.attention_pool, AttentionPool1d) or isinstance(self.attention.attention_pool,AttentionPool2d):
                    std = self.attention.attention_pool.c_proj.in_features ** -0.5
                    nn.init.normal_( self.attention.attention_pool.q_proj.weight, std=std)
                    nn.init.normal_(self.attention.attention_pool.k_proj.weight, std=std)
                    nn.init.normal_(self.attention.attention_pool.v_proj.weight, std=std)
                    nn.init.normal_(self.attention.attention_pool.c_proj.weight, std=std)
        except Exception as e:
            pass
        
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        return sample
    
    def forward(self, ecg,mask=None):
        '''
        ecg: [bs, num_leads, time_steps]
        '''
        # print ("attention AE is enabled")
        ecg_features = self.get_features_after_pooling(ecg, mask)
        ## feature before linear layer

        if self.if_VAE:
            ecg_mean = self.dense_mean(ecg_features)
            ecg_std = self.dense_std(ecg_features)
            self.z = self.reparameterize(ecg_mean, ecg_std)
            return ecg_mean, ecg_std     
        else:
            z = self.linear(ecg_features)
        self.z = ecg_features
        return z
    
    def get_features_after_pooling(self,x, mask):
        ## before linear layer
        bs = x.shape[0]
        ecg_features = self.input_conv(x)
        self.input_conv_output = ecg_features
        ecg_features = ecg_features.view(bs, self.base_feature_dim, self.num_leads, self.time_steps)

        ## if add time information
        if self.add_time:
            time_pos = get_1d_sincos_pos_embed_from_grid(self.base_feature_dim,np.arange(0,self.time_steps*self.num_leads)) ##ECG_length*time_dim
            time_enc = np.reshape(time_pos, (self.time_steps,self.num_leads,self.base_feature_dim))
            time_enc = np.transpose(time_enc, (1,0,2))
            time_vec = torch.from_numpy(time_enc).float()
            time_vec = time_vec.to(x.device)
            batch_timevec = time_vec.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(-1)
                batch_timevec = mask*batch_timevec ## bs, lead, L, T
            else:
                mask = torch.ones_like(x)
                mask = mask.unsqueeze(-1)
                batch_timevec = mask*batch_timevec
            ## ns, 12, 1024,t 
            batch_timevec = batch_timevec.permute(0,3,1,2) ## N*T*12*1024
            ecg_features = torch.cat([ecg_features,batch_timevec],dim=1) ## N*(T*2)*12*1024
            
        ecg_features = self.encoder(ecg_features)
        ecg_features = self.attention(ecg_features,mask=mask)
        return ecg_features


class ECGDecoder(nn.Module):
    def __init__(self,in_channels=64,base_spatial = 32, num_leads=12,output_dim=12,norm=nn.InstanceNorm1d,upsample_factor=5,
                    n_features = [12,64,32,32,16,16],groups=1,act=nn.GELU(),base_feature=2,num_linear_in_D=2):
        super(ECGDecoder, self).__init__()
        # n_features = [output_dim]*(upsample_factor+1)
        self.base_spatial = base_spatial
        self.base_feature = base_feature    

        if num_linear_in_D==2:
            self.linear = nn.Sequential(
                    nn.Linear(in_channels,base_feature*base_spatial), 
                    nn.Linear(base_feature*base_spatial,base_feature*output_dim*base_spatial), 
                )
            n_features = [self.base_feature]*(upsample_factor+1)

        elif num_linear_in_D==1:
            self.linear = nn.Sequential(
                    nn.Linear(in_channels,base_feature*output_dim*base_spatial),
                    # nn.Linear(in_channels*base_spatial,self.base_feature*output_dim*base_spatial),
            )
            n_features = [self.base_feature]*(upsample_factor+1)
        else:
            raise NotImplementedError
        
        self.output_dim  = output_dim
     
        list_decoder_layers = [
        ]
        list_decoder_layers.append(
                            MyResidualBlock2D(upsample=True,in_ch=self.base_feature,out_ch=n_features[0],norm=nn.BatchNorm2d,kernel_size=5,act =act))
       
        for i in range(1,upsample_factor):
            list_decoder_layers.append(
                            MyResidualBlock2D(upsample=True,in_ch=n_features[i],out_ch=n_features[i+1],norm=nn.BatchNorm2d,kernel_size=5,act=act))          
        list_decoder_layers.append(
                            MyResidualBlock2D(upsample=False,in_ch=n_features[i+1],out_ch=1,norm=nn.BatchNorm2d,kernel_size=5,act=act))
            
        self.decoder = nn.Sequential(*list_decoder_layers)

        self.final_conv = nn.Sequential(
                    # MultiScaleBlock(output_dim,output_dim,dilation =2,norm = nn.InstanceNorm1d, act=act,groups=output_dim),
                    # MultiScaleBlock(output_dim,output_dim,dilation =2,norm = nn.InstanceNorm1d, act=act,groups=output_dim),
                    nn.Conv1d(output_dim,output_dim,kernel_size=1, stride=1, padding=0, bias=True, groups=output_dim),
                    nn.InstanceNorm1d(output_dim)
                    )
    def forward(self,x,mask=None):
        x = self.linear(x)
        bs = x.size(0)
        x = x.view(bs, -1,self.output_dim,self.base_spatial)

        x = self.decoder(x)
        x = x.squeeze(1)
        x = self.final_conv(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,apply_batchwise_dropout=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        if apply_batchwise_dropout:
            self.drop = BatchwiseDropout(drop,batch_first=True)
        else: self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x   
class STAttentionModel(nn.Module):
    def __init__(self,n_feature=256, num_leads = 12, signal_length=1024 ,out_feature = 64,use_attention_mask=False, no_time_attention=False,no_lead_attention=False,apply_subsequent_attention=False, apply_lead_mask=False,
                 shared_time_attention=False,apply_batchwise_dropout=False):
        '''
        Given a feature input in a size of (batch_size, n_features, num_leads, signal_length)
        first conduct time-wise self-attention for each lead, then conduct lead-wise self-attention to exploit the relationship between different leads
        '''
        super().__init__()
        self.n_feature = n_feature
        self.num_leads = num_leads
        self.signal_length = signal_length
        self.no_lead_attention = no_lead_attention
        self.no_time_attention = no_time_attention
        self.apply_subsequent_attention = apply_subsequent_attention
        self.apply_lead_mask = apply_lead_mask
        self.shared_time_attention = shared_time_attention
        self.apply_batchwise_dropout = apply_batchwise_dropout

        ## define time-wise self-attention for each lead
        if not no_time_attention:
            if not shared_time_attention:
                self.time_mha = nn.ModuleList()
                self.time_norm_1 = nn.ModuleList()
                self.time_mlp = nn.ModuleList()
                self.mlp_norm_1 = nn.ModuleList()
                for i in range(self.num_leads):
                    self.time_norm_1.append(nn.LayerNorm(self.n_feature))
                    self.time_mha.append(nn.MultiheadAttention(embed_dim=self.n_feature,num_heads = 8))
                    self.time_mlp.append(Mlp(in_features=self.n_feature, hidden_features=4*self.n_feature, 
                                            out_features=self.n_feature, act_layer=nn.GELU, drop=0.2,apply_batchwise_dropout=apply_batchwise_dropout))
                    self.mlp_norm_1.append(nn.LayerNorm(self.n_feature))
            else:
                self.time_norm_1= nn.LayerNorm(self.n_feature)
                self.time_mha= nn.MultiheadAttention(embed_dim=self.n_feature,num_heads = 8)
                self.time_mlp = Mlp(in_features=self.n_feature, hidden_features=4*self.n_feature, 
                                            out_features=self.n_feature, act_layer=nn.GELU, drop=0.2,apply_batchwise_dropout=apply_batchwise_dropout)
                self.mlp_norm_1 = nn.LayerNorm(self.n_feature)

                ## fixed
            self.time_encoder = PositionalEncoding(d_model =self.n_feature,dropout=0.2, apply_batchwise_dropout=apply_batchwise_dropout)
        else: 
            self.time_mha=None
            self.time_encoder = None
            self.time_mlp = None
            self.mlp_norm_1 = None

        lead_feature_embed = n_feature*signal_length
        if use_attention_mask: ## only consider backward time points
            attn_mask = torch.ones(signal_length,signal_length)
            self.attn_mask = 1-torch.triu(attn_mask, diagonal=1)
        else:
            self.attn_mask = None
        
        if not no_lead_attention:
            self.lead_mha = nn.MultiheadAttention(embed_dim=lead_feature_embed,num_heads =8)
            if self.apply_batchwise_dropout:
                self.lead_dropout = BatchwiseDropout(0.2,batch_first=False)
            else:
                self.lead_dropout = nn.Dropout(0.2)
            self.lead_encoder = PositionalEncoding(d_model =lead_feature_embed,dropout=0.2)
            self.lead_mlp =Mlp(in_features=lead_feature_embed, hidden_features=4*lead_feature_embed, 
                                            out_features=lead_feature_embed, act_layer=nn.GELU, drop=0.2)
            self.lead_norm_1 = nn.LayerNorm(lead_feature_embed)
            # self.lead_embedding = nn.Parameter(torch.randn(num_leads,1, lead_feature_embed) / lead_feature_embed ** 0.5) ## use learnable positional embedding for lead related positional information
        else:
            self.lead_mha = None
            self.lead_dropout = None
            self.lead_embedding = None
        self.norm2 = nn.LayerNorm(lead_feature_embed)

      
        ## use adaptive average pooling instead
        if not self.apply_batchwise_dropout:
            self.attention_pool_1 = nn.Sequential(
                    nn.Dropout(0.2) if (no_lead_attention and no_time_attention) else nn.Identity(),
                    nn.Conv1d(n_feature*num_leads,out_feature//2,kernel_size=1, stride=1, padding=0, bias=True),
                    nn.AdaptiveAvgPool1d(output_size=1),
                    nn.Flatten()
                    )
        else:
            self.attention_pool_1 = nn.Sequential(
                BatchwiseDropout(0.2)if (no_lead_attention and no_time_attention) else nn.Identity(),
                nn.Conv1d(n_feature*num_leads,out_feature//2,kernel_size=1, stride=1, padding=0, bias=True),
                nn.AdaptiveAvgPool1d(output_size=1),
                nn.Flatten()
                )

        ## use adaptive max pooling 
        if not self.apply_batchwise_dropout:
            self.attention_pool_2 = nn.Sequential(
                    nn.Dropout(0.2) if (no_lead_attention and no_time_attention) else nn.Identity(),
                    nn.Conv1d(n_feature*num_leads,out_feature//2,kernel_size=1, stride=1, padding=0, bias=True),
                    nn.AdaptiveMaxPool1d(output_size=1),
                    nn.Flatten()
                    )
        else:
            self.attention_pool_2 = nn.Sequential(
                    BatchwiseDropout(0.2) if (no_lead_attention and no_time_attention) else nn.Identity(),
                    nn.Conv1d(n_feature*num_leads,out_feature//2,kernel_size=1, stride=1, padding=0, bias=True),
                    nn.AdaptiveMaxPool1d(output_size=1),
                    nn.Flatten()
                    )

    def forward(self, input, mask=None):
        '''
        Input: N*C*L*T, N: batch_size, C: n_channels, L: num_leads, T: signal_length
        mask: N*L*T: 0 for masked region, 1 for unmasked region
        return: N*F [embeded feature]
        '''
        batch_size = input.shape[0]
        num_leads = self.num_leads
        lead_time_output=[]
        lead_attention_masks = []
        ## for each lead, compute self-attention on time dimension
        if not self.no_time_attention:
            for i in range(num_leads):
                lead_time_input = input[:, :, i, :] ## (batch_size, n_features, signal_length)
                lead_time_input = lead_time_input.permute(2,0,1) ## (signal_length, batch_size, n_features)
                lead_input_w_time_encoding = self.time_encoder(lead_time_input)
                if not self.shared_time_attention:
                    ## pre-ln
                    lead_input_w_time_encodin_normed =self.time_norm_1[i](lead_input_w_time_encoding)
                    lead_out, lead_attention_weight = self.time_mha[i](lead_input_w_time_encodin_normed,
                                                                    lead_input_w_time_encodin_normed,
                                                                    lead_input_w_time_encodin_normed,
                                                                    attn_mask=self.attn_mask)
                    lead_out =lead_out+lead_input_w_time_encoding

                    ## mlp processing
                    lead_out = lead_out+ self.time_mlp[i](self.mlp_norm_1[i](lead_out))
                else:
                    ## shared time attention
                    lead_input_w_time_encodin_normed =self.time_norm_1(lead_input_w_time_encoding)
                    lead_out, lead_attention_weight = self.time_mha(lead_input_w_time_encodin_normed,
                                                                    lead_input_w_time_encodin_normed,
                                                                    lead_input_w_time_encodin_normed,
                                                                    attn_mask=self.attn_mask)
                    lead_out =lead_out+lead_input_w_time_encoding

                    ## mlp processing
                    lead_out = lead_out+ self.time_mlp(self.mlp_norm_1(lead_out))

                lead_time_output.append(lead_out)
                lead_attention_masks.append(lead_attention_weight)
            lead_time_output = torch.stack(lead_time_output,dim=2) ## (signal_length, batch_size, num_lead, features)
            # lead_batch_feature = lead_time_output.permute(2,1,3,0).reshape(num_leads,batch_size,-1) ## (num_lead, batch_size, n_features,signal_length)
            batch_lead_time_output = lead_time_output.permute(1,3,2,0) ## (batch_size, n_features, num_lead,signal_length)
            lead_time_output = batch_lead_time_output.reshape(batch_size,self.n_feature*self.num_leads,self.signal_length)
        else:
            # print ('no time-wise attention')
            lead_attention_masks = None
            # lead_batch_feature = input.permute(2,0,1,3).reshape(num_leads,batch_size,-1) ## (num_lead, batch_size, n_features*signal_length)
            lead_time_output = input.reshape(batch_size,self.n_feature*self.num_leads,self.signal_length)

        ## for time series output, compute cross-lead attention
        if not self.no_lead_attention:
            if self.apply_subsequent_attention:
                ## use lead time output
                lead_input = lead_time_output.reshape(batch_size,self.n_feature,self.num_leads,self.signal_length)
                lead_input = lead_input.permute(2,0,1,3)
                lead_input = lead_input.reshape(num_leads, batch_size,-1)  ## (num_lead, batch_size, n_features*signal_length)
                ## normalization over the feature dimension.
            else:
                lead_input = input.permute(2,0,1,3).reshape(num_leads,batch_size,-1) ## (num_lead, batch_size, n_features*signal_length)
            
            lead_input_w_lead_encoding = self.lead_encoder(lead_input)
            # lead_input_w_lead_encoding = self.lead_dropout(self.lead_embedding+lead_input)
            normed_lead_input_w_lead_encoding = self.norm2(lead_input_w_lead_encoding)
            if self.apply_lead_mask:
                ## mask[b,i,j] = 1  if mask_lead[b,j] ==-infnity or  mask[b,i,j] = 0
                lead_attn_mask = torch.zeros((batch_size,num_leads,num_leads),device =lead_input_w_lead_encoding.device)
                
                if mask is not None:
                    ##  N*L*T: 0 for masked region, 1 for unmasked region
                    mask_lead = torch.mean(mask,dim=2) ## (batch_size, num_leads)
                    lead_attn_mask.masked_fill_(mask_lead.unsqueeze(1)==0, 1)  ## set to true for masking
                ## duplicate the mask to match the number of heads
                num_heads =8
                ## matching the heads dimension)
                lead_attn_mask = lead_attn_mask.unsqueeze(0).repeat(num_heads, 1, 1, 1)
                lead_attn_mask = lead_attn_mask.view(-1, num_leads, num_leads)
                lead_attn_mask= lead_attn_mask>0
            else:
                lead_attn_mask = None
            cross_lead,lead_attn_output_weights = self.lead_mha(normed_lead_input_w_lead_encoding,
                                                                normed_lead_input_w_lead_encoding,
                                                                normed_lead_input_w_lead_encoding,
                                                                attn_mask=lead_attn_mask)
            
            ## residual connection
            cross_lead = cross_lead+lead_input_w_lead_encoding
            ## mlp processing
            cross_lead = cross_lead+ self.lead_mlp(self.lead_norm_1(cross_lead))

            cross_lead = cross_lead.permute(1,2,0).reshape(batch_size,self.n_feature,self.signal_length,self.num_leads) ## (batch_size, n_features, num_lead, signal_length)
            cross_lead = cross_lead.permute(0,1,3,2) ## (batch_size, n_features, num_lead, signal_length)
            cross_lead_time = cross_lead.reshape(batch_size,self.n_feature*self.num_leads,self.signal_length)
        else:
            lead_attn_output_weights = None
            if self.no_time_attention:
                cross_lead_time = input.reshape(batch_size,self.n_feature*self.num_leads,self.signal_length)
            else:
                cross_lead_time = lead_time_output
        ## add the two outputs together
        if not self.apply_subsequent_attention:
            cross_lead_sum = cross_lead_time+lead_time_output
        else:
            cross_lead_sum = cross_lead_time

        cross_lead_time_flattenn = torch.cat([self.attention_pool_1(cross_lead_sum),self.attention_pool_2(cross_lead_sum)],dim=1)
       
        self.lead_attention = lead_attn_output_weights
        self.time_attention = lead_attention_masks 

        return cross_lead_time_flattenn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.2, max_len: int = 5000,apply_batchwise_dropout=False):
        super().__init__()
        if apply_batchwise_dropout:
            self.dropout = nn.Dropout(p=dropout)
        else: self.dropout = BatchwiseDropout(p=dropout,batch_first=False)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe, persistent=False)
    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

if __name__ == '__main__':
    ## test the model 
    batch_size = 32
    n_features = 256
    num_lead = 12
    signal_length = 1024//32
    input = torch.randn(batch_size,n_features,num_lead,signal_length)
    model = STAttentionModel(n_feature=256, num_leads = 12, signal_length=1024//32 ,out_feature = 64)
    output = model(input)
    import numpy as np
    ## print model parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print (count_parameters(model))