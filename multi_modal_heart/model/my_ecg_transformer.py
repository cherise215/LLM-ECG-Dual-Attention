import math
import torch
from torch import Tensor
import torch.nn as nn
from multi_modal_heart.model.basic_conv1d import MultiScaleBlock2d


class ECG_transformer(nn.Module):
    def __init__(self, ECG_length=1024,patch_size=16,n_lead: int=12,n_head: int=12,nlayers: int=4, dropout: float = 0.5,decoder_outdim=12):
        super().__init__()
        self.model_type = 'Transformer'
        self.embedding = ECGPatchEmbed(ECG_length, patch_size=patch_size, in_chans=n_lead)
        d_model = n_lead*patch_size
        self.n_lead= n_lead
        dim_feedforward = d_model*2
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer_encoder = nn.Transformer(nhead=n_head, num_encoder_layers=nlayers,d_model=d_model,dim_feedforward=dim_feedforward,activation='gelu').encoder

        self.transformer_decoder = nn.Transformer(nhead=n_head, num_encoder_layers=nlayers,d_model=d_model,dim_feedforward=d_model,activation='gelu').encoder
        # self.upsampler = nn.Sequential(
        #     nn.Conv1d(d_model, n_lead*patch_size*decoder_outdim//n_head, kernel_size=1, stride=1),
        # )
        self.decoder_outdim=decoder_outdim
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.uniform_(-initrange, initrange)
                if m.bias is not None:
                    m.bias.data.zero_()
         # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.embedding.proj[0].weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
       

    def forward(self, src: Tensor, mask=None,src_mask: Tensor = None, tgt_mask:Tensor=None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        b, n_lead, seq_len = src.shape
        src = self.embedding(src) ## (batches, num_pathes, num_feature)
        # [seq_len, batch_size, embedding_dim]``
        src = src.permute(1,0,2) ## (num_pathes, batch_size, num_feature=path_size*n_lead)
        src = self.pos_encoder(src)

        output = self.transformer_encoder(src, src_mask)
        output = self.pos_encoder(output)
        output = self.transformer_decoder(output,tgt_mask)
        output = output.permute(1,2,0) ## (batch_size, n_lead,seq_len)
        output = output.reshape(b,n_lead,seq_len)
        # output = output.reshape(output.shape[0],self.decoder_outdim,-1) ## (batch_size, n_lead,seq_len)
        return output
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class ECGPatchEmbed(nn.Module):
    """ ECG to Patch Embedding
    """
    def __init__(self, ECG_length=1024, patch_size=16, in_chans=12):
        super().__init__()
        
        num_patches = (ECG_length//patch_size)
        self.patch_shape = (patch_size, in_chans)
        self.ECG_length = ECG_length
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.in_conv = nn.Sequential(
              nn.Conv2d(1, 1, kernel_size=(1,15), stride=(1,1), padding=(0,7)),
              nn.BatchNorm2d(1),
              nn.GELU(),
        )

        self.proj=nn.Sequential(
                nn.Conv2d(1, patch_size*in_chans, kernel_size=(in_chans,patch_size), stride=(in_chans,patch_size)),
                )

    def forward(self, x, **kwargs):
        B, C, L = x.shape
        x = x.view(B,1,C,L)
        x = self.in_conv(x)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
