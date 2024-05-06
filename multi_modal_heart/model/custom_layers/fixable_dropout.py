import torch.nn.functional as F
import torch.nn as nn
import torch 

import torch
import torch.nn as nn
import torch.nn.functional as F
class FixableDropout2d(nn.Module):
    """
     based on 2D pytorch dropout, supporting lazy load with last generated mask.
     To use last generated mask, set lazy_load to True
    """
    def __init__(self, p: float = 0.5,inplace=False,lazy_load: bool = False,training=True):
        super(FixableDropout2d, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p
        self.inplace = inplace
        self.seed  = None
        self.lazy_load = lazy_load
        self.training=training

    def forward(self, X):
        if self.training:
            if self.lazy_load:
                if not self.seed is None:
                    seed  = self.seed
                else:
                    seed = torch.seed()
            else:seed = torch.seed()
        else:
            seed = torch.seed()
        self.seed=seed
        torch.manual_seed(seed)
        X = F.dropout2d(X, p=self.p, training=self.training, inplace=self.inplace)
        return X
    def fix(self):
        self.lazy_load=True
    def activate(self):
        self.lazy_load=False

class FixableDropout1d(nn.Module):
    """
     based on 1D pytorch dropout, supporting lazy load with last generated mask.
     To use last generated mask, set lazy_load to True
    """
    def __init__(self, p: float = 0.5,inplace=False,lazy_load: bool = False,training=True):
        super(FixableDropout1d, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p
        self.inplace = inplace
        self.seed  = None
        self.lazy_load = lazy_load
        self.training=training

    def forward(self, X):
        if self.training:
            if self.lazy_load:
                if not self.seed is None:
                    seed  = self.seed
                else:
                    seed = torch.seed()
            else:seed = torch.seed()
        else:
            seed = torch.seed()
        self.seed=seed
        torch.manual_seed(seed)
        X = F.dropout1d(X, p=self.p, training=self.training, inplace=self.inplace)
        return X
    def fix(self):
        self.lazy_load=True
    def activate(self):
        self.lazy_load=False


class BatchwiseDropout(nn.Module):
    '''
    perform same mask to all features in a batch

    '''
    def __init__(self, p=0.5, batch_first=True):
        super(BatchwiseDropout, self).__init__()
        self.p = p
        self.batch_first =batch_first

    def forward(self, x):
        if self.training:
            # Generate a mask of the same shape as the input
            if self.batch_first:
                mask = torch.rand(1,*x.shape[1:], device=x.device) > self.p
            else:
                ## sequential, batch, feature [transformer input]
                if len(x.shape)>2:
                    mask = torch.rand(x.shape[0],1, *x.shape[2:], device=x.device) > self.p
                elif len(x.shape)==2:
                    mask = torch.rand(x.shape[0],1, device=x.device) > self.p
                else:
                    raise ValueError("input shape not supported")
                # Broadcast the mask to match the shape of the input tensor
            mask = mask.expand_as(x)
            self.mask = mask            
            # Apply the mask to the input
            x = x * mask / (1 - self.p)
        
        return x      
if __name__ =="__main__":
    a = torch.rand(3,3,1,1)
    training= True
    dropout_layer = FixableDropout2d(0.5,training=training)
    masked_a = dropout_layer(a)
    dropout_layer.lazy_load=True

    mask = dropout_layer(torch.ones_like(a))
    # mask = mask.bool()
    print(torch.sum(masked_a-(a*mask)))