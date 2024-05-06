import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatPooling2DLayer(nn.Module):
    def __init__(self):
        super(ConcatPooling2DLayer, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling layer
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Max Pooling layer

    def forward(self, x):
        global_avg_pool = self.global_avg_pool(x)
        max_pool = self.max_pool(x)
        concat_pool = torch.cat((global_avg_pool, max_pool), dim=1)
        concat_pool = torch.squeeze(concat_pool)
        return concat_pool

class ConcatPooling1DLayer(nn.Module):
    def __init__(self):
        super(ConcatPooling1DLayer, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling layer
        self.max_pool = nn.AdaptiveMaxPool1d(1)  # Max Pooling layer

    def forward(self, x):
        global_avg_pool = self.global_avg_pool(x)
        max_pool = self.max_pool(x)
        concat_pool = torch.cat((global_avg_pool, max_pool), dim=1)
        concat_pool = torch.squeeze(concat_pool)
        return concat_pool