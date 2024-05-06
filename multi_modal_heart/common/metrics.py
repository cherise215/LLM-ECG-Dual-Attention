from tslearn import metrics
import numpy as np
from torchmetrics import Metric
import torch

def cal_multi_lead_dtw(x, y,reduction=True):
    '''
    input: x: (12, 5000)
           y: (12, 5000)
    '''
    n_leads = x.shape[0]
    dists = []
    for i in range(n_leads):
        dist = metrics.dtw(x[i].reshape(-1,1), y[i].reshape(-1,1))
        dists.append(dist)
    if reduction:
        return np.mean(dists)
    return dists

def cal_batch_multi_lead_dtw(x,y,average_lead=False):
    '''
    input: x: (batch_size, 12, 5000)
           y: (batch_size, 12, 5000)
    return scalar or vector of shape (12)
    '''
    batch_size = x.shape[0]
    for i in range(batch_size):
        if i ==0:
            dist = np.array(cal_multi_lead_dtw(x[i], y[i], average_lead))
            
        else:
            dist += np.array(cal_multi_lead_dtw(x[i], y[i], average_lead))
    return dist/batch_size

       



class MyDynamicTimeWarpingScore(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("score", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        
        preds = preds.cpu().numpy()
        target = target.cpu().numpy()
        bs = preds.shape[0]
        for i in range(bs):
            self.score += cal_multi_lead_dtw(preds[i], target[i])
            self.total += 1

    def compute(self):
        return self.score.float() / self.total