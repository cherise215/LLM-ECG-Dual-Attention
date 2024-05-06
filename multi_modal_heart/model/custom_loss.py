import torch
import torch.nn as nn
import numpy as np
import torch.cuda
from numba import jit
from torch.autograd import Function
from numba import cuda
from torch.nn import functional as F
import math
from typing import Tuple, List
## Implementation of the Survival Prediction Loss
class CoxPHLoss(torch.nn.Module):
    def __init__(self):
        super(CoxPHLoss, self).__init__()

    def forward(self, input, time, event):
        # sort patients by survival time in descending order
        sorted, idx = torch.sort(time, 0, descending=True)
        input = input[idx]

        # get risk scores by exponentiating (-input)
        # because higher predictions should be associated with lower survival
        risk_scores = (-input).exp()

        # calculate log of cummulative hazard
        log_cumulative_hazard = (risk_scores.log().cumsum(0)).view(-1)

        # compute loss
        loss = (input - log_cumulative_hazard).sum()

        # mask out data for which event=0, because these are censored data points
        event = event[idx].view(-1)
        loss = loss - (1 - event).sum()

        # average the loss by batch size
        loss /= input.size()[0]
        return loss

    

class BarlowTwinsLoss(nn.Module):
    def __init__(self, batch_size, lambda_coeff=5e-3, z_dim=128):
        super().__init__()

        self.z_dim = z_dim
        self.batch_size = batch_size
        self.lambda_coeff = lambda_coeff

    def off_diagonal_ele(self, x):
        # taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        # N x D, where N is the batch size and D is output dim of projection head
        z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
        z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)

        cross_corr = torch.matmul(z1_norm.T, z2_norm) / self.batch_size

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()

        return on_diag + self.lambda_coeff * off_diag
    

class UniformityLoss(nn.Module):
    def __init__(self, lambda_coeff=5e-3):
        super().__init__()
        self.lambda_coeff = lambda_coeff

    def off_diagonal_ele(self, x):
        # taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1):
        # N x D, where N is the batch size and D is output dim of projection head
        z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
        batch_size = z1_norm.shape[0]
        cross_corr = torch.matmul(z1_norm.T, z1_norm) / batch_size
        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()
        return on_diag + self.lambda_coeff * off_diag

class CLIPLoss(torch.nn.Module):
  """
  Loss function for multimodal contrastive learning based on the CLIP paper.
  
  Embeddings are taken, L2 normalized and dot product between modalities is calculated to generate a cosine
  similarity between all combinations of subjects in a cross-modal fashion. Tempered by temperature.
  Loss is calculated attempting to match each subject's embeddings between the modalities i.e. the diagonal. 
  """
  def __init__(self, 
               temperature: float = 0.1,
               lambda_0: float = 0.5) -> None:
    super(CLIPLoss, self).__init__()

    self.temperature = temperature
    self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    if lambda_0 > 1 or lambda_0 < 0:
      raise ValueError('lambda_0 must be a float between 0 and 1.')
    self.lambda_0 = lambda_0
    self.lambda_1 = 1-lambda_0

  def forward(self, out0: torch.Tensor, out1: torch.Tensor, indices: List[int] = None) -> Tuple:
    # normalize the embedding onto the unit hypersphere
    out0 = nn.functional.normalize(out0, dim=1)
    out1 = nn.functional.normalize(out1, dim=1)

    #logits = torch.matmul(out0, out1.T) * torch.exp(torch.tensor(self.temperature))
    logits = torch.matmul(out0, out1.T) / self.temperature
    labels = torch.arange(len(out0), device=out0.device)
    
    loss_0 = self.lambda_0 * self.cross_entropy(logits, labels)
    loss_1 = self.lambda_1 * self.cross_entropy(logits.T, labels)
    loss = loss_0 + loss_1
  
    return loss, logits, labels

def cal_dtw_loss(ecg1, ecg2, reduction=True): # to do: plot the curve of x-y axis.
    """
    计算两个ECG序列之间的Dynamic Time Warping（DTW）损失。

    参数：
    - ecg1: 第一个ECG序列，形状为 (batch_size, seq_len1, num_features)
    - ecg2: 第二个ECG序列，形状为 (batch_size, seq_len2, num_features)

    返回：
    - dtw_loss: DTW损失，标量张量
    """
    batch_size, seq_len1, num_features = ecg1.size()
    _, seq_len2, _ = ecg2.size()

    # 计算两个ECG序列之间的距离矩阵
    distance_matrix = torch.cdist(ecg1, ecg2)  # 形状为 (batch_size, seq_len1, seq_len2)

    # 初始化动态规划表格
    # torch.autograd.set_detect_anomaly(True)
    dp = torch.zeros((batch_size, seq_len1, seq_len2)).to(ecg1.device)

    # 填充动态规划表格
    dp[:, 0, 0] = distance_matrix[:, 0, 0]
    for i in range(1, seq_len1):
        dp[:, i, 0] = distance_matrix[:, i, 0] + dp[:, i-1, 0].clone()
    for j in range(1, seq_len2):
        dp[:, 0, j] = distance_matrix[:, 0, j] + dp[:, 0, j-1].clone()
    for i in range(1, seq_len1):
        for j in range(1, seq_len2):
            dp[:, i, j] = distance_matrix[:, i, j] + torch.min(torch.stack([
                dp[:, i-1, j].clone(),
                dp[:, i, j-1].clone(),
                dp[:, i-1, j-1].clone()
            ], dim=1), dim=1).values

    dtw_loss = torch.mean(dp[:, seq_len1-1, seq_len2-1] / (seq_len1 + seq_len2))

    return dtw_loss

def calc_SNR_loss(original_signal, reconstructed_signal):
    '''
    Calculates the SNR loss between the original and reconstructed signals.
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3077765/
    '''
    signal_power = torch.mean(original_signal**2)
    noise_power = torch.mean((original_signal - reconstructed_signal)**2)
    snr = 10 * torch.log10(signal_power / noise_power)
    snr_loss = -snr  # Negate SNR to create a loss (minimize the loss to maximize SNR)
    return snr_loss

def calc_scale_invariant_SNR_loss(target,preds):
    '''
    reference: https://github.com/Lightning-AI/torchmetrics/blob/v1.0.0/src/torchmetrics/functional/audio/sdr.py
    https://www.tutorialexample.com/implement-scale-invariant-source-to-noise-ratio-si-snr-in-tensorflow/
    Args:
        preds: float tensor with shape ``(...,time)``
        target: float tensor with shape ``(...,time)``
        zero_mean: If to zero mean target and preds or not

    Returns:
        Float tensor with shape ``(...,)`` of SDR values per sample
    '''
    eps = torch.finfo(preds.dtype).eps
    target = target - torch.mean(target, dim=-1, keepdim=True)
    preds = preds - torch.mean(preds, dim=-1, keepdim=True)

    alpha = (torch.sum(preds * target, dim=-1, keepdim=True) + eps) / (
        torch.sum(target**2, dim=-1, keepdim=True) + eps
    )
    target_scaled = alpha * target

    noise = target_scaled - preds

    val = (torch.sum(target_scaled**2, dim=-1) + eps) / (torch.sum(noise**2, dim=-1) + eps)
    return -torch.mean(10 * torch.log10(val))
# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_softdtw_cuda(D, gamma, bandwidth, max_i, max_j, n_passes, R):
    """
    :param seq_len: The length of the sequence (both inputs are assumed to be of the same size)
    :param n_passes: 2 * seq_len - 1 (The number of anti-diagonals)
    """
    # Each block processes one pair of examples
    b = cuda.blockIdx.x
    # We have as many threads as seq_len, because the most number of threads we need
    # is equal to the number of elements on the largest anti-diagonal
    tid = cuda.threadIdx.x

    # Compute I, J, the indices from [0, seq_len)

    # The row index is always the same as tid
    I = tid

    inv_gamma = 1.0 / gamma

    # Go over each anti-diagonal. Only process threads that fall on the current on the anti-diagonal
    for p in range(n_passes):

        # The index is actually 'p - tid' but need to force it in-bounds
        J = max(0, min(p - tid, max_j - 1))

        # For simplicity, we define i, j which start from 1 (offset from I, J)
        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds
        if I + J == p and (I < max_i and J < max_j):
            # Don't compute if outside bandwidth
            if not (abs(i - j) > bandwidth > 0):
                r0 = -R[b, i - 1, j - 1] * inv_gamma
                r1 = -R[b, i - 1, j] * inv_gamma
                r2 = -R[b, i, j - 1] * inv_gamma
                rmax = max(max(r0, r1), r2)
                rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
                softmin = -gamma * (math.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin

        # Wait for other threads in this block
        cuda.syncthreads()

# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_softdtw_backward_cuda(D, R, inv_gamma, bandwidth, max_i, max_j, n_passes, E):
    k = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    # Indexing logic is the same as above, however, the anti-diagonal needs to
    # progress backwards
    I = tid

    for p in range(n_passes):
        # Reverse the order to make the loop go backward
        rev_p = n_passes - p - 1

        # convert tid to I, J, then i, j
        J = max(0, min(rev_p - tid, max_j - 1))

        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds
        if I + J == rev_p and (I < max_i and J < max_j):

            if math.isinf(R[k, i, j]):
                R[k, i, j] = -math.inf

            # Don't compute if outside bandwidth
            if not (abs(i - j) > bandwidth > 0):
                a = math.exp((R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) * inv_gamma)
                b = math.exp((R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) * inv_gamma)
                c = math.exp((R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) * inv_gamma)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c

        # Wait for other threads in this block
        cuda.syncthreads()

# ----------------------------------------------------------------------------------------------------------------------
def jacobean_product_squared_euclidean(X, Y, Bt):
    '''
    jacobean_product_squared_euclidean(X, Y, Bt):
    
    Jacobean product of squared Euclidean distance matrix and alignment matrix.
    See equations 2 and 2.5 of https://arxiv.org/abs/1703.01541
    '''
    # print(X.shape, Y.shape, Bt.shape)
    
    ones = torch.ones(Y.shape).to('cuda' if Bt.is_cuda else 'cpu')
    return 2 * (ones.matmul(Bt) * X - Y.matmul(Bt))

class _SoftDTWCUDA(Function):
    """
    CUDA implementation is inspired by the diagonal one proposed in https://ieeexplore.ieee.org/document/8400444:
    "Developing a pattern discovery method in time series data and its GPU acceleration"
    """

    @staticmethod
    def forward(ctx, X, Y, D, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma = torch.cuda.FloatTensor([gamma])
        bandwidth = torch.cuda.FloatTensor([bandwidth])

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        # Prepare the output array
        R = torch.ones((B, N + 2, M + 2), device=dev, dtype=dtype) * math.inf
        R[:, 0, 0] = 0

        # Run the CUDA kernel.
        # Set CUDA's grid size to be equal to the batch size (every CUDA block processes one sample pair)
        # Set the CUDA block size to be equal to the length of the longer sequence (equal to the size of the largest diagonal)
        compute_softdtw_cuda[B, threads_per_block](cuda.as_cuda_array(D.detach()),
                                                   gamma.item(), bandwidth.item(), N, M, n_passes,
                                                   cuda.as_cuda_array(R))
        ctx.save_for_backward(D, X, Y, R, gamma, bandwidth)
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, X, Y, R, gamma, bandwidth = ctx.saved_tensors

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        D_ = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        D_[:, 1:N + 1, 1:M + 1] = D

        R[:, :, -1] = -math.inf
        R[:, -1, :] = -math.inf
        R[:, -1, -1] = R[:, -2, -2]

        E = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        E[:, -1, -1] = 1

        # Grid and block sizes are set same as done above for the forward() call
        compute_softdtw_backward_cuda[B, threads_per_block](cuda.as_cuda_array(D_),
                                                            cuda.as_cuda_array(R),
                                                            1.0 / gamma.item(), bandwidth.item(), N, M, n_passes,
                                                            cuda.as_cuda_array(E))
        E = E[:, 1:N + 1, 1:M + 1]
        G = jacobean_product_squared_euclidean(X.transpose(1,2), Y.transpose(1,2), E.transpose(1,2)).transpose(1,2)

        return grad_output.view(-1, 1, 1).expand_as(G) * G, None, None, None, None

# ----------------------------------------------------------------------------------------------------------------------
class SoftDTW(torch.nn.Module):
    """
    The soft DTW implementation that optionally supports CUDA
    """

    def __init__(self, use_cuda, gamma=1.0, normalize=False, bandwidth=None, dist_func=None):
        """
        Initializes a new instance using the supplied parameters
        :param use_cuda: Flag indicating whether the CUDA implementation should be used
        :param gamma: sDTW's gamma parameter
        :param normalize: Flag indicating whether to perform normalization
                          (as discussed in https://github.com/mblondel/soft-dtw/issues/10#issuecomment-383564790)
        :param bandwidth: Sakoe-Chiba bandwidth for pruning. Passing 'None' will disable pruning.
        :param dist_func: Optional point-wise distance function to use. If 'None', then a default Euclidean distance function will be used.
        """
        super(SoftDTW, self).__init__()

        assert use_cuda, "Only the CUDA version is supported."

        self.normalize = normalize
        self.gamma = gamma
        self.bandwidth = 0 if bandwidth is None else float(bandwidth)
        self.use_cuda = use_cuda

        # Set the distance function
        if dist_func is not None:
            self.dist_func = dist_func
        else:
            self.dist_func = SoftDTW._euclidean_dist_func

    def _get_func_dtw(self, x, y):
        """
        Checks the inputs and selects the proper implementation to use.
        """
        bx, lx, dx = x.shape
        by, ly, dy = y.shape
        # Make sure the dimensions match
        assert bx == by  # Equal batch sizes
        assert dx == dy  # Equal feature dimensions

        use_cuda = self.use_cuda

        if use_cuda and (lx > 1024 or ly > 1024):  # We should be able to spawn enough threads in CUDA
                print("SoftDTW: Cannot use CUDA because the sequence length > 1024 (the maximum block size supported by CUDA)")
                use_cuda = False

        # Finally, return the correct function
        return _SoftDTWCUDA.apply

    @staticmethod
    def _euclidean_dist_func(x, y):
        """
        Calculates the Euclidean distance between each element in x and y per timestep
        """
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        return torch.pow(x - y, 2).sum(3)

    def forward(self, X, Y):
        """
        Compute the soft-DTW value between X and Y
        :param X: One batch of examples, batch_size x seq_len x dims
        :param Y: The other batch of examples, batch_size x seq_len x dims
        :return: The computed results
        """

        # Check the inputs and get the correct implementation
        func_dtw = self._get_func_dtw(X, Y)

        if self.normalize:
            # Stack everything up and run
            x = torch.cat([X, X, Y])
            y = torch.cat([Y, X, Y])
            D = self.dist_func(x, y)
            out = func_dtw(X, Y, D, self.gamma, self.bandwidth)
            out_xy, out_xx, out_yy = torch.split(out, X.shape[0])
            return out_xy - 1 / 2 * (out_xx + out_yy)
        else:
            D_xy = self.dist_func(X, Y)
            return func_dtw(X, Y, D_xy, self.gamma, self.bandwidth)