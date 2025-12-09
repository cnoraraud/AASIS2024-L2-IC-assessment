import math
import torch
import numpy as np

def dim_fit(sizes, dims=2, dtype=None, repeat=-1):
    if isinstance(dtype, type(None)):
        dtype = type(sizes)
    if type(sizes) in (int, np.int32, np.int64):
        sizes = [sizes]
    if type(sizes) in (np.ndarray, tuple, torch.Size, torch.Tensor):
        sizes = list(sizes)
    if len(sizes) > dims:
        sizes = sizes[len(sizes)-dims:]
    elif len(sizes) == dims:
        sizes = sizes
    elif len(sizes) < dims:
        sizes = sizes + [sizes[repeat]] * (dims - len(sizes))
    
    for i in range(len(sizes)):
        sizes[i] = int(sizes[i])
    
    if dtype is torch.Size:
        sizes = torch.Size(sizes)
    elif dtype is np.ndarray:
        sizes = np.array(sizes)
    elif dtype is tuple:
        sizes = tuple(sizes)
    elif dtype is torch.Tensor:
        sizes = torch.Tensor(sizes)
    
    if dims == 1:
        sizes = sizes[0]

    return sizes

def get_param_counts(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def size_norm(size):
    new_size = round(size)
    new_size = int(new_size)
    new_size = max(1, new_size)
    return new_size

def size_mult(size, mult):
    new_size = size * mult
    new_size = size_norm(new_size)
    return new_size

def check(input, check_none=False, check_false=False):
    if check_none and isinstance(input, type(None)):
        return True
    if check_false and (isinstance(input, bool) and (not input)):
        return True
    return False

def get_size_list(input_size=1, hidden_size=1, output_size=1, in_n=1, out_n=1):
    if check(in_n, True, True):
        in_n = 0
    if check(out_n, True, True):
        out_n = 0
    if check(hidden_size, True, True):
        hidden_size = size_norm(math.sqrt(input_size * output_size))
    if in_n > 0:
        in_base = pow(hidden_size/input_size, 1/in_n)
    if out_n > 0:
        out_base = pow(hidden_size/output_size, 1/out_n)

    
    size_list = [input_size]
    for i in range(1, in_n):
        next_size = size_mult(input_size, pow(in_base, i))
        size_list.append(next_size)
    size_list.append(hidden_size)
    for i in range(1, out_n):
        next_size = size_mult(output_size, pow(out_base, i))
        size_list.append(next_size)
    size_list.append(output_size)
    
    return size_list

def f_shape_caller(x_shape, *modules, debug=False):
    for module in modules:
        x_shape = module.f_shape(x_shape)
        if debug:
            print(x_shape)
    return x_shape

def find_hidden_dim(input_dim, low=True):
        hidden_dim = 4
        while hidden_dim < input_dim:
            hidden_dim = hidden_dim * 2
        if low:
            return hidden_dim // 2
        return hidden_dim

def forward_caller(x, *modules, debug=False):
    for module in modules:
        x = module.foward(x)
        if debug:
            print(x.shape)
    return x

def get(option, recipe, key):
    return option if option is not None else recipe.get(key)

# https://github.com/aalto-speech/slate-2025/blob/main/closed_track/HF_baseline_CornLoss.py
# https://www.isca-archive.org/slate_2025/porwal25_slate.pdf
class CORNLogitsLoss(torch.nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(weight)
    
    def forward_logits(self, raw_logits):
        K1 = raw_logits.shape[-1]
        shifted_logits = torch.zeros_like(raw_logits)
        shifted_logits[:, 0] = raw_logits[:, 0]
        for k in range(1, K1):
            log_odds_prev = shifted_logits[:, k-1]
            shifted_logits[:, k] = raw_logits[:, k] + log_odds_prev
        return shifted_logits
    
    def loss_with_logits(self, raw_logits, labels):
        """
        raw_logits: shape (B, K-1)
        labels: shape (B,) in [0..K-1]
        Return: a scalar loss
    
        B: Batch size
        K: number of classes
        """
        device = raw_logits.device
        K1 = raw_logits.shape[-1]
        z = self.forward_logits(raw_logits)  # (B, K-1)
        
        range_vec = torch.arange(K1, device=device).unsqueeze(0)  # shape (1, K-1)
        # target[i,k] = 1 if labels[i] > k else 0
        target = (labels.unsqueeze(1) > range_vec).float()
        loss = self.bce(z, target)
        return loss
    
    def inference(self, raw_logits, pass_thresh=0.5):
        """
        raw_logits: shape (B, K-1)
        Return integer predictions in [0..K].
        We'll do the same shifting, then threshold each probability at 0.5.
        """
        z = self.forward_logits(raw_logits)  # shift
        p = torch.sigmoid(z)
        passes = (p >= pass_thresh)
        preds = passes.sum(dim=1)
        return preds

    def forward(self, input, target):
        return self.loss_with_logits(input, target)