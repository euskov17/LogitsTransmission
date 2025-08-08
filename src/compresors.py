import torch
import torch.nn as nn
import random
from abc import ABC, abstractmethod

class Compressor(ABC):
    @abstractmethod
    def compress(self, x):
        pass

class IdenticalCompressor(Compressor):
    def compress(self, x):
        return x

class RandKCompressor(Compressor):
    def __init__(self, k = 3):
        self.k = k

    def compress(self, x):
        return self.decomp_rand_k(*self.comp_rand_k(x))
    
    def comp_rand_k(self, logits):
        seed = random.randint(0, 1000)
        torch.manual_seed(seed)
        size = logits.size()
        new_size = size[:-1] + (self.k,)
        sparse_logits = torch.zeros(new_size, dtype= logits.dtype)
        for i in range(size[0]):
            for j in range(size[1]):
                indices = torch.randperm(size[-1], device=logits.device)[:k]
                indices, _ = torch.sort(indices)
                sparse_logits[i, j] = logits[i, j, indices]
        sparse_logits *= size[-1] / self.k
        return sparse_logits, seed, size

    def decomp_rand_k(self, sparse_logits, seed, size):
        torch.manual_seed(seed)
        decomp_logits  = torch.full(size, float('-inf') ,dtype= sparse_logits.dtype)
        for i in range(size[0]):
            for j in range(size[1]):
                indices = torch.randperm(size[-1], device=sparse_logits.device)[:self.k]
                indices, _ = torch.sort(indices)
                decomp_logits[i, j, indices] = sparse_logits[i, j]
        return decomp_logits

class RandKCompressor(Compressor):
    def __init__(self, k = 3):
        self.k = k

    def compress(self, x):
        return self.decomp_top_k(*self.comp_top_k(x))
    
    def comp_top_k(self, logits):
        size = logits.size()
        _, indices = torch.topk(logits, self.k, dim= -1)
        sparse_logits = torch.gather(logits, dim=-1, index=indices).to(logits.device)
        return sparse_logits, indices, size

    def decomp_top_k(self, sparse_logits, indices, size):
        decomp_logits  = torch.full(size, float('-inf'), dtype= sparse_logits.dtype, device= sparse_logits.device)
        decomp_logits.scatter_(-1, indices, sparse_logits)
        return decomp_logits

class ExpDitheringCompressor(Compressor):
    def __init__(self, b = 2, s = 8, p = 2):
        self.b = b
        self.s = s 
        self.p = p
    
    def compress(self, logits):
        norm_logits = torch.norm(logits.float(), self.p, dim=-1, keepdim=True)
        normalized_logits = logits.abs() / norm_logits
        
        levels = self.b ** -torch.arange(1, self.s + 1, dtype=logits.dtype, device=logits.device)  
        levels = torch.cat([torch.tensor([1.0], device=logits.device, dtype=logits.dtype), levels, torch.tensor([0.0], device=logits.device, dtype=logits.dtype)])
        u = torch.sum(normalized_logits.unsqueeze(-1) <= levels.unsqueeze(0), dim=-1, dtype= float) 
        
        lower = self.b ** -u
        upper = self.b ** -(u - 1)
        lower = torch.where(u == 10, 0.0, lower)
        
        probs_upper = (normalized_logits - lower) / (upper - lower)
        bern = torch.bernoulli(probs_upper)
        quantized = torch.where(bern == 1, upper, lower)
        result = norm_logits * torch.sign(logits) * quantized
        return result

class DifferentiableCompress(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, compressor: Compressor = IdenticalCompressor()):
        return compressor.compress(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
    
def compress_model(model, compressor: Compressor = IdenticalCompressor()):
    return nn.Sequential(
        model, 
        compressor
    )
    