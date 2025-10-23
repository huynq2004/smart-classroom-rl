# algorithms/common/noisy_layers.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyLinear(nn.Module):
    """
    Noisy linear layer from Fortunato et al. (NoisyNet).
    Factorised gaussian noise variant.
    """
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init)

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init)

    def _f(self, x):
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        # factorised gaussian noise
        in_noise = torch.randn(self.in_features)
        out_noise = torch.randn(self.out_features)
        f_in = self._f(in_noise)
        f_out = self._f(out_noise)
        self.weight_epsilon = f_out.ger(f_in)  # outer product
        self.bias_epsilon = f_out

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon.to(self.weight_mu.device)
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon.to(self.bias_mu.device)
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)
