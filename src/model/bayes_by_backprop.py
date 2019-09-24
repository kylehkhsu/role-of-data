# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# import math
# from functools import partial
# import numpy as np
# import ipdb
#
# neg_half_log_2_pi = -0.5 * math.log(2.0 * math.pi)
# softplus = lambda x: math.log(1 + math.exp(x))
# std_to_rho = lambda x: math.log(math.exp(x) - 1)
#
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
# class BNNLinearLayer(nn.Module):
#     def __init__(self, n_input, n_output, activation, W_prior_mean, W_prior_std, b_prior_mean, b_prior_std):
#         super(BNNLinearLayer, self).__init__()
#         self.n_input = n_input
#         self.n_output = n_output
#
#         W_prior_rho = std_to_rho(W_prior_std)
#         b_prior_rho = std_to_rho(b_prior_std)
#
#         self.W_mean = nn.Parameter(torch.ones([self.n_input, self.n_output]) * W_prior_mean)
#         self.W_rho = nn.Parameter(torch.ones([self.n_input, self.n_output]) * W_prior_rho)
#         # Blundell: std = softplus(rho)
#
#         self.b_mean = nn.Parameter(torch.ones([self.n_output]) * b_prior_mean)
#         self.b_rho = nn.Parameter(torch.ones([self.n_output]) * b_prior_rho)
#
#         # self.W_mean = nn.Parameter(torch.Tensor(self.n_input, self.n_output).uniform_(-0.1, 0.1))
#         # self.W_rho = nn.Parameter(torch.Tensor(self.n_input, self.n_output).uniform_(-2, -1.9))
#
#         # self.b_mean = nn.Parameter(torch.Tensor(self.n_output).uniform_(-0.1, 0.1))
#         # self.b_rho = nn.Parameter(torch.Tensor(self.n_output).uniform_(-2, -1.9))
#
#         # Blundell: don't optimize prior via backprop, only by hyperparameter search
#         # Blundell: 0 mean prior, isotropic variance on weights (and biases?)
#         # TODO Blundell: weight prior is scale mixture of two Gaussians, mixing factor a hyperparameter
#         # local reparameterization trick: https://arxiv.org/pdf/1506.02557.pdf
#         # instead of injecting noise on weights, inject noise on pre-activations; shouldn't affect calculation of KL
#
#         self.register_buffer('W_prior_var', torch.Tensor([W_prior_std ** 2]))
#         self.register_buffer('W_prior_mean', torch.Tensor([W_prior_mean]))
#
#         self.register_buffer('b_prior_var', torch.Tensor([b_prior_std ** 2]))
#         self.register_buffer('b_prior_mean', torch.Tensor([b_prior_mean]))
#
#         self.activation = None
#         if activation == 'relu':
#             self.activation = F.relu
#         elif activation == 'softmax':
#             self.activation = partial(F.softmax, dim=1)
#         else:
#             raise ValueError
#
#     def forward(self, x, mode):
#         assert x.dim() == 2
#         batch_size = x.shape[0]
#
#         assert mode in ['forward', 'MAP', 'MC']
#
#         z_mean = x.mm(self.W_mean) + self.b_mean
#
#         if mode == 'MAP':
#             return self.activation(z_mean)
#
#         # sample from standard normal independently for each example
#         z_noise = torch.randn([batch_size, self.n_output], requires_grad=False).to(device)
#         z_std = (x.pow(2).mm(F.softplus(self.W_rho).pow(2)) + F.softplus(self.b_rho).pow(2)).sqrt()
#         z = z_mean + z_std * z_noise
#         z = self.activation(z)
#
#         if mode == 'MC':
#             return z
#
#         # KL[q(w|\theta) || p(w)]
#         W_kl = BNNLinearLayer.kl_between_gaussians(
#             self.W_mean, F.softplus(self.W_rho).pow(2), self.W_prior_mean, self.W_prior_var
#         )
#         b_kl = BNNLinearLayer.kl_between_gaussians(
#             self.b_mean, F.softplus(self.b_rho).pow(2), self.b_prior_mean, self.b_prior_var
#         )
#         layer_kl = W_kl.sum() + b_kl.sum()
#         return z, layer_kl
#
#     @staticmethod
#     def kl_between_gaussians(p_mean, p_var, q_mean, q_var):
#         """KL(p||q)"""
#         return 0.5 * ((q_var / p_var).log() + (p_var + (p_mean - q_mean).pow(2)) / q_var - 1)
#
#
# class BNN(nn.Module):
#     def __init__(self, *layers):
#         super(BNN, self).__init__()
#
#         for i_layer, layer in enumerate(layers):
#             self.add_module(f'layer{i_layer}', layer)
#
#         self.layers = [module for name, module in self.named_modules() if 'layer' in name]
#
#     def forward(self, x, mode):
#         x = x.view([x.shape[0], -1])
#
#         if mode == 'forward':
#             net_kl = 0.0
#             for layer in self.layers:
#                 x, layer_kl = layer(x, mode)
#                 net_kl += layer_kl
#
#             return x, net_kl
#         else:
#             for layer in self.layers:
#                 x = layer(x, mode)
#             return x
#
#     def forward_train(self, x, y, n_samples):
#         x = x.view([x.shape[0], -1])
#         y = y.view([y.shape[0], -1])
#
#         running_kl = 0.0
#         running_log_likelihood = 0.0
#         for i in range(n_samples):
#             out, kl = self(x, 'forward')
#
#             # TODO: mean or sum?
#             log_likelihood = out.gather(1, y).log().mean()
#
#             running_kl += kl
#             running_log_likelihood += log_likelihood
#
#         return running_kl / n_samples, running_log_likelihood / n_samples
#
#     # TODO: double check
#     @staticmethod
#     def loss(kl, log_likelihood, dataset_size):
#         return kl / dataset_size - log_likelihood
#
#
# def make_bnn_mlp(n_input, n_output, hidden_layer_sizes, W_prior_mean, W_prior_std, b_prior_mean, b_prior_std):
#     layers = []
#     n_in = n_input
#     assert len(hidden_layer_sizes) > 0
#
#     for h in hidden_layer_sizes:
#         layers.append(
#             BNNLinearLayer(
#                 n_in, h, 'relu', W_prior_mean, W_prior_std, b_prior_mean, b_prior_std
#             )
#         )
#         n_in = h
#     layers.append(
#         BNNLinearLayer(
#             n_in, n_output, 'softmax', W_prior_mean, W_prior_std, b_prior_mean, b_prior_std
#         )
#     )
#
#     return BNN(*layers)