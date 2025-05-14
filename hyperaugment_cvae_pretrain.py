
from __future__ import division
from __future__ import print_function
import argparse
import numpy as np
import scipy.sparse as sp
import sys
import copy
import random
import torch.optim as optim
import pickle
import os
import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
from tqdm import trange, tqdm
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import dhg

from dhg import Hypergraph
from dhg.nn import HyperGCNConv
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator
from torch import Tensor
from torch.nn import Linear
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_add
from scatter_func import scatter
from torch_geometric.typing import Adj, Size, OptTensor
from typing import Optional
from sklearn.metrics import accuracy_score, f1_score

import random
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from scipy.sparse import csr_matrix
import gc
torch.manual_seed(42)
np.random.seed(42)
from utils import accuracy, normalize_features

import torch
import torch.nn as nn
import torch.nn.functional as F
import dhg

from dhg.nn import HGNNConv, HyperGCNConv
import math
from dhg.structure.graphs import Graph


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HyperGCN(nn.Module):
    r"""The HGNN model proposed in `Hypergraph Neural Networks <https://arxiv.org/pdf/1809.09401>`_ paper (AAAI 2019).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to 0.5.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        use_mediator: bool = False,
        use_bn: bool = False,
        fast: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.fast = fast
        self.cached_g = None
        self.with_mediator = use_mediator
        self.layers = nn.ModuleList()
        self.layers.append(
            HyperGCNConv(in_channels, hid_channels, use_mediator, use_bn=use_bn, drop_rate=drop_rate)
        )
        self.layers.append(
            HyperGCNConv(hid_channels, num_classes, use_mediator, use_bn=use_bn, is_last=True)
        )

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        # if self.fast:
        #     if self.cached_g is None:
        #         self.cached_g = Graph.from_hypergraph_hypergcn(
        #             hg, X, self.with_mediator
        #         )

        #     for layer in self.layers:
        #         X = layer(X, hg, self.cached_g)
        # else:
        for layer in self.layers:
            X = layer(X, hg)
        return X
    

class CVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, conditional=False, conditional_size=0):
        super(CVAE, self).__init__()
        self.conditional = conditional
        # if self.conditional:
        #     latent_dim += conditional_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # self.latent_dim = latent_dim

        # hypergraph convolution
        # self.hg_conv = HGNNConv(input_dim+conditional_size, hidden_dim)
        self.fc1 = nn.Linear(input_dim+conditional_size, hidden_dim)
        # Encoder
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.sigma = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.fc2 = nn.Linear(input_dim+latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x, c=None):
        if self.conditional:
            x = torch.cat((x, c), dim=-1)
        h1 = F.relu(self.fc1(x))
        return self.mu(h1), self.sigma(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, c):
        if self.conditional:
            z = torch.cat((z, c), dim=-1)
        
        h2 = self.fc2(z)
        h2 = F.relu(h2)
        return F.sigmoid(self.fc3(h2))
        

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        # print(z.shape)
        return self.decode(z, c), mu, logvar, z
    
    def inference(self, z, c):
        recon_x = self.decode(z, c)
        return recon_x


def adjacency_matrix(hg, s=1, weight=False):
        r"""
        The :term:`s-adjacency matrix` for the dual hypergraph.

        Parameters
        ----------
        s : int, optional, default 1

        Returns
        -------
        adjacency_matrix : scipy.sparse.csr.csr_matrix

        """
        
        tmp_H = hg.H.to_dense().numpy()
        A = tmp_H @ (tmp_H.T)
        A[np.diag_indices_from(A)] = 0
        if not weight:
            A = (A >= s) * 1

        del tmp_H
        gc.collect()

        return csr_matrix(A)

def feature_tensor_normalize(feature):
    # feature = torch.tensor(feature)
    rowsum = torch.div(1.0, torch.sum(feature, dim=1))
    rowsum[torch.isinf(rowsum)] = 0.
    feature = torch.mm(torch.diag(rowsum), feature)
    return feature


def consis_loss(logps):
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p/len(ps)
    #p2 = torch.exp(logp2)
    
    sharp_p = (torch.pow(avg_p, 1./0.5) / torch.sum(torch.pow(avg_p, 1./0.5), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        loss += torch.mean((p-sharp_p).pow(2).sum(1))
    loss = loss/len(ps)
    return 1.0 * loss

def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return (BCE + KLD) / x.size(0)

def normalize_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def neighbor_of_node(adj_matrix, node):
    # Find the row corresponding to node i in the adjacency matrix
    node_row = adj_matrix[node, :].toarray().flatten()

    # Find the column index corresponding to the non-zero element, i.e. the neighbour node
    neighbors = np.nonzero(node_row)[0]
    return neighbors.tolist()

def aug_features_concat(concat, features, cvae_model):
    X_list = []
    cvae_features = torch.tensor(features, dtype=torch.float32).to(device)
    for _ in range(concat):
        z = torch.randn([cvae_features.size(0), 8]).to(device)
        augmented_features = cvae_model.inference(z, cvae_features)
        augmented_features = feature_tensor_normalize(augmented_features).detach()
        
        X_list.append(augmented_features.to(device))
        
    return X_list

def get_augmented_features(args, hg, features, labels, idx_train, features_normalized, device):
    adj = adjacency_matrix(hg, s=1, weight=False)
    num_nodes = adj.shape[0]

    # Precompute neighbors for all nodes
    neighbors = [neighbor_of_node(adj, i) for i in range(num_nodes)]
    neighbors = [n if len(n) > 0 else [i] for i, n in enumerate(neighbors)]

    # Use list comprehension for batch operations and avoid looping
    x_list = [features[neighbor].cpu().numpy().reshape(-1, features.shape[1]) for neighbor in neighbors]
    c_list = [np.tile(features[i].cpu().numpy(), (len(neighbor), 1)) for i, neighbor in enumerate(neighbors)]
    
    features_x = torch.tensor(np.vstack(x_list), dtype=torch.float32, device=device)
    features_c = torch.tensor(np.vstack(c_list), dtype=torch.float32, device=device)

    cvae_features = torch.tensor(features, dtype=torch.float32).to(device)
    
    # Initialize DataLoader
    cvae_dataset = TensorDataset(features_x, features_c)
    cvae_dataset_dataloader = DataLoader(cvae_dataset, batch_size=32, sampler=RandomSampler(cvae_dataset), num_workers=4)

    # Model parameters setup
    hidden = args.hidden
    dropout = args.dropout
    pretrain_lr = args.pretrain_lr  # 添加这一行定义 pretrain_lr
    lr = pretrain_lr
    weight_decay = args.weight_decay
    epochs = args.epochs

    # 根据 args.model 选择不同的模型
    if args.model == 'hypergcn':
        model = HyperGCN(in_channels=features.shape[1], hid_channels=hidden, 
                        num_classes=labels.max().item()+1, use_mediator=args.use_mediator, 
                        use_bn=True, drop_rate=dropout).to(device)
    elif args.model == 'hgnn':
        from dhg.models import HGNN
        model = HGNN(in_channels=features.shape[1], hid_channels=hidden, 
                    num_classes=labels.max().item()+1, use_bn=True, 
                    drop_rate=dropout).to(device)
    elif args.model == 'unigcn':
        from dhg.models import UniGCN
        model = UniGCN(in_channels=features.shape[1], hid_channels=hidden, 
                      num_classes=labels.max().item()+1, use_bn=True, 
                      drop_rate=dropout).to(device)
    elif args.model == 'unisage':
        from dhg.models import UniSAGE
        model = UniSAGE(in_channels=features.shape[1], hid_channels=hidden, 
                       num_classes=labels.max().item()+1, use_bn=True, 
                       drop_rate=dropout).to(device)
    elif args.model == 'unigin':
        from dhg.models import UniGIN
        model = UniGIN(in_channels=features.shape[1], hid_channels=hidden, 
                      num_classes=labels.max().item()+1, use_bn=True, 
                      drop_rate=dropout).to(device)
    elif args.model == 'unigat':
        from dhg.models import UniGAT
        model = UniGAT(in_channels=features.shape[1], hid_channels=hidden, 
                      num_classes=labels.max().item()+1, num_heads=args.num_heads, 
                      use_bn=True, drop_rate=dropout).to(device)
    else:
        raise ValueError(f"Model {args.model} not supported")
        
    model_optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    features_normalized, hg, cvae_features, labels, idx_train = [x.to(device) for x in (features_normalized, hg, cvae_features, labels, idx_train)]

    for _ in range(int(epochs / 2)):
        model.train()
        model_optimizer.zero_grad()
        output = model(features_normalized, hg)
        output = torch.log_softmax(output, dim=1)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        loss_train.backward()
        model_optimizer.step()

    # Pretraining CVAE
    cvae = CVAE(features.shape[1], 256, args.latent_size, True, features.shape[1]).to(device)
    cvae_optimizer = optim.Adam(cvae.parameters(), lr=args.pretrain_lr)

    best_augmented_features = None
    best_score = -float("inf")

    for epoch in trange(args.pretrain_epochs, desc=f'Run CVAE Train for {args.dataset} with {args.model}'):
        for x, c in tqdm(cvae_dataset_dataloader):
            cvae.train()
            x, c = x.to(device), c.to(device)
            recon_x, mean, log_var, _ = cvae(x, c)
            
            cvae_loss = loss_fn(recon_x, x, mean, log_var)
            cvae_optimizer.zero_grad()
            cvae_loss.backward()
            cvae_optimizer.step()

        # 每个 epoch 结束后评估模型
        z = torch.randn([cvae_features.size(0), args.latent_size]).to(device)
        augmented_feats = cvae.inference(z, cvae_features)
        augmented_feats = feature_tensor_normalize(augmented_feats)

        # 计算模型评分
        model_outputs = []
        for _ in range(args.num_models):
            model_outputs.append(model(augmented_feats, hg))
        
        total_logits = sum(model_outputs)
        output = torch.log_softmax(total_logits / args.num_models, dim=1)
        
        # 计算 U_score
        train_loss = F.nll_loss(output[idx_train], labels[idx_train])
        model_losses = [F.nll_loss(torch.log_softmax(out, dim=1), labels) for out in model_outputs]
        avg_model_loss = sum(model_losses) / args.num_models
        U_score = avg_model_loss - train_loss

        if U_score > best_score: 
            best_score = U_score
            best_augmented_features = augmented_feats.clone().detach()
            
            # 更新模型
            for _ in range(args.update_epochs):
                model.train()
                model_optimizer.zero_grad()
                output = model(best_augmented_features, hg)
                loss_train = F.nll_loss(torch.log_softmax(output[idx_train], dim=1), labels[idx_train])
                loss_train.backward()
                model_optimizer.step()
    
    # 保存模型，包含数据集名称和模型类型
    model_path = f"model/{args.dataset}_{args.model}_cvae.pkl"
    torch.save(cvae, model_path)
    print(f"CVAE model saved to {model_path}")
    
    return best_augmented_features, cvae
# def get_augmented_features(args, hg, features, labels, idx_train, features_normalized, device):
#     adj = adjacency_matrix(hg, s=1, weight=False)
#     num_nodes = adj.shape[0]

#     # Precompute neighbors for all nodes
#     neighbors = [neighbor_of_node(adj, i) for i in range(num_nodes)]
#     neighbors = [n if len(n) > 0 else [i] for i, n in enumerate(neighbors)]

#     # Use list comprehension for batch operations and avoid looping
#     x_list = [features[neighbor].cpu().numpy().reshape(-1, features.shape[1]) for neighbor in neighbors]
#     c_list = [np.tile(features[i].cpu().numpy(), (len(neighbor), 1)) for i, neighbor in enumerate(neighbors)]
    
#     features_x = torch.tensor(np.vstack(x_list), dtype=torch.float32, device=device)
#     features_c = torch.tensor(np.vstack(c_list), dtype=torch.float32, device=device)

#     cvae_features = torch.tensor(features, dtype=torch.float32).to(device)
    
#     # Initialize DataLoader
#     cvae_dataset = TensorDataset(features_x, features_c)
#     cvae_dataset_dataloader = DataLoader(cvae_dataset, batch_size=32, sampler=RandomSampler(cvae_dataset), num_workers=4)

#     # Model parameters setup
#     hidden = args.hidden
#     dropout = args.dropout
#     lr = pretrain_lr
#     weight_decay = args.weight_decay
#     epochs = args.epochs

#     model = HyperGCN(in_channels=features.shape[1], hid_channels=hidden, 
#                      num_classes=labels.max().item()+1, use_mediator=False, 
#                      use_bn=True, drop_rate=dropout).to(device)
#     model_optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

#     features_normalized, hg, cvae_features, labels, idx_train = [x.to(device) for x in (features_normalized, hg, cvae_features, labels, idx_train)]

#     for _ in range(int(epochs / 2)):
#         model.train()
#         model_optimizer.zero_grad()
#         output = model(features_normalized, hg)
#         output = torch.log_softmax(output, dim=1)
#         loss_train = F.nll_loss(output[idx_train], labels[idx_train])
#         loss_train.backward()
#         model_optimizer.step()

#     # Pretraining CVAE
#     cvae = CVAE(features.shape[1], 256, args.latent_size, True, features.shape[1]).to(device)
#     cvae_optimizer = optim.Adam(cvae.parameters(), lr=args.pretrain_lr)

#     best_augmented_features = None
#     best_score = -float("inf")

#     for epoch in trange(args.pretrain_epochs, desc='Run CVAE Train'):
#         for x, c in tqdm(cvae_dataset_dataloader):
#             cvae.train()
#             x, c = x.to(device), c.to(device)
#             recon_x, mean, log_var, _ = cvae(x, c)
            
#             cvae_loss = loss_fn(recon_x, x, mean, log_var)
#             cvae_optimizer.zero_grad()
#             cvae_loss.backward()
#             cvae_optimizer.step()

#             z = torch.randn([cvae_features.size(0), args.latent_size]).to(device)
#             augmented_feats = cvae.inference(z, cvae_features)
#             augmented_feats = feature_tensor_normalize(augmented_feats)

#             total_logits = sum(model(augmented_feats, hg) for _ in range(args.num_models))
#             output = torch.log(total_logits / args.num_models)
#             U_score = F.nll_loss(output[idx_train], labels[idx_train]) - (sum(F.nll_loss(model(augmented_feats, hg), labels) for _ in range(args.num_models)) / args.num_models)

#             if U_score > best_score: 
#                 best_score = U_score
#                 best_augmented_features = augmented_feats.clone().detach().requires_grad_(True)
#                 # Update model if in warmup
#                 for _ in range(args.update_epochs):
#                     model.train()
#                     model_optimizer.zero_grad()
#                     output = model(best_augmented_features, hg)
#                     loss_train = F.nll_loss(torch.log_softmax(output[idx_train], dim=1), labels[idx_train])
#                     loss_train.backward()
#                     model_optimizer.step()
                
#         return best_augmented_features, cvae


