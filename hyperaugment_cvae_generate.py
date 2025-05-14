import argparse
import numpy as np
import scipy.sparse as sp
import torch
import sys
import random
import torch.nn.functional as F
import torch.optim as optim
import os

import hypergcn_cvae_pretrain_new_coauthorcora
import hgnn_cvae_pretrain_new_coauthorcora
import unigcn_cvae_pretrain_coauthorshipcora
import unisage_cvae_pretrain_coauthorshipcora
import unigin_cvae_pretrain_coauthorshipcora
import unigat_cvae_pretrain_coauthorshipcora

from utils import load_data, accuracy, normalize_adj, normalize_features, sparse_mx_to_torch_sparse_tensor
from models import *
from tqdm import trange
import dhg
from dhg.data import *
from dhg import Hypergraph
from dhg.nn import HyperGCNConv
from sklearn.model_selection import train_test_split


exc_path = sys.path[0]

parser = argparse.ArgumentParser()
parser.add_argument("--pretrain_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--latent_size", type=int, default=20)
parser.add_argument("--pretrain_lr", type=float, default=0.001 )
parser.add_argument("--conditional", action='store_true', default=True)
parser.add_argument('--update_epochs', type=int, default=20, help='Update training epochs')
parser.add_argument('--num_models', type=int, default=100, help='The number of models for choice')
parser.add_argument('--warmup', type=int, default=200, help='Warmup')
parser.add_argument('--runs', type=int, default=3, help='The number of experiments.')

parser.add_argument('--dataset', default='coauthorshipcora',
                    help='Dataset string: coauthorshipcora, cora, citeseer, 20newsgroups, housecommittees, senatebills, contact-high-school')
parser.add_argument('--model', type=str, default='hypergcn',
                    help='Model to use: hypergcn, hgnn, unigcn, unisage, unigin, unigat')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=400,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--num_heads', type=int, default=8,
                    help='Number of attention heads for UniGAT.')
parser.add_argument('--use_mediator', action='store_true', default=False, 
                    help='Whether to use mediator to transform the hyperedges to edges in the graph.')

args = parser.parse_args()


torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.cuda = torch.cuda.is_available()

print('args:\n', args)

# setup device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# make sure the model folder exists
os.makedirs("model", exist_ok=True)


# Load data
print(f"Loading dataset: {args.dataset}")
if args.dataset == 'coauthorshipcora':
    data = CoauthorshipCora()
    hg = Hypergraph(data["num_vertices"], data["edge_list"])
    X = data["features"].to(device)
    features = data["features"].numpy()
    labels = data["labels"]
elif args.dataset == 'cora':
    data = Cora()
    hg = Hypergraph(data["num_vertices"], data["edge_list"])
    X = data["features"].to(device)
    features = data["features"].numpy()
    labels = data["labels"]
elif args.dataset == 'citeseer':
    data = CiteSeer()
    hg = Hypergraph(data["num_vertices"], data["edge_list"])
    X = data["features"].to(device)
    features = data["features"].numpy()
    labels = data["labels"]
elif args.dataset == '20newsgroups':
    data = News20()
    hg = Hypergraph(data["num_vertices"], data["edge_list"])
    X = data["features"].to(device)
    features = data["features"].numpy()
    labels = data["labels"]
elif args.dataset == 'housecommittees':
    data = HouseCommittees()
    hg = Hypergraph(data["num_vertices"], data["edge_list"])
    X = data["features"].to(device)
    features = data["features"].numpy()
    labels = data["labels"]
elif args.dataset in ['senatebills', 'contact-high-school']:
    # 从本地文件加载数据
    adj, features, labels, idx_train, idx_val, idx_test = load_data(f"data/{args.dataset}")
    # 创建超图
    num_nodes = features.shape[0]
    edge_list = []
    for i in range(adj.shape[1]):
        edge = np.where(adj[:, i] > 0)[0].tolist()
        if len(edge) > 1:  # 只添加包含多个节点的超边
            edge_list.append(edge)
    hg = Hypergraph(num_nodes, edge_list)
    X = torch.FloatTensor(features).to(device)
    data = {"num_vertices": num_nodes, "edge_list": edge_list, "features": torch.FloatTensor(features), "labels": torch.LongTensor(labels)}
else:
    raise ValueError(f"Dataset {args.dataset} not supported")

print(f"Dataset loaded: {args.dataset}")
print(hg)



X = data["features"].to(device)
# Normalize adj and features
features = data["features"].numpy()
features_normalized = normalize_features(features)
labels = data["labels"]
features_normalized = torch.FloatTensor(features_normalized).to(device)

random_seed = args.seed
node_idx = [i for i in range(data['num_vertices'])]
idx_train, idx_temp = train_test_split(node_idx, test_size=0.5, random_state=random_seed)
idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=random_seed)

assert len(set(idx_train) & set(idx_val)) == 0
assert len(set(idx_train) & set(idx_test)) == 0
assert len(set(idx_val) & set(idx_test)) == 0
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)


train_mask = torch.zeros(data['num_vertices'], dtype=torch.bool)
val_mask = torch.zeros(data['num_vertices'], dtype=torch.bool)
test_mask = torch.zeros(data['num_vertices'], dtype=torch.bool)
train_mask[idx_train] = True
val_mask[idx_val] = True
test_mask[idx_test] = True

# move data to device
if args.cuda:
    idx_train = idx_train.to(device)
    labels = torch.LongTensor(labels).to(device) if not isinstance(labels, torch.Tensor) else labels.to(device)


print(f"Training CVAE model for {args.model} on {args.dataset}...")
if args.model == 'hypergcn':
    cvae_augmented_features, cvae_model = hypergcn_cvae_pretrain_new_coauthorcora.get_augmented_features(
        args, hg, X, labels, idx_train, features_normalized, device)
    model_path = f"model/{args.dataset}_{args.model}.pkl"
elif args.model == 'hgnn':
    cvae_augmented_features, cvae_model = hgnn_cvae_pretrain_new_coauthorcora.get_augmented_features(
        args, hg, X, labels, idx_train, features_normalized, device)
    model_path = f"model/{args.dataset}_{args.model}.pkl"
elif args.model == 'unigcn':
    cvae_augmented_features, cvae_model = unigcn_cvae_pretrain_coauthorshipcora.get_augmented_features(
        args, hg, X, labels, idx_train, features_normalized, device)
    model_path = f"model/{args.dataset}_{args.model}.pkl"
elif args.model == 'unisage':
    cvae_augmented_features, cvae_model = unisage_cvae_pretrain_coauthorshipcora.get_augmented_features(
        args, hg, X, labels, idx_train, features_normalized, device)
    model_path = f"model/{args.dataset}_{args.model}.pkl"
elif args.model == 'unigin':
    cvae_augmented_features, cvae_model = unigin_cvae_pretrain_coauthorshipcora.get_augmented_features(
        args, hg, X, labels, idx_train, features_normalized, device)
    model_path = f"model/{args.dataset}_{args.model}.pkl"
elif args.model == 'unigat':
    cvae_augmented_features, cvae_model = unigat_cvae_pretrain_coauthorshipcora.get_augmented_features(
        args, hg, X, labels, idx_train, features_normalized, device)
    model_path = f"model/{args.dataset}_{args.model}.pkl"
else:
    raise ValueError(f"Model {args.model} not supported")

# save model
cvae_augmented_featuers, cvae_model = hypergcn_cvae_pretrain_new_coauthorcora.get_augmented_features(args, hg, X, labels, idx_train, features_normalized, device)
torch.save(cvae_model,model_path)
print(f"CVAE model saved to {model_path}")


