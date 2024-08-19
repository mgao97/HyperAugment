import argparse
import numpy as np
import scipy.sparse as sp
import torch
import sys
import random
import torch.nn.functional as F
import torch.optim as optim
import hypergcn_cvae_pretrain_new_coauthorcora

from utils import load_data, accuracy, normalize_adj, normalize_features, sparse_mx_to_torch_sparse_tensor
from hypergcn_cvae_pretrain_new_coauthorcora import HyperGCN
from tqdm import trange
import dhg
from dhg.data import CoauthorshipCora
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
                    help='Dataset string.')
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

parser.add_argument('--use_mediator', default=False, help='Whether to use mediator to transform the hyperedges to edges in the graph.')

args = parser.parse_args()


torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.cuda = torch.cuda.is_available()

print('args:\n', args)

# Load data

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# CA-Cora Data
data = CoauthorshipCora()
hg = Hypergraph(data["num_vertices"], data["edge_list"])
print(hg)

X = data["features"].to(device)

# Normalize adj and features
features = data["features"].numpy()
features_normalized = normalize_features(features)
labels = data["labels"]
features_normalized = torch.FloatTensor(features_normalized).to(device)


random_seed = 42

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

cvae_augmented_featuers, cvae_model = hypergcn_cvae_pretrain_new_coauthorcora.get_augmented_features(args, hg, X, labels, idx_train, features_normalized, device)
torch.save(cvae_model,"model/%s_hypergcn.pkl"%args.dataset)


