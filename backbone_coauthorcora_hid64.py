import time
import csv
from copy import deepcopy
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from dhg import Hypergraph
from dhg.data import *
from dhg.models import *
from dhg.random import set_seed
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator

data = CoauthorshipCora()
G = Hypergraph(data["num_vertices"], data["edge_list"])
print(G)


random_seed = 42

node_idx = [i for i in range(data['num_vertices'])]

labels = data["labels"]
y = labels.numpy()

idx_train, idx_temp, train_y, tem_y = train_test_split(node_idx, y, test_size=0.5, random_state=random_seed, stratify=y)
idx_val, idx_test, val_y, test_y = train_test_split(idx_temp, tem_y, test_size=0.5, random_state=random_seed, stratify=tem_y)

assert len(set(idx_train) & set(idx_val)) == 0
assert len(set(idx_train) & set(idx_test)) == 0
assert len(set(idx_val) & set(idx_test)) == 0

train_mask = torch.zeros(data['num_vertices'], dtype=torch.bool)
val_mask = torch.zeros(data['num_vertices'], dtype=torch.bool)
test_mask = torch.zeros(data['num_vertices'], dtype=torch.bool)
train_mask[idx_train] = True
val_mask[idx_val] = True
test_mask[idx_test] = True


X = data["features"]
lbls = data["labels"]
print('X dim:', X.shape)
print('labels:', len(torch.unique(lbls)))


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

set_seed(42)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
num_epochs = 500


X, lbls = X.to(device), lbls.to(device)
G = G.to(device)

############################-----HGNN Model-------##############################
# net = HGNN(X.shape[1], 64, data["num_classes"], use_bn=False)
# optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)
# net = net.to(device)

# print(f'net:{net}')

# best_state = None
# best_epoch, best_val = 0, 0

# all_acc, all_microf1, all_macrof1 = [],[],[]
# for run in range(5):
    

#     train_losses = []  
#     val_losses = []  
#     for epoch in range(num_epochs):
#         # train
#         net.train()
#         optimizer.zero_grad()
#         outs = net(X,G)
#         outs, lbl = outs[idx_train], lbls[idx_train]
#         loss = F.cross_entropy(outs, lbl)
#         loss.backward()
#         optimizer.step()
#         train_losses.append(loss.item())

#         # validation
#         net.eval()
#         with torch.no_grad():
#             outs = net(X,G)
#             outs, lbl = outs[idx_val], lbls[idx_val]
#             val_loss = F.cross_entropy(outs, lbl)
#             val_losses.append(val_loss.item())  

#             _, predicted = torch.max(outs, 1)
#             correct = (predicted == lbl).sum().item()
#             total = lbl.size(0)
#             val_acc = correct / total

#             if epoch % 10 == 0:
#                 print(f"Epoch: {epoch}, LR: {optimizer.param_groups[0]['lr']}, Loss: {loss.item():.5f}, Val Loss: {loss.item():.5f}, Validation Accuracy: {val_acc}")
            

#             # Save the model if it has the best validation accuracy
#             if val_acc > best_val:
#                 print(f"update best: {val_acc:.5f}")
#                 best_val = val_acc
#                 best_state = deepcopy(net.state_dict())
#                 torch.save(net.state_dict(), 'model/hgnn_coauthorshipcora.pth')
#         # scheduler.step()
#     print("\ntrain finished!")
#     print(f"best val: {best_val:.5f}")

#     if run == 0:
#         # 绘制曲线图
#         plt.plot(range(num_epochs), train_losses, label='Train Loss')
#         plt.plot(range(num_epochs), val_losses, label='Validation Loss')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.legend()
#         plt.savefig(f'./figs/cacora_hgnn_loss_hgnn.png')
#         # plt.show()

#     # test
#     print("test...")
#     net.load_state_dict(best_state)

#     net.eval()
#     with torch.no_grad():
#         outs = net(X, G)
#         outs, lbl = outs[idx_test], lbls[idx_test]
        
        
#         # Calculate accuracy
#         _, predicted = torch.max(outs, 1)

#         predicted_array = predicted.cpu().numpy()
        
#         correct = (predicted == lbl).sum().item()
#         total = lbl.size(0)
#         test_acc = correct / total
#         print(f'Test Accuracy: {test_acc}')

#         # Calculate micro F1
#         micro_f1 = f1_score(lbl.cpu(), predicted.cpu(), average='micro')
#         print(f'Micro F1: {micro_f1}')

#         # Calculate macro F1
#         macro_f1 = f1_score(lbl.cpu(), predicted.cpu(), average='macro')
#         print(f'Macro F1: {macro_f1}')

#     all_acc.append(test_acc)
#     all_microf1.append(micro_f1)
#     all_macrof1.append(macro_f1)

# # avg of 5 times
# print('Model HGNN Results:\n')
# print('test acc:', np.mean(all_acc), 'test acc std:', np.std(all_acc))
# print('\n')
# print('test macrof1:', np.mean(all_macrof1), 'test macrof1 std:', np.std(all_macrof1))

# print('='*100)


# #  保存到CSV文件
# with open('res/hgnn_cacora_results.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Metric', 'Mean', 'Standard Deviation'])
#     writer.writerow(['Test Accuracy', np.mean(all_acc), np.std(all_acc)])
#     # writer.writerow(['Micro F1', np.mean(all_microf1), np.std(all_microf1)]) # micro f1 == accuracy for multi-classification 
#     writer.writerow(['Macro F1', np.mean(all_macrof1), np.std(all_macrof1)])

############################-----HyperGCN Model-------##############################
# net = HyperGCN(X.shape[1], 64, data["num_classes"])
# optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)
# scheduler = StepLR(optimizer, step_size=int(num_epochs/5), gamma=0.01)
# net = net.to(device)

# print(f'net:\n')
# print(net)

# best_state = None
# best_epoch, best_val = 0, 0

# all_acc, all_microf1, all_macrof1 = [],[],[]
# for run in range(5):
#     train_losses = []  
#     val_losses = []  
#     for epoch in range(num_epochs):
#         # train
#         net.train()
#         optimizer.zero_grad()
#         outs = net(X,G)
#         outs, lbl = outs[idx_train], lbls[idx_train]
#         loss = F.cross_entropy(outs, lbl)
#         loss.backward()
#         optimizer.step()
#         train_losses.append(loss.item())

#         # validation
#         net.eval()
#         with torch.no_grad():
#             outs = net(X,G)
#             outs, lbl = outs[idx_val], lbls[idx_val]
#             val_loss = F.cross_entropy(outs, lbl)
#             val_losses.append(val_loss.item())  

#             _, predicted = torch.max(outs, 1)
#             correct = (predicted == lbl).sum().item()
#             total = lbl.size(0)
#             val_acc = correct / total

#             if epoch % 10 == 0:
#                 print(f"Epoch: {epoch}, LR: {optimizer.param_groups[0]['lr']}, Loss: {loss.item():.5f}, Val Loss: {loss.item():.5f}, Validation Accuracy: {val_acc}")
            

#             # Save the model if it has the best validation accuracy
#             if val_acc > best_val:
#                 print(f"update best: {val_acc:.5f}")
#                 best_val = val_acc
#                 best_state = deepcopy(net.state_dict())
#                 torch.save(net.state_dict(), 'model/hypergcn_coauthorshipcora.pth')
#         # scheduler.step()
#     print("\ntrain finished!")
#     print(f"best val: {best_val:.5f}")

#     if run == 0:
#         # 绘制曲线图
#         plt.plot(range(num_epochs), train_losses, label='Train Loss')
#         plt.plot(range(num_epochs), val_losses, label='Validation Loss')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.legend()
#         plt.savefig(f'./figs/cacora_hypergcn_loss_hgnn.png')
#         # plt.show()

#     # test
#     print("test...")
#     net.load_state_dict(best_state)

#     net.eval()
#     with torch.no_grad():
#         outs = net(X, G)
#         outs, lbl = outs[idx_test], lbls[idx_test]

#         # Calculate accuracy
#         _, predicted = torch.max(outs, 1)
#         correct = (predicted == lbl).sum().item()
#         total = lbl.size(0)
#         test_acc = correct / total
#         print(f'Test Accuracy: {test_acc}')

#         # Calculate micro F1
#         micro_f1 = f1_score(lbl.cpu(), predicted.cpu(), average='micro')
#         print(f'Micro F1: {micro_f1}')

#         # Calculate macro F1
#         macro_f1 = f1_score(lbl.cpu(), predicted.cpu(), average='macro')
#         print(f'Macro F1: {macro_f1}')

#     all_acc.append(test_acc)
#     all_microf1.append(micro_f1)
#     all_macrof1.append(macro_f1)

# # avg of 5 times
# print('Model HyperGCN Results:\n')
# print('test acc:', np.mean(all_acc), 'test acc std:', np.std(all_acc))
# print('\n')
# print('test macrof1:', np.mean(all_macrof1), 'test macrof1 std:', np.std(all_macrof1))
# print('='*150)


# #  保存到CSV文件
# with open('res/hypergcn_cacora_results.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Metric', 'Mean', 'Standard Deviation'])
#     writer.writerow(['Test Accuracy', np.mean(all_acc), np.std(all_acc)])
#     # writer.writerow(['Micro F1', np.mean(all_microf1), np.std(all_microf1)]) # micro f1 == accuracy for multi-classification 
#     writer.writerow(['Macro F1', np.mean(all_macrof1), np.std(all_macrof1)])



############################-----UniGIN Model-------##############################
# model_unigin = UniGIN(X.shape[1], 64, data["num_classes"], use_bn=False)
# optimizer = optim.Adam(model_unigin.parameters(), lr=0.001, weight_decay=5e-4)
# # scheduler = StepLR(optimizer, step_size=int(num_epochs/5), gamma=0.1)
# model_unigin = model_unigin.to(device)
# print(f'model: {model_unigin}')

# best_state = None
# best_epoch, best_val = 0, 0
# num_epochs = 500
# all_acc, all_microf1, all_macrof1 = [],[],[]
# for run in range(5):

    

#     train_losses = []  
#     val_losses = []  
#     for epoch in range(num_epochs):
#         # train
#         model_unigin.train()
#         optimizer.zero_grad()
#         outs = model_unigin(X,G)
#         outs, lbl = outs[idx_train], lbls[idx_train]
#         loss = F.cross_entropy(outs, lbl)
#         loss.backward()
#         optimizer.step()
#         train_losses.append(loss.item())

#         # validation
#         model_unigin.eval()
#         with torch.no_grad():
#             outs = model_unigin(X,G)
#             outs, lbl = outs[idx_val], lbls[idx_val]
#             val_loss = F.cross_entropy(outs, lbl)
#             val_losses.append(val_loss.item())  # 新增：记录val_loss

#             _, predicted = torch.max(outs, 1)
#             correct = (predicted == lbl).sum().item()
#             total = lbl.size(0)
#             val_acc = correct / total

#             if epoch % 10 == 0:
#                 print(f"Epoch: {epoch}, LR: {optimizer.param_groups[0]['lr']}, Loss: {loss.item():.5f}, Val Loss: {loss.item():.5f}, Validation Accuracy: {val_acc}")
            

#             # Save the model if it has the best validation accuracy
#             if val_acc > best_val:
#                 print(f"update best: {val_acc:.5f}")
#                 best_val = val_acc
#                 best_state = deepcopy(model_unigin.state_dict())
#                 torch.save(model_unigin.state_dict(), 'model/unigin_coauthorshipcora.pth')

#     print("\ntrain finished!")
#     print(f"best val: {best_val:.5f}")

#     if run == 0:
#         # 绘制曲线图
#         plt.plot(range(num_epochs), train_losses, label='Train Loss')
#         plt.plot(range(num_epochs), val_losses, label='Validation Loss')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.legend()
#         plt.savefig(f'./figs/cacora_unigin_loss_hgnn.png')
#         # plt.show()

    

#     # test
#     print("test...")
#     model_unigin.load_state_dict(best_state)

#     model_unigin.eval()
#     with torch.no_grad():
#         outs = model_unigin(X, G)
#         outs, lbl = outs[idx_test], lbls[idx_test]

#         # Calculate accuracy
#         _, predicted = torch.max(outs, 1)
#         correct = (predicted == lbl).sum().item()
#         total = lbl.size(0)
#         test_acc = correct / total
#         print(f'Test Accuracy: {test_acc}')

#         # Calculate micro F1
#         micro_f1 = f1_score(lbl.cpu(), predicted.cpu(), average='micro')
#         print(f'Micro F1: {micro_f1}')

#         # Calculate macro F1
#         macro_f1 = f1_score(lbl.cpu(), predicted.cpu(), average='macro')
#         print(f'Macro F1: {macro_f1}')

#     all_acc.append(test_acc)
#     all_microf1.append(micro_f1)
#     all_macrof1.append(macro_f1)

# # avg of 5 times
# print('Model UniGIN Results:\n')
# print('test acc:', np.mean(all_acc), 'test acc std:', np.std(all_acc))
# print('\n')
# print('test microf1:', np.mean(all_macrof1), 'test macrof1 std:', np.std(all_macrof1))
# print('='*200)


# #  保存到CSV文件
# with open('res/unigin_cacora_results.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Metric', 'Mean', 'Standard Deviation'])
#     writer.writerow(['Test Accuracy', np.mean(all_acc), np.std(all_acc)])
#     # writer.writerow(['Micro F1', np.mean(all_microf1), np.std(all_microf1)]) # micro f1 == accuracy for multi-classification 
#     writer.writerow(['Macro F1', np.mean(all_macrof1), np.std(all_macrof1)])



############################-----UniSAGE Model-------##############################
model_unisage = UniSAGE(X.shape[1], 64, data["num_classes"], use_bn=False)
optimizer = optim.Adam(model_unisage.parameters(), lr=0.001, weight_decay=5e-4)
# scheduler = StepLR(optimizer, step_size=int(num_epochs/5), gamma=0.1)
model_unisage = model_unisage.to(device)
print(f'model: {model_unisage}')

best_state = None
best_epoch, best_val = 0, 0
num_epochs = 500
all_acc, all_microf1, all_macrof1 = [],[],[]

for run in range(5):
    train_losses = []  
    val_losses = []  
    for epoch in range(num_epochs):
        # train
        model_unisage.train()
        optimizer.zero_grad()
        outs = model_unisage(X,G)
        outs, lbl = outs[idx_train], lbls[idx_train]
        loss = F.cross_entropy(outs, lbl)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # validation
        model_unisage.eval()
        with torch.no_grad():
            outs = model_unisage(X,G)
            outs, lbl = outs[idx_val], lbls[idx_val]
            val_loss = F.cross_entropy(outs, lbl)
            val_losses.append(val_loss.item())  

            _, predicted = torch.max(outs, 1)
            correct = (predicted == lbl).sum().item()
            total = lbl.size(0)
            val_acc = correct / total

            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, LR: {optimizer.param_groups[0]['lr']}, Loss: {loss.item():.5f}, Val Loss: {loss.item():.5f}, Validation Accuracy: {val_acc}")
            

            # Save the model if it has the best validation accuracy
            if val_acc > best_val:
                print(f"update best: {val_acc:.5f}")
                best_val = val_acc
                best_state = deepcopy(model_unisage.state_dict())
                torch.save(model_unisage.state_dict(), 'model/unisage_coauthorshipcora.pth')

    print("\ntrain finished!")
    print(f"best val: {best_val:.5f}")

    if run == 0:
        # 绘制曲线图
        plt.plot(range(num_epochs), train_losses, label='Train Loss')
        plt.plot(range(num_epochs), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'./figs/cacora_unisage_loss_hgnn.png')
        # plt.show()

    

    # test
    print("test...")
    model_unisage.load_state_dict(best_state)

    model_unisage.eval()
    with torch.no_grad():
        outs = model_unisage(X, G)
        outs, lbl = outs[idx_test], lbls[idx_test]

        # Calculate accuracy
        _, predicted = torch.max(outs, 1)
        correct = (predicted == lbl).sum().item()
        total = lbl.size(0)
        test_acc = correct / total
        print(f'Test Accuracy: {test_acc}')

        # Calculate micro F1
        micro_f1 = f1_score(lbl.cpu(), predicted.cpu(), average='micro')
        print(f'Micro F1: {micro_f1}')

        # Calculate macro F1
        macro_f1 = f1_score(lbl.cpu(), predicted.cpu(), average='macro')
        print(f'Macro F1: {macro_f1}')

    all_acc.append(test_acc)
    all_microf1.append(micro_f1)
    all_macrof1.append(macro_f1)

# avg of 5 times
print('Model UniSAGE Results:\n')
print('test acc:', np.mean(all_acc), 'test acc std:', np.std(all_acc))
print('\n')
print('test microf1:', np.mean(all_macrof1), 'test macrof1 std:', np.std(all_macrof1))

#  保存到CSV文件
with open('res/unisage_cacora_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Metric', 'Mean', 'Standard Deviation'])
    writer.writerow(['Test Accuracy', np.mean(all_acc), np.std(all_acc)])
    # writer.writerow(['Micro F1', np.mean(all_microf1), np.std(all_microf1)]) # micro f1 == accuracy for multi-classification 
    writer.writerow(['Macro F1', np.mean(all_macrof1), np.std(all_macrof1)])
