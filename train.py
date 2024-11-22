import sys
print("sys.path:",sys.path)
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from data_preprocessing import load_dataset
import msamf_model
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data


# 加载数据集
data_list = load_dataset('dataset-1.csv')

if data_list[0].edge_attr is not None:
    num_bond_features = data_list[0].edge_attr.shape[1]
else:
    print("Warning: edge_attr is None")
    num_bond_features = 0  # 或者根据需要设为默认值


# 添加 mols 属性用于计算全局特征
for data in data_list:
    mol = Chem.MolFromSmiles(data.smiles)
    data.mols = [mol]

# 划分训练集和测试集
train_dataset = data_list[:int(0.8 * len(data_list))]
test_dataset = data_list[int(0.8 * len(data_list)):]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 模型实例化
num_atom_features = data_list[0].x.shape[1]
num_bond_features = data_list[0].edge_attr.shape[1]
model = MSAMF(num_atom_features, num_bond_features)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

# 测试循环
def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        out = model(data)
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
    return correct / len(loader.dataset)

# 主训练过程
for epoch in range(1, 51):
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
