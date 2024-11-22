import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from rdkit.Chem import Descriptors
from torch_geometric.utils import to_networkx
import networkx as nx

class MSAMF(nn.Module):
    def __init__(self, num_atom_features, num_bond_features, fingerprint_dim=256):
        super(MSAMF, self).__init__()
        self.num_atom_features = num_atom_features
        self.num_bond_features = num_bond_features
        self.fingerprint_dim = fingerprint_dim

        # 原子级别特征提取（GCN）
        self.atom_conv1 = GATConv(num_atom_features, 64, heads=4, concat=True)
        self.atom_conv2 = GATConv(64*4, 128, heads=4, concat=True)

        # 自注意力机制（原子级别）
        self.atom_self_attn = nn.Sequential(
            nn.Linear(128*4, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # 子结构级别特征提取
        self.substructure_linear = nn.Linear(256, 128)

        # 全局分子特征提取
        self.global_linear = nn.Linear(4, 128)  # 示例中提取4个全局特征

        # 交叉注意力机制
        self.cross_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4)

        # 指纹生成
        self.fingerprint_linear = nn.Sequential(
            nn.Linear(128, fingerprint_dim),
            nn.ReLU(),
            nn.Linear(fingerprint_dim, fingerprint_dim)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # 原子级别特征提取
        x = F.relu(self.atom_conv1(x, edge_index))
        x = F.relu(self.atom_conv2(x, edge_index))

        # 自注意力机制
        attn_weights = self.atom_self_attn(x)
        attn_weights = F.softmax(attn_weights, dim=0)
        atom_representation = x * attn_weights

        # 全局池化
        atom_representation = global_mean_pool(atom_representation, batch)

        # 子结构级别特征提取
        substructure_representation = self.extract_substructures(data)
        substructure_representation = F.relu(self.substructure_linear(substructure_representation))

        # 全局分子特征提取
        global_features = self.compute_global_features(data)
        global_representation = F.relu(self.global_linear(global_features))

        # 交叉注意力机制
        atom_representation = atom_representation.unsqueeze(0)  # (1, batch_size, feature_dim)
        substructure_representation = substructure_representation.unsqueeze(0)
        global_representation = global_representation.unsqueeze(0)

        multi_scale_representation = torch.cat([atom_representation, substructure_representation, global_representation], dim=0)

        cross_attn_output, _ = self.cross_attn(multi_scale_representation, multi_scale_representation, multi_scale_representation)

        # 指纹生成
        fingerprint = self.fingerprint_linear(cross_attn_output.mean(dim=0))

        return fingerprint

    def extract_substructures(self, data):
        # 示例：简单地对节点特征求和作为子结构表示
        # 实际应用中，可以使用 Motif 提取或预定义的官能团识别
        batch = data.batch
        x = data.x
        substructure_representation = global_mean_pool(x, batch)
        return substructure_representation

    def compute_global_features(self, data):
        # 使用 RDKit 计算全局分子特征
        global_features = []
        for mol in data.mols:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            features = [mw, logp, tpsa, num_rotatable_bonds]
            global_features.append(features)
        global_features = torch.tensor(global_features, dtype=torch.float)
        return global_features.to(data.x.device)

