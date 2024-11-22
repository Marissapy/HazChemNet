import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # 原子特征
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([atom.GetAtomicNum()])  # 仅使用原子序数作为特征
    atom_features = torch.tensor(atom_features, dtype=torch.float)

    # 边索引和边特征
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append([start, end])
        edge_index.append([end, start])

        # 边特征：键类型
        bond_type = bond.GetBondType()
        bond_features = [
            bond_type == Chem.rdchem.BondType.SINGLE,
            bond_type == Chem.rdchem.BondType.DOUBLE,
            bond_type == Chem.rdchem.BondType.TRIPLE,
            bond_type == Chem.rdchem.BondType.AROMATIC,
        ]
        edge_attr.append(bond_features)
        edge_attr.append(bond_features)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float) if edge_attr else None

    # 构建图数据
    data = Data(x=atom_features, edge_index=edge_index)
    if edge_attr is not None:
        data.edge_attr = edge_attr

    return data

def load_dataset(csv_file):
    import pandas as pd
    df = pd.read_csv(csv_file)
    data_list = []
    for index, row in df.iterrows():
        smiles = row['SMILES']
        data = smiles_to_graph(smiles)
        if data is not None:
            data.smiles = smiles
            data.y = torch.tensor([row['haz']], dtype=torch.long)
            data_list.append(data)
        else:
            print(f"Invalid SMILES: {smiles}")
    return data_list

# 加载数据集并验证
data_list = load_dataset('dataset-1.csv')

# 检查第一个样本的边特征
if data_list[0].edge_attr is not None:
    num_bond_features = data_list[0].edge_attr.shape[1]
else:
    print("Warning: edge_attr is None")
    num_bond_features = 0

print(f"Number of bond features: {num_bond_features}")
