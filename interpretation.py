import torch
from rdkit.Chem import Draw

def visualize_attention(data, model):
    model.eval()
    x, edge_index = data.x, data.edge_index
    # 获取注意力权重
    with torch.no_grad():
        x = F.relu(model.atom_conv1(x, edge_index))
        x = F.relu(model.atom_conv2(x, edge_index))
        attn_weights = model.atom_self_attn(x)
        attn_weights = F.softmax(attn_weights, dim=0)

    # 将注意力权重映射到原子上
    attn_weights = attn_weights.squeeze().numpy()
    mol = Chem.MolFromSmiles(data.smiles)

    # 生成原子着色
    atom_contrib = {}
    for idx, weight in enumerate(attn_weights):
        atom_contrib[idx] = weight

    # 可视化分子，并突出显示重要原子
    fig = Draw.MolToMPL(mol, size=(300, 300), highlightAtoms=atom_contrib.keys(),
                        highlightAtomColors={k: (1, 0, 0, v) for k, v in atom_contrib.items()})
    fig.savefig('attention_visualization.png')

# 示例用法
data = data_list[0]
visualize_attention(data, model)
