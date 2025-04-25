import torch
import torch.nn as nn
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.utils import softmax
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.loader import NeighborLoader, DataLoader
import pickle
import numpy as np



## UTILS ##################################################################################
def get_edge_relation_index(G):
    edge_index = []
    edge_type = {'rel': np.zeros((G.number_of_edges())), 'hierarchy': np.zeros((G.number_of_edges()))}
    # node_features = []
    node_map = {node: i for i, node in enumerate(G.nodes())}

    for i, (u, v, attr) in enumerate(G.edges(data=True)):
        edge_index.append([node_map[u], node_map[v]])
        if attr.get('rel') != 'na':
            edge_type['rel'][i] = 1
        elif attr.get('hierarchy') != 'na':
            edge_type['hierarchy'][i] = 1
        else:
            edge_type.append(-1)
    return torch.tensor(edge_index).t().contiguous(), torch.tensor(edge_type['rel']), torch.tensor(edge_type['hierarchy'])


def k_hop_subgraph(
    node_idx,
    num_hops,
    num_nodes,
    edge_idx,
    relabel_nodes= False,
    # flow= 'source_to_target',
    directed= False):

    # assert flow in ['source_to_target', 'target_to_source']
    # if flow == 'target_to_source':
    #     row, col = edge_index
    # else:
    #     col, row = edge_index
    
    row, col = edge_idx

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]
    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True

    if not directed:
        edge_mask = node_mask[row] & node_mask[col]

    edge_idx = edge_idx[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_idx = node_idx[edge_idx]   

    return subset, edge_idx, inv, edge_mask



def node_idx_list_map(node_map, nodes):
        return [node_map.get(i, 'None') for i in nodes if node_map.get(i, 'None') != 'None']



def get_subgraph(G, node_map, icd92cui, atc2cui, dataset, task, id):
    node_set = []
    for i in dataset.patients[id].visits.values():
        temp = node_idx_list_map(node_map, [icd92cui[code] for code in i.get_code_list('PROCEDURES_ICD')])
        node_set += temp
        temp = node_idx_list_map(node_map, [icd92cui[code] for code in i.get_code_list('DIAGNOSES_ICD')])
        node_set += temp
        temp = node_idx_list_map(node_map, [atc2cui[code] for code in i.get_code_list('PRESCRIPTIONS')])
        node_set += temp


    # while len(node_set) == 0:
    #     id -= 1
    #     patient = dataset[id]

    edge_index, _, _ = get_edge_relation_index(G)
    _, edge_idx, _, _ = k_hop_subgraph(node_set, 2, num_nodes, edge_index, relabel_nodes=False)
    # mask_idx = torch.where(edge_mask)[0]
    P = G.edge_subgraph(list(map(tuple, edge_idx.t().tolist())))
    # P = L.subgraph(node_set)

    if task.task_name == "drug_recommendation_mimic3_fn":
        label = [sample for sample in task if sample["patient_id"] == id][-1]['drugs']
        P.label = label
    elif task.task_name == "length_of_stay_prediction_mimic3_fn":
        label = np.zeros(10)
        day = [sample for sample in task if sample["patient_id"] == id][-1]['label']
        label[day] = 1
        P.label = torch.tensor(label)
    else:
        label = [sample for sample in task if sample["patient_id"] == id][-1]['label']
        P.label = label

    # P.visit_padded_node = patient['visit_padded_node']
    # P.ehr_nodes = patient['ehr_node_set']
    P.patient_id = id
    
    return P


############################################################################################






# Hypernetwork: Generates GAT weights based on task embedding
class AttentionHypernetwork(nn.Module):
    def __init__(self, task_embed_dim, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(task_embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.out_dim)
        )

    def forward(self, task_embedding):  # shape: (batch_size, task_embed_dim)
        a = self.mlp(task_embedding).view(-1, self.out_dim, 1)  # (batch_size, total_param)
        return a

class TaskAdaptiveGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.my_hidden = nn.Linear(in_dim, int(out_dim/2))

    def forward(self, x, edge_index, a):
        # W: (1, out_dim, in_dim), a: (1, 2*out_dim, 1)
        a = a[0]
        h = self.my_hidden(x)  # (N, out_dim)
        row, col = edge_index

        h_row = h[row]
        h_col = h[col]
        attn_input = torch.cat([h_row, h_col], dim=-1)  # (E, 2*out_dim)
        e = self.leaky_relu(torch.matmul(attn_input, a).squeeze())  # (E,)

        alpha = softmax(e, row)
        alpha = F.dropout(alpha, p=0.6, training=self.training)
        out = torch.zeros_like(h)
        out.index_add_(0, row, h_col * alpha.unsqueeze(-1)) 
        return out


class RelationalGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_relations, dropout=0.1):
        super(RelationalGATLayer, self).__init__()
        self.rel_gats = nn.ModuleList([
            TaskAdaptiveGATLayer(in_dim, out_dim)
            for _ in range(num_relations)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type, a):

        rel_outputs = []

        for rel in range(len(self.rel_gats)):
            mask = (edge_type == rel)
            edge_index_rel = edge_index[:, mask]
            if edge_index_rel.size(1) == 0:
                rel_outputs.append(torch.zeros(x.size(0), self.rel_gats[rel].out_channels * self.rel_gats[rel].heads, device=x.device))
                continue
            out = self.rel_gats[rel](x, edge_index_rel, a)
            rel_outputs.append(out)

        out = torch.stack(rel_outputs, dim=0).sum(dim=0)
        return self.dropout(out)


class MetaGAT(nn.Module):
    def __init__(self, num_tasks, num_relations, task_embed_dim, in_dim, out_dim):
        super().__init__()
        self.task_embeddings = nn.Embedding(num_tasks, task_embed_dim)
        self.hypernet = AttentionHypernetwork(task_embed_dim, in_dim, out_dim)
        self.gats = RelationalGATLayer(in_dim, out_dim, num_relations, dropout=0.1)
        # self.classifier = nn.Linear(out_dim, 1)  # For binary prediction tasks

    def forward(self, x, edge_index, edge_type, task_id):
        task_embed = self.task_embeddings(task_id)  # shape: (1, dim)
        a = self.hypernet(task_embed)
        h = self.gats(x, edge_index, edge_type, a)
        # return self.classifier(h)
        return h

###############################################################





class Dataset(torch.utils.data.Dataset):
    def __init__(self, G, node_map, icd92cui, atc2cui, dataset, task):
        self.G = G
        self.dataset=dataset
        self.task = task
        self.node_map, self.icd92cui, self.atc2cui = node_map, icd92cui, atc2cui
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return get_subgraph(G=self.G, node_map=self.node_map, 
                            icd92cui=self.icd92cui, atc2cui=self.atc2cui, 
                            dataset=self.dataset, task=self.task, idx=idx)

def get_dataloader(G_tg, train_dataset, val_dataset, test_dataset, task, batch_size):
    train_set = Dataset(G=G_tg, dataset=train_dataset, task=task)
    val_set = Dataset(G=G_tg, dataset=val_dataset, task=task)
    test_set = Dataset(G=G_tg, dataset=test_dataset, task=task)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader



if __name__ == "__main__":
    device = torch.device("cpu")

    model = MetaGAT(num_tasks=5, num_relations= 3, task_embed_dim=32, in_dim=16, out_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # synthetic data for debugging
    num_nodes = 5
    num_node_features = 16
    num_relations = 3
    num_tasks = 4

    x = torch.randn((num_nodes, num_node_features), device=device)
    print("Input node embeddings shape:", x.shape)

    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 0],
        [1, 2, 3, 4, 0, 2]
    ], dtype=torch.long, device=device)

    edge_type = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long, device=device)

    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    task_id = torch.tensor([2], device=device) 
    out = model(data.x, data.edge_index, data.edge_type, task_id=task_id)

    print("Output node embeddings shape:", out.shape)







    # feature_dim = 16

    # with open('resources/umls_mimicentity_2-hop_hierarchy_graph.pkl', 'rb') as f:
    #     G = pickle.load(f)

    # for node in G.nodes:
    #     G.nodes[node]['x'] = torch.rand(feature_dim)
    
    # G = from_networkx(G)

    # num_nodes = G.num_nodes
    # idx = torch.randperm(num_nodes)

    # train_idx = idx[:int(0.6 * num_nodes)]
    # val_idx = idx[int(0.6 * num_nodes):int(0.8 * num_nodes)]
    # test_idx = idx[int(0.8 * num_nodes):]

    # G.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    # G.train_mask[train_idx] = True


    # loader = NeighborLoader(
    #     G,                   # Your big Data object
    #     num_neighbors=[15, 10], # 2-hop neighbors: 15 in 1st hop, 10 in 2nd
    #     batch_size=32,
    #     input_nodes=G.train_mask  # or all nodes
    # )
    # for batch in loader:
    # # batch.x, batch.edge_index, batch.batch, etc.
    #     out = model(batch.x, batch.edge_index, batch.batch)

    # for epoch in range(epochs):
    #     for task_id, subgraph_data in task_loader:  # subgraph per task
    #         subgraph_data = subgraph_data.to(device)
    #         out = model(subgraph_data.x, subgraph_data.edge_index, task_id)
    #         # Use only relevant nodes for loss (e.g., target patients)
    #         loss = F.binary_cross_entropy_with_logits(out[subgraph_data.train_mask], subgraph_data.y[subgraph_data.train_mask])
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
