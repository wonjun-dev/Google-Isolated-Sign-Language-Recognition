import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super(GCNLayer, self).__init__()
        self.projection = nn.Linear(c_in, c_out)

    def forward(self, node_feats, adj_matrix):
        """
        Inputs:
            node_feats - Tensor with node features of shape [batch_size, num_nodes, c_in]
            adj_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[b,i,j]=1 else 0.
                         Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections.
                         Shape: [batch_size, num_nodes, num_nodes]
        """
        # Num neighbours = number of incoming edges
        num_neighbours = adj_matrix.sum(dim=-1, keepdims=True)
        node_feats = self.projection(node_feats)
        N, T, V, C = node_feats.shape
        node_feats = torch.bmm(adj_matrix.view(-1, V, V), node_feats.view(-1, V, C))
        node_feats = node_feats / num_neighbours.view(-1, V, 1)
        node_feats = node_feats.view(N, T, V, C)
        return node_feats


if __name__ == "__main__":
    import os
    from dataset import load_relevant_data_subset
    from preproc import preproc_v1_1
    from deprecated.graph import Graph

    max_len = 384
    xyz = load_relevant_data_subset(
        "/sources/dataset/train_landmark_files/2044/635217.parquet"
    )
    xyz = torch.from_numpy(xyz).float()
    node_feats = preproc_v1_1(xyz, max_len)  # [T, 53, 2]
    T = node_feats.shape[0]
    node_feats = node_feats.repeat(2, 1, 1, 1)

    # node_feats = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
    adj_matrix = Graph().A
    adj_matrix = adj_matrix.repeat(2, T, 1, 1)

    print("Node features:\n", node_feats)
    print("\nAdjacency matrix:\n", adj_matrix)
    layer = GCNLayer(c_in=2, c_out=2)
    layer.projection.weight.data = torch.Tensor([[1.0, 0.0], [0.0, 1.0]])
    layer.projection.bias.data = torch.Tensor([0.0, 0.0])

    with torch.no_grad():
        out_feats = layer(node_feats, adj_matrix)

    print("Adjacency matrix", adj_matrix)
    print("Input features", node_feats)
    print("Output features", out_feats[0, 0])
    print("output features shape", out_feats.shape)
    N, T, V, C = out_feats.shape
    out_feats = out_feats.reshape(N, T, -1)
    print(out_feats[0, 0])
