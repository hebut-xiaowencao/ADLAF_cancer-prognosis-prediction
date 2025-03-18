import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool  # 引入自注意力卷积层和平均池化

class GNN(torch.nn.Module):
    def __init__(self, gnn, n_layer, feature_len, dim, heads=1):
        super(GNN, self).__init__()
        self.gnn = gnn
        self.n_layer = n_layer
        self.feature_len = feature_len
        self.dim = dim
        self.gnn_layers = torch.nn.ModuleList([])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for i in range(self.n_layer):
            self.gnn_layers.append(GATConv(in_channels=2 * feature_len if i == 0 else dim * heads,
                                            out_channels=dim,
                                            heads=heads))
            
    def forward(self, x, edge_index):
        edge_index = edge_index.to(self.device)
        for index, layer in enumerate(self.gnn_layers):
            x = layer(x, edge_index)
            if index != self.n_layer - 1:
                x = torch.relu(x)
                
        graph_embedding = x
        return graph_embedding



