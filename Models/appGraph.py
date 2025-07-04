import torch.nn
from Models.appGNN import GNN
from Models.CoxNN import Coxnn
import time
import pandas as pd


class GraphSurv(torch.nn.Module):
    def __init__(self, input_nodes):
        super(GraphSurv, self).__init__()
        torch.manual_seed(1234)
        torch.set_printoptions(profile='full')
        
        self.GNN = GNN(gnn='gcn', n_layer=2, feature_len=2, dim=1)
        self.CoxNN = Coxnn(input_nodes=input_nodes)

    def forward(self, x, edge_index):
        GNN_out = self.GNN(x, edge_index)
        GNN_out = GNN_out.T
        risk = self.CoxNN(GNN_out)

        return risk, GNN_out