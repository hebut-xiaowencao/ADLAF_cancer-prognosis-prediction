# encoding: utf-8
import os.path

import pandas as pd
import torch
from Models.GraphSurv import GraphSurv
from Models.CoxNN import DeepCox_LossFunc
from utils.Indicators import concordance_index
from utils.support import split_censor, split_data, sort_data
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from utils.EarlyStopping import EarlyStopping
from utils.DataProcessing import CancerDataset
import torch_geometric.transforms as T

import random

import warnings

# 忽略FutureWarning和UserWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


torch.set_printoptions(profile='full')

k = 5  
base_path = "D:\\ADLAF\\ADLAF"

Environment = f"{base_path}\\k_{k}\\data"
Training = f"{base_path}\\k_{k}\\Train"
Pretrain = f"{base_path}\\k_{k}\\Pretrain"
Indicators = f"{base_path}\\k_{k}\\indicator"
writer = SummaryWriter(log_dir="D:\\ADLAF\\ADLAF\\log")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(40)

def get_augmented_features(data, cvae_model, concat, device):
    augmented_features_list = []
    for _ in range(concat):  
        z = torch.randn([data.x.size(0), cvae_model.latent_size]).to(device)  
        augmented_features = cvae_model.inference(z, data.x).detach()
        augmented_features_list.append(augmented_features)
    return torch.cat(augmented_features_list, dim=1)  

def run(cancer_name, data, lr):
    num_data = len(data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path_info = '{}\\{}\\processed\\{}_info.pkl'.format(Environment, cancer_name, cancer_name)
    clinic_info = torch.load(path_info, map_location='cpu')
    event, duration = clinic_info['event'], clinic_info['duration']

    loader = DataLoader(dataset=data, batch_size=1, shuffle=False)

    num_nodes = data[0].num_nodes
    model_path = "{}\\{}".format(Training, cancer_name)

    #you should input the best model number before the ADLAF !
    model = torch.load("{}\\{}ADLAF_best_model.pth".format(model_path, cancer_name)) 
    cvae_model = torch.load('{}\\{}\\{}_pretrain.pth'.format(Pretrain, cancer_name, cancer_name)).to(
        device=device)

    Cindex, risks = evaluate(model=model, cvae_model=cvae_model, dataloader=loader,
                             test_time=duration, device=device)
    print(Cindex)

    median = risks.median()
    risk_group = []
    for index, item in enumerate(risks.numpy()):
        if item[0] > median:
            risk_group.append('high_risk')
        else:
            risk_group.append('low_risk')
    risk_df = pd.DataFrame(data=risk_group)
    #risk_df.drop()
    #risk_df.to_csv("risk_group_PRO2.csv", index=False, header=False) # save results
    pass


def evaluate(model, cvae_model, dataloader, test_time, device):
    model.eval()
    with torch.no_grad():
        risks = torch.zeros([test_time.shape[0], 1], dtype=torch.float)
        for id, graphs in enumerate(dataloader):
            graphs = graphs.to(device)
            augmented_features = get_augmented_features(data=graphs, cvae_model=cvae_model, concat=1,
                                                        device=device)
            graphs.x = torch.cat((graphs.x, augmented_features), dim=1)  # 拼接特征

            risk, _ = model(graphs.x, graphs.adj_t)
            risks[id] = risk
        risks_save = risks.detach()
        cindex = concordance_index(test_time, -risks_save.cpu().numpy())
    return cindex, risks_save



cancer = 'BRCA'
data = CancerDataset(root=os.path.join(Environment, cancer), transform=T.ToSparseTensor())
run(cancer_name=cancer, data=data, lr=None)
