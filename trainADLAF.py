import os.path
from Models.ADLAFtrain import train
from utils.DataProcessing import CancerDataset
import torch_geometric.transforms as T
import warnings

# ignore FutureWarning & UserWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


if __name__ == "__main__":
    k = 45
    base_path = "D:\\ADLAF\\ADLAF"

    Environment = f"{base_path}\\k_{k}\\data"
    cancers = ['BLCA', 'BRCA', 'CESC', 'COAD', 'HNSC', 'LGG', 'LUAD', 'MESO','SARC', 'SKCM']# cancer type
    for cancer in cancers:  
        cancer_name =  cancer 
        data = CancerDataset(root=os.path.join(Environment, cancer_name), transform=T.ToSparseTensor())
        
        lr = 0.001  
        value = 0.7  #  thresholds for assessing indicators

        train(cancer_name=cancer_name, data=data, lr=lr, value=value)
