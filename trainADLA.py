import os.path
from Models.ADLAtrain import train
from utils.DataProcessing import CancerDataset
import torch_geometric.transforms as T
import warnings

# ignore FutureWarning & UserWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


if __name__ == "__main__":
    ROOT_DATA = 'D:\\ADLA\\ADLA\\cancer'
    cancers = ['BRCA']# cancer type
    for cancer in cancers:  
        cancer_name =  cancer 
        data = CancerDataset(root=os.path.join(ROOT_DATA, cancer_name), transform=T.ToSparseTensor())
        
        lr = 0.001  
        value = 0.7  #  thresholds for assessing indicators

        train(cancer_name=cancer_name, data=data, lr=lr, value=value)
