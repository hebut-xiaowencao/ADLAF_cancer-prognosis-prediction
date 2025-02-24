# encoding: utf-8
import os.path
from sklearn.model_selection import KFold
import torch
import torch_geometric.transforms as T
from Models.appGraph import GraphSurv
from Models.CoxNN import DeepCox_LossFunc
from utils.Indicators import concordance_index
from utils.support import split_censor, sort_data
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from utils.EarlyStopping import EarlyStopping
from sklearn.utils import resample
import random

torch.set_printoptions(profile='full')
Environment = "D:\\ADLA\\ADLA\\cancer"
Training = "D:\\ADLA\\ADLA\\SavedModels\\Train1"
PRE = "D:\\ADLA\\ADLA\SavedModels\\Pretrain1"
Indicators = "D:\\ADLA\\ADLA\\SavedIndicators"
writer = SummaryWriter(log_dir="D:\\ADLA\\ADLA\\log")

def save_evaluation_indicators(indicators, model_name):
    if not os.path.exists(Indicators):
        os.makedirs(Indicators)

    path = "{}\\ADLA\\{}_Indicators.xlsx".format(Indicators, model_name)
    file = open(path, "a")

    str_indicators = ''
    for index, indicator in enumerate(indicators):
        if index == 0:
            str_indicators += str(indicator) + " "
        else:
            str_indicators += str(np.round(indicator, 4)) + " "
    str_indicators += "\n"
    file.write(str_indicators)
    file.close()

def get_augmented_features(data, wcvae_model, concat, device):
    augmented_features_list = []
    for _ in range(concat):  
        z = torch.randn([data.x.size(0), wcvae_model.latent_size]).to(device)  # latten variable z
        augmented_features = wcvae_model.inference(z, data.x).detach()
        augmented_features_list.append(augmented_features)
    return torch.cat(augmented_features_list, dim=1)  

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

#TRAIN FUNCTION
def train(cancer_name, data, lr, value):
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(data)
    path_info = '{}\\{}\\processed\\{}_info.pkl'.format(Environment, cancer_name, cancer_name)
    clinic_info = torch.load(path_info, map_location='cpu')
    event, duration = clinic_info['event'], clinic_info['duration']
    
    print(len(event))
    # divide sample
    event_indices = np.where(np.array(event) == 1)[0]
    non_event_indices = np.where(np.array(event) == 0)[0]

    train_validate_data, train_validate_event, train_validate_duration = [], [], []
    test_data, test_event, test_time = [], [], []
    sampled_index = set()

    #set seed
    set_seed(42)

    event_sample_size = round(len(event_indices)*0.8)
    non_event_sample_size = round(len(non_event_indices)*0.8)

    sampled_index = set()
    #Segmenting the dataset
    for _ in range(10):
        sampled_event = resample(event_indices, n_samples=event_sample_size,  replace=False,random_state=42)
        sampled_non_event = resample(non_event_indices, n_samples=non_event_sample_size, replace= False ,random_state=42)

        sampled = np.concatenate([sampled_event, sampled_non_event])
        sampled_index |= set(sampled)  

    for item in sampled:
        train_validate_data.append(data[item])
        train_validate_event.append(event[item])
        train_validate_duration.append(duration[item])
    print('sample num',len(sampled_index))
    print('trainvali',len(train_validate_data))
    test_index = set(np.arange(len(data))) - sampled_index
    print('test len',len(test_index))
    for item in test_index:
        test_data.append(data[item])
        test_event.append(event[item])
        test_time.append(duration[item])

    censored_time, censored_data, uncensored_time, uncensored_data = split_censor(
        data=train_validate_data,
        status=np.array(train_validate_event),
        time=np.array(train_validate_duration)
    )

    kf = KFold(n_splits=10, shuffle=True)
    censored_data.extend(uncensored_data)
    censored_time = np.vstack((censored_time, uncensored_time))

    best_num = 0
    best_vali = []
    num = 0
    for train_index, validate_index in kf.split(X=censored_data):
        num += 1

        train_data = [censored_data[i] for i in train_index]
        
        train_time = censored_time[train_index]

        validate_data = [censored_data[i] for i in validate_index]
        validate_time = censored_time[validate_index]
        
        train_event = np.zeros_like(train_time)
        validate_event = np.zeros_like(validate_time)


        num_train, num_validate = len(train_data), len(validate_data)

        _, train_data, train_time = sort_data(train_data, train_time) #安升序排序
        train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False)
        validate_loader = DataLoader(dataset=validate_data, batch_size=1)
        test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
        
        num_nodes = data[0].num_nodes
        model = GraphSurv(input_nodes=num_nodes).to(device=device)

        wcvae_model = torch.load('{}\\{}\\{}_pretrain.pth'.format(PRE, cancer_name, cancer_name)).to(
            device=device)
        loss_func = DeepCox_LossFunc()
        early_stopping = EarlyStopping(patience=30, verbose=True)
       
        model_path = "{}\\{}".format(Training, cancer_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=1.2e-4)
       
        # ReduceLROnPlateau 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )

        vali = []
        for i in range(60):
            #early stopping
            if early_stopping.early_stop:  
                print("Early stopping triggered. Stopping training.")
                break
            #train
            model.train()
            risks = torch.zeros([num_train, 1], dtype=torch.float)
            
            for id, graphs in enumerate(train_loader):
                graphs = graphs.to(device)
                augmented_features = get_augmented_features(data=graphs, wcvae_model=wcvae_model, concat=1,
                                                            device=device)
                graphs.x = torch.cat((graphs.x, augmented_features), dim=1)  
                risk, _ = model(graphs.x, graphs.adj_t)
                risks[id] = risk
            print('risk:',risk[0])
            loss = loss_func(risks, train_time)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_risks = risks.detach()
            train_Cindex = concordance_index(train_time, -train_risks.cpu().numpy())
            
            #evaluate
            model.eval()
            
            with torch.no_grad():
                validate_risks = torch.zeros([num_validate, 1], dtype=torch.float)
                for id, graphs in enumerate(validate_loader):
                    graphs = graphs.to(device)
                    augmented_features = get_augmented_features(data=graphs, wcvae_model=wcvae_model, concat=1,
                                                                device=device)

                    graphs.x = torch.cat((graphs.x, augmented_features), dim=1)  
                    risk, _ = model(graphs.x, graphs.adj_t)
                    validate_risks[id] = risk
                validate_loss = loss_func(validate_risks, validate_time)
                try:
                    validate_Cindex = concordance_index(validate_time, -validate_risks.cpu().numpy())
                except ValueError:
                    print("VaEr")
               
            
                writer.add_scalar(tag='Validate Loss', scalar_value=validate_loss.item(), global_step=i)
                writer.add_scalar(tag='Validate C-index', scalar_value=validate_Cindex, global_step=i)
                early_stopping(val_loss=1 - validate_Cindex, model=model,
                                        path="{}\\{}ADLA_best_model.pth".format(model_path, cancer_name + str(num)))

            # update learning rate
            scheduler.step(validate_loss)
            
            print(
                'epoch: {}    train loss: {}  train C-index: {}   validate loss: {}   validate C-index: {}'.format(
                    i, loss.item(), train_Cindex, validate_loss.item(), validate_Cindex))
            vali.append(validate_Cindex)
        fold_best_vali = max(vali)
        best_vali.append(fold_best_vali)
        print(best_vali)
    best_num = best_vali.index(max(best_vali)) + 1
    print('The number of best model in 10-fold validation :',best_num)
    print("====================RESULT=======================")
    model_best = torch.load("{}\\{}ADLA_best_model.pth".format(model_path, cancer_name + str(best_num)))
    model.eval()
    test_Cindex, test_risks = evaluate(model=model_best, wcvae_model=wcvae_model, dataloader=test_loader,
                                            test_time=np.array(test_time), device=device)

    indicators = [cancer_name + str(best_num), test_Cindex]
    save_evaluation_indicators(indicators=indicators, model_name='AUG')
    print(test_Cindex)

#test C-index
def evaluate(model, wcvae_model, dataloader, test_time, device):
    model.eval()
    with torch.no_grad():
        risks = torch.zeros([test_time.shape[0], 1], dtype=torch.float)
        for id, graphs in enumerate(dataloader):
            graphs = graphs.to(device)
            augmented_features = get_augmented_features(data=graphs, wcvae_model=wcvae_model, concat=1,
                                                        device=device)
            graphs.x = torch.cat((graphs.x, augmented_features), dim=1)  # 拼接特征

            risk, _ = model(graphs.x, graphs.adj_t)
            risks[id] = risk
        risks_save = risks.detach()
        cindex = concordance_index(test_time, -risks_save.cpu().numpy())
    return cindex, risks_save
           