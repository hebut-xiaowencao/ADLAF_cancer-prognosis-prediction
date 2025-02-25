import os
import sys
import gc
import random
import time
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import torch.optim as optim
from tqdm import trange, tqdm
from Models.VAE import VAE
from torch.utils.tensorboard import SummaryWriter
from utils.EarlyStopping import EarlyStopping
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from utils import WCVAEplot

# Training settings
exc_path = sys.path[0] 

def loss_fn(recon_x, x, z, means_encoder, log_var_encoder, means_decoder, log_var_decoder):
    BCE = torch.nn.functional.mse_loss(recon_x, x, reduction='sum') 
    KLD = gaussian_loss(z, means_decoder, log_var_decoder, means_encoder, log_var_encoder)

    return (BCE + KLD) / x.size(0), BCE, KLD  

def log_normal(x, mu, var):
    var = torch.exp(var)
    eps = 1e-8
    if eps > 0.0:
        var = var + eps 
    return 0.5 * torch.mean(
        torch.log(torch.FloatTensor([2.0 * np.pi]).cuda()).sum(0) + torch.log(var) + torch.pow(x - mu, 2) / var, dim=-1)

def gaussian_loss(z, z_mu, z_var, z_mu_prior, z_var_prior):
    loss = log_normal(z, z_mu, z_var) - log_normal(z, z_mu_prior, z_var_prior)
    return loss.mean() 

# top-k
def generated_generator(args, device, adj_scipy, features, cancer_name):
    features = features.to(torch.device('cpu'))  
    x_list, c_list = [], []  
    #kk = [5 10 15 20 25 30]
    for i in trange(adj_scipy.shape[0]):  
        neighbors_index = list(adj_scipy[i].nonzero()[1])  

        if len(neighbors_index) == 0:
            continue

        x = features[neighbors_index]  
        c = features[i].unsqueeze(0)  
        
        sim = np.abs(cosine_similarity(c, x).flatten())  

        weights = abs(sim) /(np.abs(sim).sum() + 1e-8)
        
       
        if len(neighbors_index) < 20:
           
            weights[:] = weights[:]  # reset weight
           
        else:

            top_indices = np.argsort(weights)[-20:]  
            weights[:] = 0  
            top_weights = sim[top_indices] 
            print('top',top_weights)

            normalized_weights = top_weights / np.sum(top_weights)  

            weights[top_indices] = normalized_weights   

        weighted_x = x * torch.tensor(weights).unsqueeze(1).to(torch.float32)
        
        x_list.append(weighted_x)
        c_list.append(c.repeat(weighted_x.shape[0], 1))
    
    if not x_list or not c_list:
        raise ValueError("no avaliable feature")

    features_x = np.vstack([x.cpu().numpy() for x in x_list])
    features_c = np.vstack([c.cpu().numpy() for c in c_list])

    del x_list, c_list
    gc.collect() 

    hidden_features = 128
    wcvae = VAE(
        encoder_layer_sizes=[features.shape[1], hidden_features],
        latent_size=args.latent_size,
        decoder_layer_sizes=[hidden_features, features.shape[1]],
        conditional=args.conditional,
        conditional_size=features.shape[1]
    ).to(device)
    wcvae_optimizer = optim.Adam(wcvae.parameters(), lr=args.pretrain_lr)
    writer = SummaryWriter(log_dir="D:\\ADLA\\ADLA\\log")
    early_stopping = EarlyStopping(patience=50, verbose=False)

    for epoch in range(args.total_iterations):
        if len(features_x) == 0 or len(features_c) == 0:
            print("end this epoch, no valid features")
            continue
        
        index = random.sample(range(features_c.shape[0]), args.batch_size)
        x, c = features_x[index], features_c[index]
        x = torch.tensor(x, dtype=torch.float32).to(device)
        c = torch.tensor(c, dtype=torch.float32).to(device)

        wcvae.train()
        if args.conditional:
            recon_x, z, means_encoder, log_var_encoder, means_decoder, log_var_decoder = wcvae(x, c)
        else:
            recon_x, z, means_encoder, log_var_encoder, means_decoder, log_var_decoder = wcvae(x)

        wcvae_loss, BCE, KLD = loss_fn(recon_x, x, z, means_encoder, log_var_encoder, means_decoder, log_var_decoder)
        print(f"Epoch: {epoch}, BCE: {BCE}, KLD: {KLD}, Loss: {wcvae_loss}")

        writer.add_scalar('wcvae_loss', wcvae_loss, epoch)
        writer.add_scalar('BCE', BCE, epoch)
        writer.add_scalar('KLD', KLD, epoch)

        wcvae_optimizer.zero_grad()
        wcvae_loss.backward()
        wcvae_optimizer.step()

      
        model_path = f"D:\\ADLA\\ADLA\\Models\\Pretrain1\\{cancer_name}"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        early_stopping(wcvae_loss.item(), wcvae, path=f"{model_path}\\{cancer_name}_pretrain.pth")
    

    writer.close()
    del features_x, features_c
    gc.collect()   
    return wcvae

    

