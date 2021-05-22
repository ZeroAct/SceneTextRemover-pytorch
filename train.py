import sys
import os, argparse, time, tqdm, random, cv2
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn

from dataset import CustomDataset, postprocess_image, tensor_to_mat
from network import STRNet
from losses import TSDLoss, TRGLoss

random_seed = 123

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", default='dataset', help="data root path")
    
    parser.add_argument("-e", "--num_epochs", default=100, type=int, help="num epochs")
    parser.add_argument("-b", "--batch_size", default=16, type=int, help="batch size > 1")
    
    parser.add_argument("-n", "--num_workers", default=8, type=int, help="num_workers for DataLoader")
    parser.add_argument("-sn", "--show_num", default=4, type=int, help="show result images during training num")
    
    args = parser.parse_args()
    
    return args

def load_weights_from_directory(model, weight_path) -> int:
    if weight_path.endswith('.pth'):
        wp = weight_path
    else:
        wps = sorted(os.listdir(weight_path), key=lambda x: int(x.split('_')[0]))
        if wps:
            wp = wps[-1]
        else:
            return 0
    
    print(f"Loading weights from {wp}...")
    model.load_state_dict(torch.load(os.path.join(weight_path, wp)))
    return int(wp.split('_')[0])

if __name__ == "__main__":
    
    args = get_args()
    
    ### Path
    model_path     = "results"
    weight_path    = os.path.join(model_path, "weights")
    show_path      = os.path.join(model_path, "show")
    
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(weight_path, exist_ok=True)
    os.makedirs(show_path, exist_ok=True)
    
    
    ### Hyperparameters
    epochs         = args.num_epochs
    batch_size     = args.batch_size
    if batch_size <= 1:
        raise "Batch size should bigger than 1 for batch normalization"
    
    num_workers    = args.num_workers
    show_num       = args.show_num
    
    ### DataLoader
    dataloader_params = {'batch_size': batch_size,
                         'shuffle': True,
                         'drop_last': True,
                         'num_workers': num_workers}
    
    train_data = CustomDataset(args.data_path, set_name="train")
    train_gen = DataLoader(train_data, **dataloader_params)
    
    dataloader_params = {'batch_size': 1,
                         'shuffle': True,
                         'drop_last': False,
                         'num_workers': num_workers}
    val_data = CustomDataset(args.data_path, set_name="val")
    val_gen = DataLoader(val_data, **dataloader_params)
    
    steps_per_epoch = len(train_gen)
    
    ### Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}...")
    
    model = STRNet().to(device)
    
    # load best weight
    initial_epoch = load_weights_from_directory(model, weight_path) + 1
    print(f"Training start from epoch {initial_epoch}")
    
    # Train Setting
    model_optim = Adam(model.parameters(), 0.0001)
    discrim_optim = Adam(model.discrim.parameters(), 0.0004)
    
    ### Train
    for epoch in range(initial_epoch, epochs):
        ## training
        train_loss = []
        train_discrim_loss = []
        
        model.train()
        pgbar = tqdm.tqdm(train_gen, total=len(train_gen))
        pgbar.set_description(f"Epoch {epoch}/{epochs}")
        for I, Itegt, Mm, Msgt in pgbar:
            
            I, Itegt, Mm, Msgt = I.to(device), Itegt.to(device), Mm.to(device), Msgt.to(device)
            
            # train model
            Ms, Ite, Ms_, Ite_ = model.forward(I, Mm)
            
            Ltsd = TSDLoss(Msgt, Ms, Ms_)
            Ltrg = TRGLoss(Mm, Ms, Ms_, Itegt, Ite, Ite_)
            Lgsn = -torch.mean(model.discrim(Mm, Ite_))
            
            total_loss = Ltsd + Ltrg + Lgsn
            
            model_optim.zero_grad()
            total_loss.backward()
            model_optim.step()
            
            # train discriminator
            Ms, Ite, Ms_, Ite_ = model.forward(I, Mm)
            Ldsn = torch.mean(F.relu(1-model.discrim(Mm, Itegt))) + \
                torch.mean(F.relu(1+model.discrim(Mm, Ite_)))
                          
            discrim_optim.zero_grad()
            Ldsn.backward()
            discrim_optim.step()
            
            
            ltsd = Ltsd.detach().cpu().item()
            ltrg = Ltrg.detach().cpu().item()
            lgsn = Lgsn.detach().cpu().item()
            train_loss.append(total_loss.detach().cpu().item())
            train_discrim_loss.append(Ldsn.detach().cpu().item())
            
            pgbar.set_postfix_str(f"total loss : {train_loss[-1]:.6f} ltsd : {ltsd:.6f} ltrg : {ltrg:.6f} lgsn : {lgsn:.6f} d_loss : {train_discrim_loss[-1]:.6f}")
        
        train_loss = sum(train_loss)/len(train_loss)
        
        ## validation
        val_loss = []
        
        # will saved in show directory
        result_images = []
        
        model.eval()
        pgbar = tqdm.tqdm(val_gen, total=len(val_gen))
        pgbar.set_description("Validating...")
        for I, Itegt, Mm, Msgt in pgbar:
            
            I, Itegt, Mm, Msgt = I.to(device), Itegt.to(device), Mm.to(device), Msgt.to(device)
            
            # train model
            Ms, Ite, Ms_, Ite_ = model.forward(I, Mm)
            
            Ltsd = TSDLoss(Msgt, Ms, Ms_)
            Ltrg = TRGLoss(Mm, Ms, Ms_, Itegt, Ite, Ite_)
            Lgsn = -torch.mean(model.discrim(Mm, Ite_))
            
            total_loss = Ltsd + Ltrg + Lgsn
            
            val_loss.append(total_loss.detach().cpu().item())
            
            pgbar.set_postfix_str(f"loss : {sum(val_loss[-10:]) / len(val_loss[-10:]):.6f}")
            
            if len(result_images) < args.show_num:
                result_images.append([I.cpu(), Itegt.cpu(), Ite_.cpu(), Msgt.cpu(), Ms_.cpu()])
            else:
                break
        
        val_loss = sum(val_loss) / len(val_loss)
        
        ## visualize
        fig, axs = plt.subplots(args.show_num, 1, figsize=(5, 2*args.show_num))
        fig.suptitle("Image, Gt, Gen, Stroke Gt, Stroke")
        for i, (I, Itegt, Ite_, Msgt, Ms_) in enumerate(result_images):
            I = postprocess_image(tensor_to_mat(I))[0]
            Itegt = postprocess_image(tensor_to_mat(Itegt))[0]
            Ite_ = postprocess_image(tensor_to_mat(Ite_))[0]
            Msgt = postprocess_image(tensor_to_mat(Msgt))[0]
            Ms_ = postprocess_image(tensor_to_mat(Ms_))[0]
            
            Msgt = cv2.cvtColor(Msgt, cv2.COLOR_GRAY2BGR)
            Ms_ = cv2.cvtColor(Ms_, cv2.COLOR_GRAY2BGR)
            
            axs[i].imshow(np.hstack([I, Itegt, Ite_, Msgt, Ms_]))
            axs[i].set_xticks([])
            axs[i].set_yticks([])
        
        fig.savefig(os.path.join(model_path, "show", f"epoch_{epoch}.png"))
        plt.close()
        
        print(f"train_loss : {train_loss}, val_loss : {val_loss}")
        print()
        time.sleep(0.2)
        
        torch.save(model.state_dict(), os.path.join(weight_path, f"{epoch}_train_{train_loss}_val_{val_loss}.pth"))
        