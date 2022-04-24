import warnings
warnings.filterwarnings('ignore')

import cv2
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import time
import timm
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
import torch
import random
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from copy import deepcopy
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from torch.optim.lr_scheduler import _LRScheduler   
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import wandb
wandb.login()

class CosineAnnealingWarmupRestarts(_LRScheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
                 ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in
                    self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

def load_image(path):
    img = cv2.imread(path) # np.array cv2 imread => BRG format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BRG 2 RGB format
    return img

class ImageDataset(Dataset):
    def __init__(self, data, classes, species, transform, root='./data3/train_detec_512_v3'):
        super(ImageDataset, self).__init__()
        self.path = data.image.values
        self.target = data.individual_id.values
        self.target_unique = classes
        self.species = data.species.values
        self.species_unique = species
        self.transform = transform
        self.root = root
        
        with open('./group_num.json', 'r') as f :
            self.group_num = json.loads(f.read())
        
    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, idx):
        target = self.target[idx]
        specie = self.species[idx]
        specie_idx = np.where(specie==self.species_unique)[0][0]
        one_hot = torch.zeros(self.target_unique)
        if specie_idx == 0 :
            one_hot[:self.group_num[self.species_unique[specie_idx]]] = np.round(0.1 / self.group_num[self.species_unique[specie_idx]], 6)
        
        else :
            one_hot[self.group_num[self.species_unique[specie_idx-1]]:self.group_num[self.species_unique[specie_idx]]] = np.round(0.1 / self.group_num[self.species_unique[specie_idx]], 6)
        
        one_hot[target] = 1 - sum(one_hot)

        img_path = os.path.join(self.root, self.path[idx])
        img = load_image(img_path)
        img = self.transform(image=img)["image"]

        return img, torch.as_tensor(one_hot).float()

def get_train_transforms(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, 
                           scale_limit=0.15, 
                           rotate_limit=30, 
                           p=0.5),
        A.HueSaturationValue(
                hue_shift_limit=0.2, 
                sat_shift_limit=0.2, 
                val_shift_limit=0.2, 
                p=0.5),
        A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1), 
                contrast_limit=(-0.1, 0.1), 
                p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet for Dataset2
        ToTensorV2()
    ])

def get_valid_transforms(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet for Dataset2
        ToTensorV2()
    ])

def stratified_kfold(df, fold, n_split, seed=2022, input_col='image', target_col='species'):
    skf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=seed)
    for idx, (train_index, valid_index) in enumerate(skf.split(df[input_col], df[target_col])):
        if idx == fold:
            return train_index, valid_index

def save_history(path: str, name: str, state_dict: dict, file_name: str = None) -> None:
    if not os.path.exists(f"{path}/{name}"):
        os.makedirs(f"{path}/{name}")
    torch.save(state_dict, f"{path}/{name}/{file_name}.pt")
    return

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        cosine, sine, phi = cosine.float(), sine.float(), phi.float()

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.argmax(dim=-1).view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

class HappyWhaleModel(nn.Module):
    def __init__(self, model_name, emb_size=512, n_classes=15587, s=30.0, m=0.30, easy_margin=False, ls_eps=0.0, pretrain=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrain)
        
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features=in_features, out_features=emb_size)

        self.arc = ArcMarginProduct(emb_size, n_classes, s=s, m=m, easy_margin=easy_margin, ls_eps=ls_eps)
        
    def forward(self, image, label):
        emb = self.model(image)
        output = self.arc(emb, label)
        return output, emb

def train_epoch(model, optimizer, loss_fn, loader, scheduler, scaler=None, iters_to_accumulate=1):
    model.train()
    
    losses, y_true, y_pred, embed = [], [], [], []
    for i, (x, y) in enumerate(tqdm(loader)):
        x, y = x.to(device), y.to(device)
        
        with torch.cuda.amp.autocast():
            output, emb = model(x, y)
            loss = loss_fn(output, y)
        scaler.scale(loss).backward()

        if ((i + 1) % iters_to_accumulate == 0) or ((i + 1) == len(loader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        scheduler.step()

        y_true.extend(y.detach().cpu().argmax(dim=-1))
        y_pred.extend(output.detach().cpu().argmax(dim=-1))
        losses.append(loss.detach().cpu().item())
        embed.extend(emb.detach().cpu())
    embed_stack = np.stack(embed, axis=0)
    y_true_stack = np.stack(y_true, axis=0)
    y_pred_stack = np.stack(y_pred, axis=0)
    neigh = NearestNeighbors(n_neighbors=1, metric='cosine').fit(embed_stack) 

    acc = accuracy_score(y_true_stack, y_pred_stack)
    return np.mean(losses), acc, neigh, y_true_stack

def validate(model, loss_fn, loader, neigh, train_true):
    model.eval()

    losses, y_true, y_pred, embed = [], [], [], []
    with torch.no_grad():
        for x, y in tqdm(loader):
            x, y = x.to(device), y.to(device)

            output, emb = model(x, y)

            loss = loss_fn(output, y)

            y_true.extend(y.detach().cpu().argmax(-1))
            y_pred.extend(output.detach().cpu().argmax(-1))      
            losses.append(loss.detach().cpu().item())
            embed.extend(emb.detach().cpu())
    embed_stack = np.stack(embed, axis=0)
    y_true_stack = np.stack(y_true, axis=0)
    y_pred_stack = np.stack(y_pred, axis=0)

    acc = accuracy_score(y_true_stack, y_pred_stack)
    distances, indices = neigh.kneighbors(embed_stack, return_distance=True)
    neigh_pred = train_true[indices.reshape(-1)]

    match = neigh_pred == y_true_stack
    neigh_acc = match.mean()
    return np.mean(losses), acc, neigh_acc

def seed_everything(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu vars    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars if use multi-GPU
    if torch.backends.cudnn.is_available:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return print('# SEEDING DONE')

def main(config):
    seed_everything(config.seed)
    
    df = pd.read_csv(config.train_data)
    df.species.replace({"globis": "short_finned_pilot_whale",
                      "pilot_whale": "short_finned_pilot_whale",
                      "kiler_whale": "killer_whale",
                      "bottlenose_dolpin": "bottlenose_dolphin"}, inplace=True)
    print('train len:', len(df))
    print('b4 species', len(df.species.unique()))
    print('b4 classes', len(df.individual_id.unique()))

    with open('./id_dic.json', 'r') as f :
        class_map = json.loads(f.read())
    df.individual_id = df.individual_id.map(class_map)
    species = np.sort(df.species.unique())
    classes = len(df.individual_id.unique())

    df1 = df[df['individual_id'].map(df['individual_id'].value_counts()) == 1]
    df2 = df[df['individual_id'].map(df['individual_id'].value_counts()) > 1]

    train_one, valid_one = stratified_kfold(df=df1, fold=config.fold, n_split=config.n_split, seed=config.seed, target_col='species')
    train_two, valid_two = stratified_kfold(df=df2, fold=config.fold, n_split=config.n_split, seed=config.seed, target_col='individual_id')

    train_one = np.take(df1.index.to_numpy(), train_one)
    train_two = np.take(df2.index.to_numpy(), train_two)
    valid_two = np.take(df2.index.to_numpy(), valid_two)
    
    train_index = np.sort(np.concatenate([train_one, train_two], axis=0))
    valid_index = valid_two # valid_one has no validation ids
    
    train = df.iloc[train_index].reset_index(drop=True)
    valid = df.iloc[valid_index].reset_index(drop=True)

    train_transform = get_train_transforms(config.image_size)
    valid_transform = get_valid_transforms(config.image_size)

    train_set = ImageDataset(data=train, classes=classes, species=species, transform=train_transform, root=config.train_image_root)
    valid_set = ImageDataset(data=valid, classes=classes, species=species, transform=valid_transform, root=config.train_image_root)

    # adjust num_workers = 4 * num_GPU and batch_size for full GPU Util
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = HappyWhaleModel(model_name=config.model_name, 
                        emb_size=config.emb_size,
                        s=config.s,
                        m=config.m,
                        ).to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    grad_scaler = torch.cuda.amp.GradScaler()

    cosine_annealing_scheduler_arg = dict(
        first_cycle_steps=len(train_set)//config.batch_size*config.epoch,
        cycle_mult=1.0,
        max_lr=config.learning_rate,
        min_lr=1e-06,
        warmup_steps=len(train_set)//config.batch_size*3, # wanrm up 0~3 epoch
        gamma=0.9
    )
    scheduler = CosineAnnealingWarmupRestarts(optimizer, **cosine_annealing_scheduler_arg)

    start = time.time()
    print('Start Training')
    wandb.init(config=config.__dict__,
           project='happywhale_main',
           name=f'{config.model_name}_fold_{config.fold}'
          )
    for i in range(config.epoch):
        print(f"epoch: {i}")
        lr = scheduler.get_lr()[0]
        train_loss, train_acc, neigh, train_true = train_epoch(model, optimizer, loss_fn, train_loader, scheduler, grad_scaler, config.iters_to_accumulate)
        valid_loss, valid_acc, neigh_acc = validate(model, loss_fn, valid_loader, neigh, train_true)
        

        print(f"train loss {train_loss :.4f} acc {train_acc :.4f}")
        print(f"valid loss {valid_loss :.4f} acc {valid_acc :.4f}")
        print(f"neigh acc {neigh_acc :.4f}")
        print(f"lr {lr} time {time.time() - start :.2f}s")

        check_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        save_history(path=config.checkpoint, name=config.model_name, state_dict=check_dict, 
                file_name=f"{config.fold}fold_e{i}_{config.exp_name}_{valid_loss :.4f}_neigh_{neigh_acc :.4f}")
    
        wandb_dict = {
            'train loss': train_loss,
            'train acc': train_acc,
            'valid loss': valid_loss,
            'valid acc': valid_acc,
            'neigh acc': neigh_acc,
            'learning rate': scheduler.get_lr()[0],
        }
        wandb.log(wandb_dict)
    wandb.finish()
    return


if __name__ == '__main__':
    class CFG:
        def __init__(self):
            self.project = None
            self.checkpoint = './saved/checkpoint_species_2'
            self.n_split = 5
            self.fold = 3
            self.seed = 2022
            self.epoch = 21
            self.weight_decay = 5e-4
            self.s = 30.0
            self.label_smoothing = 0.
            self.exp_name = 'default'
            self.model_name = 'tf_efficientnetv2_m'
            self.train_data = './data3/train.csv'
            self.train_image_root = './data3/train_detec_512_v3'
            self.image_size = 380
            self.batch_size = 16
            self.iters_to_accumulate = 4
            self.learning_rate = 6.25e-6 * self.batch_size * self.iters_to_accumulate
            self.emb_size = 2048
            self.m = 0.55
    config = CFG()
    main(config)