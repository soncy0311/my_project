import warnings
warnings.filterwarnings('ignore')

import os
import random
import numpy as np
import pandas as pd
import math
import json

import torch
from tqdm import tqdm
import time
from glob import glob

from sklearn.neighbors import NearestNeighbors

def mean_embeddings(emb_list, emb_size=2048,train=True):
    
    if train : total = torch.zeros(51016, emb_size)
    else : total = torch.zeros(27956, emb_size)
        
    for path in emb_list:
        size = torch.load(path).shape[0]
        emb = torch.load(path)
        if size == 51017 :
            emb = np.delete(emb, 23621, axis=0)
        elif size == 51033 :
            emb = np.delete(emb, [11604,15881,16782,21966,23306,24862,25895,29468,31831,35805,37176,40834,47480,48455,36710,47161,23626], axis=0)
        total += emb
    return total / len(emb_list)

def inference(path, embed_path, n_neighbors=1000, threshold=0.5) :
    start = int(time.time())
    
    df_train = pd.read_csv(os.path.join(path, 'train.csv'))
    df_test = pd.read_csv(os.path.join(path, 'test_species_decoding.csv'))
    
    drop_index = [11604,15881,16782,21966,23306,23626,24862,25895,29468,31831,35805,37176,40834,47480,48455,36710,47161]
    df_train = df_train.drop(drop_index, axis=0).reset_index(drop=True)
    
    idx = df_train.individual_id.values
    
    embed = glob(f'{embed_path}/*train*.pt')
    embed_mean = mean_embeddings(embed)
    print(embed_mean.shape)
    
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    knn.fit(embed_mean)
    
    test_embed = glob(f'{embed_path}/*test*.pt')
    test_embed_mean = mean_embeddings(test_embed, train=False)
    print(test_embed_mean.shape)
    
    knn_start = int(time.time())
    print('Nearest Neighbors fit Start')
    distance, indices = knn.kneighbors(test_embed_mean, return_distance=True)
    knn_end = int(time.time())
    print('Finish\n')
    print(f'Nearest Neighbor run time {(knn_end-knn_start)//60}:{(knn_end-knn_start)%60} \n')
    
    id_pred_lst = []
    for i in range(test_embed_mean.shape[0]) :
        id_pred = idx[indices[i]]
        id_pred_lst.append(id_pred)
    id_pred_nn = np.stack(id_pred_lst)
    
    df_test_image = df_test.image.values
    
    print('Create row')
    time.sleep(1)
    
    row = []
    for i in tqdm(range(len(df_test))) :
        for j in range(len(id_pred_nn[i])) :
            temp = []
            temp.append(df_test_image[i])
            temp.append(id_pred_nn[i][j])
            temp.append(1-distance[i][j])
            row.append(temp)
    
    print('\n Create Dictionary')
    time.sleep(1)
    
    df_row = {}
    for i in tqdm(range(len(row))) :
        temp_img = row[i][0]
        temp_trg = row[i][1]
        temp_conf = row[i][2]
        if temp_img in df_row :
            if len(df_row[temp_img]) == 5 :
                continue
            if temp_trg not in df_row[temp_img] :
                df_row[temp_img].append(temp_trg)

        else :
            if temp_conf > threshold :
                df_row[temp_img] = [temp_trg, 'new_individual']
            else :
                df_row[temp_img] = ['new_individual', temp_trg]
    
    sample_list = ['938b7e931166', '5bf17305f073', '7593d2aee842', '7362d7a01d00','956562ff2888']
    
    predictions = {}
    for x in df_row:
        if len(df_row[x])<5:
            remaining = [y for y in sample_list if y not in df_row]
            predictions[x] = df_row[x]+remaining
            predictions[x] = predictions[x][:5]
        else :
            predictions[x] = df_row[x]
        predictions[x] = ' '.join(predictions[x])

    predictions = pd.Series(predictions).reset_index()
    predictions.columns = ['image','predictions']
    
    end = int(time.time())
    print(f'\n Inference run-time {(end-start)//60}:{(end-start)%60}')
    
    return predictions