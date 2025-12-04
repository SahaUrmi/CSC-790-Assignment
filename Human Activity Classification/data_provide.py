
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
# from models.mamba import Mamba
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

# 128 timesteps = 2.56s for sampling at 50Hz
def create_windows_har70(data, labels, window_size, step_size):
    X, y = [], []
    for i in range(0, len(data) - window_size + 1, step_size):
      if len(np.unique(labels[i:i+window_size])) == 1:
        X.append(data[i:i+window_size])
        y.append(labels[i + window_size // 2])  # central label
    return np.stack(X), np.array(y)

def get_dataset(arg):
    if arg.dataset_name == 'har70+':
        folder_path = "/home/sahau24/csc790project/HAR/datasets/har70plus"
        files = os.listdir(folder_path)
        for file_name in files:
            df=pd.read_csv(os.path.join(folder_path, file_name))
            x = df.iloc[:, 1:-1]
            y = df.iloc[:, -1] 
            a, b = create_windows_har70(x, y, arg.window_size, arg.step_size) 
            if file_name == files[0]:
                X = a
                Y = b
            else:
                X = np.concatenate((X, a), axis=0)
                Y = np.concatenate((Y, b), axis=0)
        
        Y = np.where(Y != 1, Y - 2, Y-1) 
        
        # Normalize Features
        scaler = StandardScaler()
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = scaler.fit_transform(X_reshaped).reshape(X.shape)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y)

        # Wrap into Pytorch DataLoader
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

        train_loader = DataLoader(train_dataset, batch_size=arg.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=arg.batch_size)

        return train_loader, test_loader