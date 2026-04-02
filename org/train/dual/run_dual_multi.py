import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torch.utils.data import Dataset, TensorDataset, DataLoader

REPO_ROOT = "/home/leon/models/NeuroFlame"

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pal = sns.color_palette("tab10")
DEVICE = 'cuda:0'

import sys
sys.path.insert(0, '../../../')

from notebooks.setup import *

import pandas as pd
import torch.nn as nn
from time import perf_counter
from scipy.stats import circmean

from src.network import Network
from src.plot_utils import plot_con
from src.decode import decode_bump, circcvl
from src.lr_utils import masked_normalize, clamp_tensor, normalize_tensor

sys.path.insert(0, '../../../src')

from src.train.dual.train_dpa import train_dpa
from src.train.dual.train_gng import train_gng
from src.train.dual.train_dual import train_dual

REPO_ROOT = "/home/leon/models/NeuroFlame"
conf_name = "train_dual.yml"
DEVICE = 'cuda:0'

for ite in range(100):
    seed = ite
    print('model', seed)
    # train_dpa(REPO_ROOT, conf_name, seed, DEVICE)
    # train_gng(REPO_ROOT, conf_name, seed, DEVICE)
    train_dual(REPO_ROOT, conf_name, seed, DEVICE)
