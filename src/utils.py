import time
import torch
from torch import nn
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

def set_seed(seed):
    if seed ==0 :
        seed = int(time.time())
        
    torch.manual_seed(seed)
        
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For all


def get_theta(a, b, GM=0, IF_NORM=0):

    if IF_NORM:
        u = a / np.linalg.norm(a)
        v = b / np.linalg.norm(b)
          
    if GM:
        u = a
        v = b - np.dot(b, u) / np.dot(u, u) * u
    else:
        u=a
        v=b

    if IF_NORM:
        v = b / np.linalg.norm(b)
    
    return np.arctan2(v, u)

