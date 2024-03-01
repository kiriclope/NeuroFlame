import time
import gc
import torch
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

def set_seed(seed):
    if seed ==0 :
        seed = int(time.time())
        
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def clear_cache():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            del obj
    
    # Manually triggering the garbage collector afterwards
    gc.collect()
    torch.cuda.empty_cache()
