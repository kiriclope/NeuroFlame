import time
import gc
import torch

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

def print_activity(model, step, rates):

    times = round((step - model.N_STEADY) / model.N_STEPS * model.DURATION, 2)

    activity = []
    for i in range(model.N_POP):
        activity.append(round(torch.mean(rates[:, model.slices[i]]).item(), 2))

    print("times (s)", times, "rates (Hz)", activity)

def set_seed(seed):
    # seed0 = seed
    if seed == -1:
        seed = int(time.time())

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # print('seed', seed0, 'test', torch.rand(1), torch.cuda.FloatTensor(1).uniform_())

def clear_cache():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            del obj

    # Manually triggering the garbage collector afterwards
    gc.collect()
    torch.cuda.empty_cache()
