import torch

def get_theta(a, b, IF_NORM=0):

    u, v = a, b

    if IF_NORM:
        u = a / torch.norm(a, p='fro')
        v = b / torch.norm(b, p='fro')

    return torch.atan2(v, u)

def get_idx(ksi, ksi1):    
    theta = get_theta(ksi, ksi1, GM=0, IF_NORM=0)
    return theta.argsort()

def get_overlap(model, rates):
    ksi = model.PHI0.cpu().detach().numpy()
    return rates @ ksi.T / rates.shape[-1]
