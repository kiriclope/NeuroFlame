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

def get_ortho_pertu(m, s, dtype, device):

    m = m.real.unsqueeze(-1)
    N = m.shape[0]
    # Covariance matrix for u and v
    C = torch.tensor([[1, s], [s, 1]], dtype=dtype, device=device)

    # Cholesky decomposition
    L = torch.linalg.cholesky(C)

    # Generate a basis orthogonal to m
    # Simple method: Use random vectors and orthogonalize
    a = torch.randn(N, 1, dtype=dtype, device=device)
    b = torch.randn(N, 1, dtype=dtype, device=device)

    # Orthogonalize a and b wrt m
    a -= torch.matmul(a.T, m) * m
    b -= torch.matmul(b.T, m) * m + torch.matmul(b.T, a) * a / torch.square(torch.linalg.norm(a))
    
    # Normalize
    a /= torch.linalg.norm(a)
    b /= torch.linalg.norm(b)

    # Generating initial random vectors (uncorrelated)
    x = torch.randn(2, 1, dtype=dtype, device=device)
    
    # Apply transformation
    transformed = L @ x

    # Apply the transformation to a and b to get u and v
    u = transformed[0] * a
    v = transformed[1] * b
    
    return u.squeeze(-1), v.squeeze(-1)
