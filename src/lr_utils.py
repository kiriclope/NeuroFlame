import torch

# if self.PROBA_TYPE[0][0] == 'lr_add':
            
#     eigenvalues, eigenvectors = torch.linalg.eig(self.Wab_T[self.slices[0], self.slices[0]].T)
            
#     # Eigenvalues are complex, get the eigenvalue with the largest real part
#     max_real_index = torch.argmax(eigenvalues.real)
#     m = eigenvectors[:, max_real_index]

#     self.PHI0[0], self.PHI0[2] = get_ortho_pertu(m, 0.0, dtype=self.FLOAT, device=self.device)
            
#     Lij = torch.outer(self.PHI0[0], self.PHI0[0])
#     Lij = Lij + torch.outer(self.PHI0[2], self.PHI0[2])

#     self.lr = self.Jab[0][0] * self.KAPPA[0][0] * Lij / torch.sqrt(self.Ka[0])
#     self.Wab_T[self.slices[0], self.slices[0]].add_(self.lr.T)
#     # self.Wab_T[self.slices[0], self.slices[0]].clamp_(min=0.0)

def initLR(model):
    # Low rank vector
    model.U = nn.Parameter(torch.randn((model.N_NEURON, int(model.RANK)),
                                        device=model.device, dtype=model.FLOAT))

    # model.V = nn.Parameter(torch.randn((model.N_NEURON, int(model.RANK)),
    # device=model.device, dtype=model.FLOAT))

    # Mask to train excitatory neurons only
    model.mask = torch.zeros((model.N_NEURON, model.N_NEURON),
                            device=model.device, dtype=model.FLOAT)

    model.mask[model.slices[0], model.slices[0]] = 1.0

    # Linear readout for supervised learning
    model.linear = nn.Linear(model.Na[0], 1, device=model.device, dtype=model.FLOAT, bias=False)
    model.lr_kappa = nn.Parameter(5 * torch.rand(1))

    # Window where to evaluate loss
    model.lr_eval_win = int(model.LR_EVAL_WIN / model.DT / model.N_WINDOW)

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
