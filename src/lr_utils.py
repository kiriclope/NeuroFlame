import time
import torch
from torch import nn


def masked_normalize(tensor):
    # Create a mask for non-zero elements
    mask = tensor != 0
    normalized_tensor = tensor.clone()  # Create a clone to avoid in-place modification
    if mask.any():
        masked_tensor = tensor[mask]
        mean = masked_tensor.mean()
        std = (
            masked_tensor.std(unbiased=False) + 1e-6
        )  # Adding epsilon for numerical stability
        # Normalize only the non-zero elements and replace them in the clone
        normalized_tensor[mask] = (masked_tensor - mean) / std
    return normalized_tensor


def ortho_quench_lr(model):
    eigenvalues, eigenvectors = torch.linalg.eig(
        model.Wab_T[model.slices[0], model.slices[0]].T
    )

    # Eigenvalues are complex, get the eigenvalue with the largest real part
    max_real_index = torch.argmax(eigenvalues.real)
    m = eigenvectors[:, max_real_index]

    model.PHI0[0], model.PHI0[2] = get_ortho_pertu(m, 0.0, device=model.device)

    Lij = torch.outer(model.PHI0[0], model.PHI0[0])
    Lij = Lij + torch.outer(model.PHI0[2], model.PHI0[2])

    model.lr = model.Jab[0][0] * model.KAPPA[0][0] * Lij / torch.sqrt(model.Ka[0])
    model.Wab_T[model.slices[0], model.slices[0]].add_(model.lr.T)
    model.Wab_T[model.slices[0], model.slices[0]].clamp_(min=0.0)


def initLR(model):
    # Low rank vector
    model.U = nn.Parameter(
        torch.randn((model.N_NEURON, int(model.RANK)), device=model.device) * 0.1
    )

    model.V = nn.Parameter(
        torch.randn((model.N_NEURON, int(model.RANK)), device=model.device) * 0.1
    )

    if model.LR_KAPPA == 1:
        model.lr_kappa = nn.Parameter(torch.rand(1, device=model.device))
    else:
        model.lr_kappa = torch.tensor(1.0, device=model.device)

    # Mask to train excitatory neurons only
    model.lr_mask = torch.zeros((model.N_NEURON, model.N_NEURON), device=model.device)

    if model.LR_MASK == 0:
        model.lr_mask[model.slices[0], model.slices[0]] = 1.0
    if model.LR_MASK == 1:
        model.lr_mask[model.slices[1], model.slices[1]] = 1.0
    if model.LR_MASK == -1:
        model.lr_mask = torch.ones(
            (model.N_NEURON, model.N_NEURON), device=model.device
        )

    # Linear readout for supervised learning
    model.linear = nn.Linear(
        model.Na[0], model.LR_CLASS, device=model.device, bias=model.LR_BIAS
    )

    model.dropout = nn.Dropout(model.DROP_RATE)

    model.odors = torch.randn(
        (3, model.Na[0]),
        device=model.device,
    )

    if model.LR_FIX_READ:
        for param in model.linear.parameters():
            param.requires_grad = False

    # Window where to evaluate loss
    if model.LR_EVAL_WIN == -1:
        model.lr_eval_win = -1
    else:
        model.lr_eval_win = int(model.LR_EVAL_WIN / model.DT / model.N_WINDOW)


def get_theta(a, b, IF_NORM=0):
    u, v = a, b

    if IF_NORM:
        u = a / torch.norm(a, p="fro")
        v = b / torch.norm(b, p="fro")

    return torch.atan2(v, u)


def get_idx(ksi, ksi1):
    theta = get_theta(ksi, ksi1, GM=0, IF_NORM=0)
    return theta.argsort()


def get_overlap(model, rates):
    ksi = model.PHI0.cpu().detach().numpy()
    return rates @ ksi.T / rates.shape[-1]


def get_ortho_pertu(m, s, device):
    m = m.real.unsqueeze(-1)
    N = m.shape[0]
    # Covariance matrix for u and v
    C = torch.tensor([[1, s], [s, 1]], device=device)

    # Cholesky decomposition
    L = torch.linalg.cholesky(C)

    # Generate a basis orthogonal to m
    # Simple method: Use random vectors and orthogonalize
    a = torch.randn(N, 1, device=device)
    b = torch.randn(N, 1, device=device)

    # Orthogonalize a and b wrt m
    a -= torch.matmul(a.T, m) * m
    b -= torch.matmul(b.T, m) * m + torch.matmul(b.T, a) * a / torch.square(
        torch.linalg.norm(a)
    )

    # Normalize
    a /= torch.linalg.norm(a)
    b /= torch.linalg.norm(b)

    # Generating initial random vectors (uncorrelated)
    x = torch.randn(2, 1, device=device)

    # Apply transformation
    transformed = L @ x

    # Apply the transformation to a and b to get u and v
    u = transformed[0] * a
    v = transformed[1] * b

    return u.squeeze(-1), v.squeeze(-1)


def gen_ortho_vec(m, desired_cov, random_seed=None, device="cuda"):
    if random_seed is not None:
        torch.manual_seed(random_seed)  # For reproducibility

    N = m.size(0)
    # Step 1: Generate a random vector u and make it orthogonal to m
    u = torch.randn(N, device=device)
    # u -= u.dot(m) / m.dot(m) * m

    # Step 2: Generate a random vector v that is orthogonal to both m and u
    v = torch.randn(N, device=device)
    # v -= v.dot(m) / m.dot(m) * m
    # v -= v.dot(u) / u.dot(u) * u

    # Normalize u and v to ensure they are unit vectors (optional)
    # u = u / u.norm()
    # v = v / v.norm()

    # Step 3: Construct the desired covariance matrix
    cov_matrix = torch.tensor([[1, desired_cov], [desired_cov, 1]])

    # Step 4: Obtain the Cholesky decomposition of the covariance matrix
    L = torch.linalg.cholesky(cov_matrix)

    # Step 5: Use L to generate vectors with the desired covariance
    u_prime = L[0, 0] * u + L[0, 1] * v
    v_prime = L[1, 0] * u + L[1, 1] * v

    return u_prime, v_prime


def gen_v_cov(u, cov, device="cuda"):
    seed = int(time.time())
    torch.manual_seed(seed)

    N = u.size(0)

    v = torch.randn(N, device=device)
    v -= v.dot(u) / u.dot(u) * u

    a = cov
    b = torch.sqrt(1.0 - cov**2)

    return a * u + b * v
