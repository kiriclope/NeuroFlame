import numpy as np


def angle_AB(A, B):
      A_norm = A / (np.linalg.norm(A) + 1e-5)
      B_norm = B / (np.linalg.norm(B) + 1e-5)

      return np.arccos(A_norm @ B_norm) * 180 / np.pi


def compute_cov(model,  order=0):
    odors = model.odors.cpu().numpy()
    U = model.low_rank.U.cpu().detach().numpy().T
    V = model.low_rank.V.cpu().detach().numpy().T

    if order==3:
        vectors = [V[0], V[1], odors[1], -odors[6]]
    elif order==2:
        vectors = [U[0] * U[1], V[0] * V[1], odors[4], -odors[9]]
    elif order==1:
        vectors = [V[0], V[1], odors[0], -odors[5]]
    elif order==0:
        vectors = [U[0], V[0], U[1], V[1]]

    num_vectors = len(vectors)
    cov_matrix = np.zeros((num_vectors, num_vectors))

    for i in range(num_vectors):
        for j in range(num_vectors):
            cov_matrix[i][j] = angle_AB(vectors[i], vectors[j])

    return cov_matrix
