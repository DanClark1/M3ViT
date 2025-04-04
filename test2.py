import numpy as np

def compute_theta(local_bases):
    """
    Given a list of local bases (each with orthonormal columns),
    compute the average projection matrix and return:
      - theta = 1 - lambda_max(average projection)
      - lambda_max, and the average projection matrix.
    """
    N = len(local_bases)
    # Each local projection matrix is B @ B.T.
    proj_matrices = [B @ B.T for B in local_bases]
    avg_proj = sum(proj_matrices) / N
    # Use eigen-decomposition (since avg_proj is symmetric) to compute the maximum eigenvalue.
    eigvals = np.linalg.eigvalsh(avg_proj)
    lambda_max = np.max(eigvals)
    theta = 1 - lambda_max
    return theta, lambda_max, avg_proj

# Set the ambient dimension (e.g., 400)
d = 400
# Number of clients
N = 3
# For simplicity, assume each client has a single local direction (rank 1).

# Scenario 1: Identical local PCs (all clients use the same direction)
v = np.random.randn(d, 1)
v, _ = np.linalg.qr(v)  # Orthonormalize
local_bases_same = [v for _ in range(N)]
theta_same, lambda_max_same, _ = compute_theta(local_bases_same)
print("Scenario 1: Identical local PCs")
print("  lambda_max =", lambda_max_same)
print("  theta =", theta_same)
print("  (Expect theta ~ 0, since all projections are identical.)\n")

# Scenario 2: Mutually orthogonal local PCs across clients
# We generate N mutually orthogonal unit vectors in R^d.
# One way is to generate a d x N matrix and orthonormalize its columns.
Q, _ = np.linalg.qr(np.random.randn(d, N))
local_bases_orthogonal = [Q[:, [i]] for i in range(N)]
theta_orth, lambda_max_orth, _ = compute_theta(local_bases_orthogonal)
print("Scenario 2: Mutually orthogonal local PCs")
print("  lambda_max =", lambda_max_orth)
print("  theta =", theta_orth)
print("  (For N =", N, "clients, we expect lambda_max ~ 1/N and theta ~", 1 - 1/N, ".)\n")

# Scenario 3: Random independent local PCs (each client has an independently generated unit vector)
local_bases_random = []
for i in range(N):
    A = np.random.randn(d, 1)
    Q, _ = np.linalg.qr(A)
    local_bases_random.append(Q)
theta_rand, lambda_max_rand, _ = compute_theta(local_bases_random)
print("Scenario 3: Random independent local PCs")
print("  lambda_max =", lambda_max_rand)
print("  theta =", theta_rand)
