import torch
import torch.nn.functional as F

def batched_power_iteration(cov, num_iters=20):
    """
    Perform batched power iteration on a batch of covariance matrices.
    cov: tensor of shape (N, d, d)
    Returns: tensor of shape (N, d, 1) containing the top eigenvector for each matrix.
    """
    N, d, _ = cov.shape
    # Initialize a random vector for each client (batch element)
    v = torch.randn(N, d, 1, device=cov.device)
    # Normalize along the d-dimension
    v = v / v.norm(dim=1, keepdim=True)
    for _ in range(num_iters):
        v = torch.bmm(cov, v)      # Batched matrix multiplication: (N, d, d) x (N, d, 1) -> (N, d, 1)
        v = v / v.norm(dim=1, keepdim=True)
    return v

def compute_avg_projection(Y, num_iters=20):
    """
    Given a batch of data matrices Y of shape (N, d, n),
    compute the average projection matrix from the top eigenvectors of the covariance matrices.
    Returns:
      avg_proj: averaged projection matrix of shape (d, d)
      v: tensor of shape (N, d, 1) with the top eigenvector for each client.
    """
    N, d, n = Y.shape
    # Compute the covariance matrix for each client: S = Y Y^T / n.
    cov = torch.bmm(Y, Y.transpose(1, 2)) / n  # shape (N, d, d)
    # Compute top eigenvectors using batched power iteration.
    v = batched_power_iteration(cov, num_iters=num_iters)  # shape (N, d, 1)
    # Build projection matrices for each client: Pi = v v^T.
    proj = torch.bmm(v, v.transpose(1, 2))  # shape (N, d, d)
    # Average the projection matrices over all clients.
    avg_proj = proj.mean(dim=0)  # shape (d, d)
    return avg_proj, v

def batched_power_iteration_single(A, num_iters=20):
    """
    Run power iteration on a single matrix A (d x d) to compute its top eigenvector.
    Returns the top eigenvalue (scalar) as (v^T A v).
    """
    d = A.shape[0]
    v = torch.randn(d, 1, device=A.device)
    v = v / v.norm()
    for _ in range(num_iters):
        v = A @ v
        v = v / v.norm()
    lambda_max = (v.transpose(0, 1) @ A @ v).squeeze()
    return lambda_max

def asymmetric_loss(lambda_max, N, alpha=10.0):
    """
    Compute the asymmetric loss on lambda_max.
    Target value is 1/N. Values below 1/N are penalized by a factor of alpha.
    """
    target = 1.0 / N
    loss_below = alpha * torch.square(torch.clamp(target - lambda_max, min=0))
    loss_above = torch.square(torch.clamp(lambda_max - target, min=0))
    return loss_below + loss_above

# Example usage:
N = 3        # Number of clients
d = 400      # Data dimension
n = 100      # Number of samples per client

# Simulate a batch of data matrices: shape (N, d, n)
Y = torch.randn(N, d, n)

# Compute the average projection matrix and the batched top eigenvectors.
avg_proj, v = compute_avg_projection(Y, num_iters=50)
# Compute the largest eigenvalue of the averaged projection matrix via power iteration.
lambda_max = batched_power_iteration_single(avg_proj, num_iters=50)
# Compute the asymmetric loss.
loss = asymmetric_loss(lambda_max, N, alpha=10.0)

print("lambda_max =", lambda_max.item())
print("loss =", loss.item())
