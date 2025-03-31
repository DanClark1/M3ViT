import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ---------------------------
# Step 1: Simulate Homogeneous Data
# ---------------------------
np.random.seed(0)
n_samples = 1000  # total number of samples
d = 50            # dimensionality of data

# Create data from a standard normal distribution (covariance = I)
data = np.random.multivariate_normal(np.zeros(d), np.eye(d), size=n_samples)

# ---------------------------
# Step 2: Standard PCA on Pooled Data
# ---------------------------
pca = PCA()
pca.fit(data)
explained_variance = pca.explained_variance_ratio_

plt.figure(figsize=(8, 6))
plt.plot(range(1, d + 1), explained_variance, 'o-', label="Standard PCA")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Scree Plot for Standard PCA on Homogeneous Data")
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------
# Step 3: Simulate Federated Setting
# ---------------------------
n_clients = 5
client_data = np.array_split(data, n_clients)

def compute_local_PCs(client_data, n_components=10):
    """
    Compute the top n_components local principal components for each client.
    Returns a list of local PC matrices of shape (d, n_components).
    """
    local_PCs = []
    for client in client_data:
        # Compute the sample covariance matrix for the client data.
        cov_mat = np.cov(client.T)
        # Eigen-decomposition (eigh returns sorted eigenvalues in ascending order)
        eigenvals, eigenvecs = np.linalg.eigh(cov_mat)
        # Sort eigenvectors in descending order of eigenvalues
        idx = np.argsort(eigenvals)[::-1]
        eigenvecs = eigenvecs[:, idx]
        local_PCs.append(eigenvecs[:, :n_components])
    return local_PCs

local_PCs = compute_local_PCs(client_data, n_components=10)

# ---------------------------
# Step 4: Compute Misalignment Parameter (θ)
# ---------------------------
def compute_avg_projection(V_list):
    """
    Computes the average projection matrix from a list of local PC matrices.
    """
    d = V_list[0].shape[0]
    P_avg = np.zeros((d, d))
    for V in V_list:
        # Projection matrix for V (V must be orthonormal: V.T @ V = I)
        P_avg += V @ V.T
    P_avg /= len(V_list)
    return P_avg

P_avg = compute_avg_projection(local_PCs)
eigvals_avg = np.linalg.eigvalsh(P_avg)  # ascending order
lambda_max = np.max(eigvals_avg)
theta = 1 - lambda_max
print("Misalignment parameter (theta):", theta)
print("Largest eigenvalue of average projection (lambda_max):", lambda_max)

# ---------------------------
# Step 5: Compare with Pooled PCA (Global Components)
# ---------------------------
# Compute the pooled covariance matrix from the entire dataset
pooled_cov = np.cov(data.T)
eigenvals_pooled, _ = np.linalg.eigh(pooled_cov)
eigenvals_pooled = eigenvals_pooled[::-1]  # sort in descending order
explained_variance_pooled = eigenvals_pooled / np.sum(eigenvals_pooled)

plt.figure(figsize=(8, 6))
plt.plot(range(1, d + 1), explained_variance_pooled, 'o-', label="Pooled PCA")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Scree Plot for Pooled PCA on Homogeneous Data")
plt.legend()
plt.grid(True)
plt.show()
