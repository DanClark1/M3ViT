import torch
import numpy as np
from tqdm import tqdm
from utils.perpca import PerPCA  # PyTorch version

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def global_variance_explained(U, covs):
    P = U @ U.T
    num = sum(torch.trace(P @ S) for S in covs)
    den = sum(torch.trace(S) for S in covs)
    return (num / den).item()

def make_synthetic(num_clients=5, d=50, n=500, true_r1=6, true_r2=2, noise_std=0.4):
    U_true = torch.linalg.qr(torch.randn(d, true_r1, device=device))[0]
    Ys = []
    for _ in range(num_clients):
        V_true = torch.linalg.qr(torch.randn(d, true_r2, device=device))[0]
        shared = U_true @ torch.randn(true_r1, n, device=device)
        local  = V_true @ torch.randn(true_r2, n, device=device)
        Ys.append(shared + local + noise_std * torch.randn(d, n, device=device))
    return Ys

if __name__ == "__main__":
    clients = make_synthetic(num_clients=5, d=384, n=500, true_r1=50, true_r2=300, noise_std=0.1)
    covs    = [(Y @ Y.T) / Y.shape[1] for Y in clients]

    candidate_r1 = list(range(20, 101))
    gv = []

    for r1 in tqdm(candidate_r1):
        model = PerPCA(r1=r1, r2=383, eta=0.01, tol=1e-3)
        U, _ = model.fit(clients)
        gv.append(global_variance_explained(U.to(device), covs))

    gv = np.array(gv)
    second_diff = np.abs(np.diff(gv, n=2))
    optimal_r1 = candidate_r1[np.argmax(second_diff) + 1]

    print("Elbow (2nd diff) =", optimal_r1)

