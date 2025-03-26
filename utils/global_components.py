import torch
import numpy as np
from tqdm import tqdm
from sklearn.utils.extmath import randomized_svd

class PerPCA:
    def __init__(self, r1, r2, num_iter=100, eta=0.01, tol=1e-5, device=None):
        self.r1 = r1
        self.r2 = r2
        self.num_iter = num_iter
        self.eta = eta
        self.tol = tol
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def polar(self, W):
        U, _, Vh = torch.linalg.svd(W, full_matrices=False)
        return U @ Vh

    def initialize_components(self, d):
        U = torch.linalg.qr(torch.randn(d, self.r1, device=self.device))[0]
        V = torch.linalg.qr(torch.randn(d, self.r2, device=self.device))[0]
        V = V - U @ (U.T @ V)
        return U, torch.linalg.qr(V)[0]

    def compute_covariance(self, Y):
        return (Y @ Y.T) / Y.shape[1]

    def fit(self, clients):
        clients = [c.to(self.device) for c in clients]
        S_list = [self.compute_covariance(Y) for Y in clients]
        d = clients[0].shape[0]

        U, V0 = self.initialize_components(d)
        V_list = [V0.clone() for _ in range(len(clients))]
        U_old = U.clone()

        for _ in range(self.num_iter):
            U_updates = []
            for i, S in enumerate(S_list):
                V = self.polar(V_list[i] - U @ (U.T @ V_list[i]))
                grad_U = S @ U
                grad_V = S @ V

                U_local = self.polar(U + self.eta * grad_U)
                V_local = self.polar(V + self.eta * grad_V)
                V_local = self.polar(V_local - U_local @ (U_local.T @ V_local))

                U_updates.append(U_local)
                V_list[i] = V_local

            U = self.polar(torch.stack(U_updates, dim=2).mean(dim=2))
            if torch.norm(U - U_old) < self.tol:
                break
            U_old = U.clone()

        return U.cpu(), [V.cpu() for V in V_list]  # PyTorch version

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pooled_scree(covs, max_components=20):
    S_avg = sum(covs) / len(covs)
    U, s, _ = randomized_svd(S_avg, n_components=max_components, random_state=0)
    total = np.trace(S_avg)
    explained = np.cumsum(s**2) / total
    return explained


def global_variance_explained(U, covs):
    P = U @ U.T
    num = sum(torch.trace(P @ S) for S in covs)
    den = sum(torch.trace(S) for S in covs)
    return (num / den).item()

def make_synthetic(num_clients=5, d=50, n=500, true_r1=6, true_r2=2, noise_std=1.0):
    U_true = torch.linalg.qr(torch.randn(d, true_r1, device=device))[0]
    Ys = []
    for _ in range(num_clients):
        V_true = torch.linalg.qr(torch.randn(d, true_r2, device=device))[0]
        shared = U_true @ torch.randn(true_r1, n, device=device)
        local  = V_true @ torch.randn(true_r2, n, device=device)
        Ys.append((shared + local + noise_std * torch.randn(d, n, device=device)).detach().cpu().numpy())
    return Ys


def get_num_global_components(clients):
    covs = [(Y @ Y.T) / Y.shape[1] for Y in clients]

    candidate_r1 = list(range(1, 20))
    gv = []

    for r1 in tqdm(candidate_r1):
        model = PerPCA(r1=r1, r2=383, eta=0.01, tol=5e-2)
        U, _ = model.fit(clients)
        gv.append(global_variance_explained(U.to(device), covs))

    gv = np.array(gv)
    second_diff = np.abs(np.diff(gv, n=2))
    optimal_r1 = candidate_r1[np.argmax(second_diff) + 1]

    print("Elbow (2nd diff) =", optimal_r1)




if __name__ == "__main__":


    clients = make_synthetic(num_clients=5, d=384, n=500, true_r1=50, true_r2=300, noise_std=0.1)

    get_num_global_components(clients)


    # covs    = [(Y @ Y.T) / Y.shape[1] for Y in clients]

    # candidate_r1 = list(range(20, 101))
    # gv = []

    # for r1 in tqdm(candidate_r1):
    #     model = PerPCA(r1=r1, r2=383, eta=0.01, tol=1e-3)
    #     U, _ = model.fit(clients)
    #     gv.append(global_variance_explained(U.to(device), covs))

    # gv = np.array(gv)
    # second_diff = np.abs(np.diff(gv, n=2))
    # optimal_r1 = candidate_r1[np.argmax(second_diff) + 1]

    # print("Elbow (2nd diff) =", optimal_r1)

