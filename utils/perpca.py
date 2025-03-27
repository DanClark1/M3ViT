import torch

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

        return U.cpu(), [V.cpu() for V in V_list]

    def compute_explained_variance(self, clients, U):
        """
        Compute the explained variance ratio for global components U across all clients.
        
        Args:
            clients: List of client data matrices
            U: Global components matrix
        
        Returns:
            explained_variance_ratio: Proportion of variance explained by each component
        """
        total_var = 0
        explained_var = torch.zeros(U.shape[1], device=self.device)
        U = U.to(self.device)
        for client in clients:
            client = client.to(self.device)
            # Center the data
            client = client - client.mean(dim=1, keepdim=True)
            
            # Project onto U
            proj = client.T @ U
            
            # Compute variance explained by each component
            explained_var += torch.sum(proj**2, dim=0)
            total_var += torch.sum(client**2)
        
        return (explained_var / total_var).cpu()


if __name__ == "__main__":
    num_clients, d, n = 16, 384, 2500
    clients = [torch.randn(d, n) for _ in range(num_clients)]
    model = PerPCA(r1=200, r2=100, eta=0.01)
    U, V_list = model.fit(clients)

    print("Global PCs (U):", U.T)
    print("Local PCs for first client (V[0]):", V_list[0].T)
