import numpy as np

class PerPCA:
    def __init__(self, r1, r2, num_iter=100, eta=0.01, tol=1e-5):
        self.r1 = r1
        self.r2 = r2
        self.num_iter = num_iter
        self.eta = eta
        self.tol = tol

    def polar(self, W):
        # Compute the polar projection of matrix W onto the Stiefel manifold.
        U, _, Vh = np.linalg.svd(W, full_matrices=False)
        return U @ Vh

    def initialize_components(self, d):
        # Initialize a random orthonormal global matrix U (d x r1)
        # and a random orthonormal local matrix V (d x r2) that is orthogonal to U.
        U = np.linalg.qr(np.random.randn(d, self.r1))[0]
        V = np.linalg.qr(np.random.randn(d, self.r2))[0]
        V = V - U @ (U.T @ V)
        V = np.linalg.qr(V)[0]
        return U, V

    def compute_covariance(self, Y):
        # Compute the sample covariance matrix for data Y.
        n = Y.shape[1]
        return (Y @ Y.T) / n

    def fit(self, clients):
        # clients: list of numpy arrays, each of shape (d, n)
        num_clients = len(clients)
        d = clients[0].shape[0]
        S_list = [self.compute_covariance(Y) for Y in clients]
        U, V0 = self.initialize_components(d)
        V_list = [V0.copy() for _ in range(num_clients)]
        U_old = U.copy()
        it = 0
        while True:
            U_updates = []
            for i in range(num_clients):
                S = S_list[i]
                V = V_list[i]
                # Ensure local V is orthogonal to current global U.
                V = self.polar(V - U @ (U.T @ V))
                grad_U = S @ U
                grad_V = S @ V
                U_local = self.polar(U + self.eta * grad_U)
                V_local = self.polar(V + self.eta * grad_V)
                V_local = self.polar(V_local - U_local @ (U_local.T @ V_local))
                U_updates.append(U_local)
                V_list[i] = V_local
            U_avg = np.mean(np.stack(U_updates, axis=2), axis=2)
            U = self.polar(U_avg)
            
            # Convergence condition based on change in U
            diff = np.linalg.norm(U - U_old, 'fro')
            if diff < self.tol:
                break
            U_old = U.copy()

            it += 1
        return U, V_list

if __name__ == "__main__":
    # Example usage: 5 clients each with a d x n matrix (e.g., d=50, n=200)
    num_clients = 5
    d = 500
    n = 2500
    clients = [np.random.randn(d, n) for _ in range(num_clients)]
    
    r1 = 200  # number of global PCs
    r2 = 100  # number of local PCs per client
    model = PerPCA(r1, r2, num_iter=100, eta=0.01, tol=1e-5)
    U, V_list = model.fit(clients)
    
    print("Global PCs (U):")
    print(U.T)
    print("\nLocal PCs for first client (V[0]):")
    print(V_list[0].T)
