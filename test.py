import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd
from kneed import KneeLocator
from tqdm import tqdm
from utils.perpca import PerPCA
from kneed import KneeLocator





def global_variance_explained(U, covs):
    # U is returned as (r1, d)
    P = U @ U.T           # projection matrix (d×d)
    num = sum(np.trace(P @ S) for S in covs)
    den = sum(np.trace(S) for S in covs)
    return num/den


def make_synthetic(num_clients=5, d=50, n=500, true_r1=6, true_r2=2, noise_std=0.4):
    """Generate each client’s data = shared + local + noise."""
    U_true = np.linalg.qr(np.random.randn(d, true_r1))[0]
    Ys = []
    for _ in range(num_clients):
        V_true = np.linalg.qr(np.random.randn(d, true_r2))[0]
        shared = U_true @ np.random.randn(true_r1, n)
        local  = V_true @ np.random.randn(true_r2, n)
        Ys.append(shared + local + noise_std * np.random.randn(d, n))
    return Ys

def broken_stick_threshold(r, d):
    return sum(1/(i) for i in range(r, d+1)) / d



# if __name__ == "__main__":
#     # 1) Generate true shared + local data (shape = d×n)
#     clients = make_synthetic(num_clients=5, d=384, n=500, true_r1=10, true_r2=241, noise_std=0.1)

#     # 2) Precompute each client’s covariance (d×d)
#     covs = [(Y @ Y.T) / Y.shape[1] for Y in clients]

#     candidate_r1 = list(range(1, 21))
#     gv = []
#     for r1 in tqdm(candidate_r1):
#         model = PerPCA(r1=r1, r2=383, num_iter=200, eta=0.01, tol=1e-2)
#         U, _ = model.fit(clients)            # expects list of (d,n)
#         gv.append(global_variance_explained(U, covs))

#     # # candidate_r1 and gv from your code
#     # kl = KneeLocator(candidate_r1, gv, curve="convex", direction="increasing")
#     # optimal_r1 = kl.knee

#     # bs = [broken_stick_threshold(r, d=50) for r in candidate_r1]
#     # optimal_r1 = max(r for r, explained, thresh in zip(candidate_r1, gv, bs) if explained >= thresh)
#     # print("Elbow (broken-stick) =", optimal_r1)


#     print(gv)
#     diff = np.diff(gv, n=1)
#     second_diff = np.abs(np.diff(gv, n=2))
#     optimal_r1 = candidate_r1[np.argmax(second_diff) + 1]
#     print('diff:', diff)
#     print('second_diff:', second_diff)
#     print("Elbow (2nd diff) =", optimal_r1)


#     print("Elbow (Kneedle) =", optimal_r1)
#     plt.plot(candidate_r1, gv, marker="o")
#     plt.title("PerPCA Scree Plot")
#     plt.xlabel("Number of global PCs (r₁)")
#     plt.ylabel("Shared‑variance fraction")
#     plt.grid(True)
#     plt.show()


def broken_stick_threshold(r, d):
    return sum(1/i for i in range(r, d+1)) / d

def pooled_scree(covs, max_components=20):
    S_avg = sum(covs) / len(covs)
    U, s, _ = randomized_svd(S_avg, n_components=max_components, random_state=0)
    total = np.trace(S_avg)
    explained = np.cumsum(s**2) / total
    return explained




if __name__ == "__main__":
    clients = make_synthetic(num_clients=5, d=384, n=500, true_r1=30, true_r2=50, noise_std=0.1)
    covs = [(Y @ Y.T) / Y.shape[1] for Y in clients]

    candidate_r1 = list(range(1, 381))
    gv = pooled_scree(covs, max_components=380)

    # Kneedle elbow
    kl = KneeLocator(candidate_r1, gv, curve="convex", direction="increasing")
    elbow_knee = kl.knee

    # 2nd‑difference elbow
    second_diff = np.abs(np.diff(gv, n=7))
    elbow_2nd = candidate_r1[np.argmax(second_diff) + 1]

    print("Shared‑variance fractions:", np.round(gv, 3))
    print("Elbow (Kneedle) =", elbow_knee)
    print("Elbow (2nd diff) =", elbow_2nd)



    # Compute differences
    diff1 = np.diff(gv, n=1)
    diff2 = np.diff(gv, n=2)
    diff3 = np.diff(gv, n=3)

    fig, ax1 = plt.subplots()

    # Left axis: shared‑variance
    ax1.plot(candidate_r1, gv, marker="o", label="Shared‑variance")
    ax1.set_xlabel("Number of global PCs (r₁)")
    ax1.set_ylabel("Shared‑variance fraction")
    ax1.grid(True)

    # Right axis: differences
    ax2 = ax1.twinx()
    ax2.plot(candidate_r1[1:], diff1, marker="o", linestyle="--", label="1st diff")
    ax2.plot(candidate_r1[2:], diff2, marker="o", linestyle="--", label="2nd diff")
    ax2.plot(candidate_r1[3:], diff3, marker="o", linestyle="--", label="3rd diff")
    ax2.set_ylabel("Difference")

    # Combined legend
    lines, labels, *_ = ax1.get_legend_handles_labels() + ax2.get_legend_handles_labels()
    ax1.legend(lines, labels, loc="best")

    plt.title("Shared‑variance & Its Differences")
    plt.show()


