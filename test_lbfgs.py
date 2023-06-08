import torch
import time
import scipy.linalg
import scipy.sparse.linalg
from sklearn.utils.extmath import _randomized_eigsh
from kernels import kernel_factory
from scipy.optimize import minimize
import numpy as np
TensorType = torch.DoubleTensor
torch.set_default_tensor_type(TensorType)

def dual(G, H):
    GH = G @ H
    HTGH = H.t() @ GH
    eigvals = torch.real(torch.linalg.eigvals(HTGH))
    R = torch.sum(torch.sqrt(eigvals))
    loss = 1 / 2.0 * torch.trace(H.t() @ H) - R
    return -float(loss)


def dual_scipy(G, s, maxiter=10):
    N = G.shape[0]
    torch.manual_seed(5)
    H = torch.randn((N,s)).double()

    eta = 1

    GH, HTGH = None, None

    def f(H):
        global HTGH
        global GH
        H = torch.from_numpy(H).view(N,s)
        GH = G @ H
        HTGH = H.t() @ GH
        eigvals = torch.real(torch.linalg.eigvals(HTGH))
        R = torch.sum(torch.sqrt(eigvals))
        loss = eta/2.0 * torch.trace(H.t() @ H) - R
        return loss.double().numpy()


    def grad(H):
        global HTGH
        global GH
        H = torch.from_numpy(H).view(N,s)
        U, D, V = torch.svd(HTGH, some=False)
        D = torch.diag(1./(2*torch.sqrt(D)))
        grad = eta * H - 2*GH@U@D@U.t()
        return grad.flatten().double().numpy()

    res = minimize(f, H.numpy().flatten(), method='L-BFGS-B', jac=grad,
                   options={'gtol': 0, 'disp': False, 'ftol': 0, 'maxiter': maxiter})
    H = torch.from_numpy(res.x).view(N,s)

    return H

#Define KPCA problem
torch.manual_seed(2)
N = 15000
d = 256
X = torch.randn((N, d))
X = X - X.mean(axis=0)
kernel = kernel_factory("laplace", {"sigma2": X.shape[1] * X.var() * 0.01})
G = kernel(X.t())
target_err = 1
s = 20
bench_times = 5

# FULL SVD
t0 = time.time()
U1, S1, V1t = scipy.linalg.svd(G.numpy())
time_svd = time.time() - t0
U1, S1, V1 = torch.from_numpy(U1), torch.from_numpy(S1), torch.from_numpy(V1t).t()
optimal_err = 0.5*float(torch.sum(S1[:s]))

# LBFGS
time_lbfgs = 0
for _ in range(bench_times):
    t0 = time.time()
    H = dual_scipy(G, s)
    time_lbfgs += time.time() - t0
time_lbfgs /= bench_times
lbfgs_err_abs = float(dual(G, H))
lbfgs_err = abs(lbfgs_err_abs-optimal_err) / optimal_err * 100

# LANCZOS
time_eigsh = 0
for _ in range(bench_times):
    t0 = time.time()
    S2, V2 = scipy.sparse.linalg.eigsh(G.numpy(), k=s)
    time_eigsh += time.time() - t0
    S2, V2 = torch.from_numpy(S2), torch.from_numpy(V2)
    eigsh_err_abs = float(dual(G, V2@torch.diag(torch.sqrt(S2))))
    eigsh_err = abs(eigsh_err_abs-optimal_err) / optimal_err * 100
time_eigsh /= bench_times

# RANDOMIZED SVD
time_rsvd = 0
for _ in range(bench_times):
    t0 = time.time()
    S, U = _randomized_eigsh(G.numpy(), n_components=s, selection="module", random_state=1)
    time_rsvd += time.time() - t0
    S, U = torch.from_numpy(S), torch.from_numpy(U)
    randsvd_err_abs = float(dual(G, U@torch.diag(torch.sqrt(S))))
    randsvd_err = abs(randsvd_err_abs-optimal_err) / optimal_err * 100
time_rsvd /= bench_times
