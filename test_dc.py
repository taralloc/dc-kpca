# %%
import torch
import time
import scipy.linalg
from dc import KPCA_DC, RobustKPCA_DC, SparseKPCA_DC
from kernels import kernel_factory

torch.manual_seed(2)
# %%
N, d, s = 1000, 20, 15
X = torch.randn((N, d)).double()
X = X - X.mean(axis=0)
kernel = kernel_factory("rbf", {"sigma2": 1.0})
G = kernel(X.t())

# %%
model = KPCA_DC(s)
torch.manual_seed(10)
model.fit(G)
# %%
H = model.H
# %%
kappa = 10
model_robust = RobustKPCA_DC(s, kappa=kappa)
model_robust.fit(G)
print("Convergence reached:", model_robust.exit_code == 0)
# %%
# Be careful, for some values of epsilon H^TGH has some zero eigenvalues
epsilon = 0.01
model_sparse = SparseKPCA_DC(s, epsilon=epsilon)
model_sparse.fit(G)
print("Convergence reached:", model_sparse.exit_code == 0)
# %%
