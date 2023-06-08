import torch
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas
import numpy as np
from sklearn.decomposition._kernel_pca import KernelPCA

class DCA(ABC):
    # Abstract class implementing the simple DCA algorithm
    # The problem to solve is inf g(x) - f(x) with (f,g) convex functions
    # For the KPCA problem f is the nuclear norm
    # g is chosen depending on the desired properties (robustness, sparsity)
    # g^\star is understood as some variance-like function

    def __init__(self, s, tol=1e-4, max_iter=1000):
        self.s = s
        self.tol = tol
        self.max_iter = max_iter
        self._sklearnmodel = None

    def fit(self, G):
        # Initializing the dual variable H
        self.n = G.shape[0]
        self.G = G
        torch.manual_seed(8) #reproduciblity
        self.H = torch.randn((self.n, self.s)).double()
        self.losses = [self.evaluate_loss()]
        self.train_table = pandas.DataFrame()
        # Applying the DCA algorithm
        for n_iter in range(self.max_iter):
            self.step()
            if self.stopping_criterion():
                self.exit_code = 0
                return
        self.exit_code = 1
        # The variable exit_code informs about the termination of the algorithm
        # It should be 0, if not the convergence is not attained
        return

    def step(self):
        # Perform a simple dca step for minimizing difference
        # of convex functions g(x) - f(x)
        # This amounts to y_k \in \partial f(x_k) followed by
        #  x_{k+1} \in \partial g^/star(y_k)
        new_y = self.subdifferential_nuclear_norm()
        new_x = self.subdifferential_variance(new_y)
        self.H = new_x
        # Storing the new loss value for computing stopping criterion
        self.losses.append(self.evaluate_loss())

    def stopping_criterion(self):
        # Simple criterion based on the absolute variation of the loss
        delta_loss = torch.abs(self.losses[-1] - self.losses[-2])
        return delta_loss <= self.tol

    def subdifferential_nuclear_norm(self):
        # Computed by hand instead of pytorch autodiff
        U, D, _ = torch.svd(self.H.t() @ self.G @ self.H, some=False)
        D = torch.diag(1./(2*torch.sqrt(D)))
        grad = 2*self.G @ self.H @ U @ D @ U.t()
        return grad

    def transform(self, Xtest, Xtrain, kernel):
        G, H = self.G, self.H
        Gtest = kernel(Xtest.t(), Xtrain.t())
        return Gtest @ H

    def inverse_transform(self, H, Xtrain, kernel):
        def get_kernel(_, x, y=None):
            if y is None:
                y = x
            return kernel(x.t(), y.t()).numpy()
        if self._sklearnmodel is None:
            self._sklearnmodel = type('KPCAModel', tuple(), {'fit_inverse_transform': True, '_get_kernel': get_kernel,
                                            'X_transformed_fit_': None, 'dual_coef_': None, 'alpha': 1.0})()
        KernelPCA._fit_inverse_transform(self._sklearnmodel, self.transform(Xtrain, Xtrain, kernel), Xtrain)
        return KernelPCA.inverse_transform(self._sklearnmodel, H)

    @abstractmethod
    def evaluate_loss(self):
        # Abstract method for loss evalutation, depends on
        # the specific instantiation of the loss
        pass

    @abstractmethod
    def subdifferential_variance(self, new_y):
        # Abstract method for subdifferential of 0.5 ||.||^2 \infconv something
        return

    def plot_losses(self):
        if not hasattr(self, 'losses'):
            print('No losses to plot')
        else:
            plt.figure()
            plt.plot(self.losses)
            plt.show()


class KPCA_DC(DCA):

    def __init__(self, s, tol=0.000001, max_iter=1000):
        super().__init__(s, tol, max_iter)

    def evaluate_loss(self):
        A = 0.5 * torch.linalg.norm(self.H, ord='fro')**2
        B = torch.linalg.norm(self.H.t() @ self.G @ self.H, ord='nuc')
        return A - B

    def subdifferential_variance(self, new_y):
        return new_y

    def get_kappa_max(self, norm='2'):
        if not hasattr(self, 'H'):
            print('Model not fitted')
        else:
            if norm == '2':
                return (self.H**2).sum(axis=1).max().sqrt()
            elif norm == 'inf':
                return self.H.abs().max()
            else:
                raise NameError("Norm not implemented")


class RobustKPCA_DC(DCA):

    def __init__(self, s, tol=0.0001, max_iter=10000, kappa=1., norm='2'):
        super().__init__(s, tol, max_iter)
        assert norm in ['2', 'inf']
        self.kappa = kappa
        self.norm = norm

    def evaluate_loss(self):
        A = 0.5 * torch.linalg.norm(self.H, ord='fro')**2
        B = torch.linalg.norm(self.H.t() @ self.G @ self.H, ord='nuc')
        if (self.H**2).sum(axis=1).max() > self.kappa**2 + 1e-7 and self.norm == '2':
            return float('inf')
        else:
            return A - B

    def subdifferential_variance(self, new_y):
        # Subdifferential of the moreau enveloppe of kappa * ||.|| with gamma=1
        # The function is differentiable, with gradient = x - prox_{kappa ||.||}(x)
        # Using Moreau decomposition, we get gradient = kappa Proj(x/kappa)
        if self.norm == '2':
            norms = torch.sqrt((new_y**2).sum(axis=1))
            indices = torch.where(norms > self.kappa)
            proj = new_y.clone()
            proj[indices] *= self.kappa / norms[indices].reshape((-1, 1))
            return proj
        elif self.norm == 'inf':
            norms = torch.abs(new_y)
            return torch.where(norms > self.kappa, self.kappa * new_y / norms, new_y)

    def stopping_criterion(self):
        if self.norm == 'inf':
            return False
        else:
            # Simple criterion based on the absolute variation of the loss
            delta_loss = torch.abs(self.losses[-1] - self.losses[-2])
            return delta_loss <= self.tol


class SparseKPCA_DC(DCA):

    def __init__(self, s, tol=0.0001, max_iter=10000, epsilon=1., norm='2'):
        super().__init__(s, tol, max_iter)
        assert norm in ['2', 'inf']
        self.epsilon = epsilon
        self.norm = norm

    def evaluate_loss(self):
        A = 0.5 * torch.linalg.norm(self.H, ord='fro')**2
        B = torch.linalg.norm(self.H.t() @ self.G @ self.H, ord='nuc')
        C = self.epsilon * torch.linalg.norm(self.H, ord=1 if self.norm == 'inf' else self.norm == '2')
        return A + C - B

    def subdifferential_variance(self, new_y):
        # Subdifferential of the moreau enveloppe of \iota_{Ball_\epsilon}(.) with gamma=1
        # The function is differentiable, with gradient = x - prox_{\iota_{Ball_\epsilon}(.)}(x)
        # In this case the proximal is the projection on the ball of radius epsilon
        if self.norm == '2':
            norms = torch.sqrt((new_y**2).sum(axis=1))
            indices = torch.where(norms > self.epsilon)
            proj = new_y.clone()
            proj[indices] *= self.epsilon / norms[indices].reshape((-1, 1))
            return new_y - proj
        elif self.norm == 'inf':
            return torch.where(self.H.abs() - self.epsilon < 0,
                               torch.zeros_like(self.H),
                               self.H.abs() - self.epsilon) * torch.sign(self.H)
        else:
            raise NameError("norm not implemented")

