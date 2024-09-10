from scipy.spatial.distance import pdist, squareform
import torch
from time import time
import math
from tqdm import tqdm
from numpy.polynomial.hermite_e import *
from itertools import product
from scipy import sparse
import copy
import numpy as np


class CoinSVGD:
    """
    Coin Stein Variational Gradient Descent and Stein Variational Gradient Descent.
    Adapted from https://github.com/dilinwang820/Stein-Variational-Gradient-Descent/.
    """
    def __init__(self, batch=True):
        self.batch = batch

    @staticmethod
    def svgd_kernel(theta, h=-1):
        """
        Compute RBF kernel

        Inputs:
            theta: current particles
            h: bandwidth (if < 0 then use median rule)

        Outputs:
            kxy: kernel
            dx_kxy: gradient of kernel
        """
        sq_dist = pdist(theta)
        pairwise_dists = squareform(sq_dist) ** 2

        # if h < 0, using median trick
        if h < 0:
            h = np.median(pairwise_dists)
            h = np.sqrt(0.5 * h / np.log(theta.shape[0] + 1))

        # rbf kernel
        kxy = np.exp(-pairwise_dists / h ** 2 / 2)

        # rbf kernel grad
        dx_kxy = -np.matmul(kxy, theta)
        sum_kxy = np.sum(kxy, axis=1)

        for i in range(theta.shape[1]):
            dx_kxy[:, i] = dx_kxy[:, i] + np.multiply(theta[:, i], sum_kxy)

        dx_kxy = dx_kxy / (h ** 2)

        return kxy, dx_kxy

    def svgd_update(self, theta0, ln_prob_grad, n_iter=1000, stepsize=1e-3, bandwidth=-1, alpha=0.9, adagrad=False):
        """
        SVGD

        Inputs:
            theta0: initial particle positions
            ln_prob_grad: gradient of log-probability of the target
            n_iter: number of iterations
            stepsize: step size (i.e., learning rate)
            bandwidth: bandwidth for kernel
            alpha: adagrad parameter
            adagrad: whether to use Adagrad to adapt learning rate

        Outputs:
            theta: list of theta at each iteration
        """

        # Check input
        if theta0 is None or ln_prob_grad is None:
            raise ValueError('theta0 or ln_prob_grad cannot be None')

        # initial theta
        theta = np.copy(theta0)

        # all theta
        all_theta = list()
        all_theta.append(np.copy(theta))

        # for adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0

        for t in range(n_iter):

            # log density
            if self.batch:
                ln_p_grad = ln_prob_grad(theta)
            else:
                ln_p_grad = np.zeros_like(theta)
                for k in range(theta.shape[0]):
                    ln_p_grad[k, :] = ln_prob_grad(theta[k, :])

            # kernel matrix
            kxy, dx_kxy = self.svgd_kernel(theta, h=bandwidth)

            # gradient
            grad_theta = (np.matmul(kxy, ln_p_grad) + dx_kxy) / theta0.shape[0]

            # adagrad
            if adagrad:
                if t == 0:
                    historical_grad = historical_grad + grad_theta ** 2
                else:
                    historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta ** 2)

                adj_grad = np.divide(grad_theta, fudge_factor + np.sqrt(historical_grad))

            else:
                adj_grad = grad_theta

            # theta update
            theta = theta + stepsize * adj_grad

            all_theta.append(np.copy(theta))

        return all_theta

    def coin_update(self, theta0, ln_prob_grad, n_iter=10000, bandwidth=-1):
        """
        Coin SVGD

        Inputs:
            theta0: initial particle positions
            ln_prob_grad: gradient of log-probability of the target
            n_iter: number of iterations
            bandwidth: bandwidth for kernel

        Outputs:
            theta: list of theta at each iteration
        """

        # Check input
        if theta0 is None or ln_prob_grad is None:
            raise ValueError('theta0 or ln_prob_grad cannot be None')

        # Initial theta
        theta0 = copy.deepcopy(theta0)
        theta = copy.deepcopy(theta0)

        # all theta
        all_theta = list()
        all_theta.append(np.copy(theta))

        # initialise other vars
        L = 1e-10
        grad_theta_sum = 0
        reward = 0
        abs_grad_theta_sum = 0

        for t in range(n_iter):

            # calculate grad log density
            if self.batch:
                ln_p_grad = ln_prob_grad(theta)
            else:
                ln_p_grad = np.zeros_like(theta)
                for k in range(theta.shape[0]):
                    ln_p_grad[k, :] = ln_prob_grad(theta[k, :])

            # calculate kernel matrix
            kxy, dx_kxy = self.svgd_kernel(theta, h=bandwidth)

            # gradient
            grad_theta = (np.matmul(kxy, ln_p_grad) + dx_kxy) / theta0.shape[0]

            # |gradient|
            abs_grad_theta = abs(grad_theta)

            # constant
            L = np.maximum(abs_grad_theta, L)

            # sum of gradients
            grad_theta_sum += grad_theta
            abs_grad_theta_sum += abs_grad_theta

            # 'reward'
            reward = np.maximum(reward + np.multiply(theta - theta0, grad_theta), 0)

            # theta update
            theta = theta0 + grad_theta_sum / (L * (abs_grad_theta_sum + L)) * (L + reward)

            if np.isnan(theta).any():
                theta = copy.deepcopy(theta0)

            all_theta.append(np.copy(theta))

        return all_theta


class CoinKSDD:
    """
    Coin Kernel Stein Discrepancy Descent and Kernel Stein Discrepancy Descent.
    Adapted from https://github.com/pierreablin/ksddescent
    """
    def __init__(self):
        pass

    # compute rbf kernel (pytorch)
    @staticmethod
    def gaussian_stein_kernel(x, score_x, h, return_kernel=False):
        """
        Compute RBF Stein Kernel

        Inputs
        x : torch.tensor, shape (n, p)
            Input particles
        score_x : torch.tensor, shape (n, p)
            The score of x
        h : float
            The bandwidth
        return_kernel : bool
            whether to return the original kernel k(xi, xj)

        Outputs
        stein_kernel : torch.tensor, shape (n, n)
            The linear Stein kernel
        kernel : torch.tensor, shape (n, n)
            The base kernel, only returned if return_kernel is True
        """

        _, p = x.shape
        # Gaussian kernel:
        norms = (x ** 2).sum(-1)
        dists = -2 * x @ x.t() + norms[:, None] + norms[None, :]

        # median rule
        if h < 0:
            h = torch.median(dists)
            h = torch.sqrt(0.5 * h / torch.log(torch.tensor(x.shape[0] + 1)))

        k = (-dists / 2 / h).exp()

        # Dot products:
        diffs = (x * score_x).sum(-1, keepdim=True) - (x @ score_x.t())
        diffs = diffs + diffs.t()
        scalars = score_x.mm(score_x.t())
        der2 = p - dists / h
        stein_kernel = k * (scalars + diffs / h + der2 / h)
        if return_kernel:
            return stein_kernel, k
        return stein_kernel

    def update(self, x0, score, step, n_iter=1000, h=-1, store=True, verbose=False, clamp=None):
        """
        Kernel Stein Discrepancy Descent

        Inputs
        x0 : torch.tensor, size n_samples x n_features
            initial positions
        score : callable
            function that computes the score
        step : float
            step size
        max_iter : int
            max numer of iters
        bw : float
            bandwidth of the stein kernel
        stores : None or list of ints
            whether to store the iterates at the indexes in the list
        verbose: bool
            whether to print the current loss
        clamp:
            if not None, should be a tuple (a, b). The points x are then
            constrained to stay in [a, b]

        Outputs
        x: torch.tensor, size n_samples x n_features
            The final positions
        loss_list : list of floats
            List of the loss values during iterations
        """
        x = x0.clone().detach()
        n_samples, p = x.shape
        x.requires_grad = True
        if store:
            all_x = []
            timer = []
            t0 = time()
        loss_list = []
        n = None
        for i in range(n_iter):
            if store:
                timer.append(time() - t0)
                all_x.append(x.clone())
            scores_x = score(x)
            k = self.gaussian_stein_kernel(x, scores_x, h)
            loss = k.sum() / n_samples ** 2
            loss.backward()
            loss_list.append(loss.item())
            if verbose and i % 100 == 0:
                print(i, loss.item())
            with torch.no_grad():
                x[:] -= step * x.grad
                if n is not None:
                    x[:] -= n
                if clamp is not None:
                    x = x.clamp(clamp[0], clamp[1])
                x.grad.data.zero_()
            x.requires_grad = True
        x.requires_grad = False
        if store:
            return x, all_x, timer
        else:
            return x

    def coin_update(self, x0, score, n_iter=1000, h=-1, store=True, verbose=False, clamp=None, L=.01):
        x = x0.clone().detach()
        n_samples, p = x.shape
        x.requires_grad = True
        if store:
            all_x = []
            timer = []
            t0 = time()
        loss_list = []
        L = torch.zeros_like(x)
        grad_x_sum = 0
        abs_grad_x_sum = 0
        reward = 0
        x_dot_grad_x_sum = torch.zeros(n_samples)
        n = None
        for i in range(n_iter):
            if store:
                timer.append(time() - t0)
                all_x.append(x.clone())
            scores_x = score(x)
            k = self.gaussian_stein_kernel(x, scores_x, h)
            loss = k.sum() / n_samples ** 2
            loss.backward()
            loss_list.append(loss.item())
            if verbose and i % 100 == 0:
                print(i, loss.item())
            with torch.no_grad():

                # |gradient|
                abs_grad_x = abs(x.grad)

                # constant
                L = torch.maximum(abs_grad_x, L)

                # sum of gradients
                grad_x_sum += (-x.grad)
                abs_grad_x_sum += abs_grad_x

                # 'reward'
                reward = np.maximum(reward + torch.multiply(x - x0, (-x.grad)), 0)

                # x dot gradient
                x_dot_grad_x = torch.einsum('ij,ji->i', x, -x.grad.T)

                # sum of x dot gradient
                x_dot_grad_x_sum += x_dot_grad_x

                # x update
                x[:] = x0.clone().detach() + grad_x_sum / (L * (abs_grad_x_sum + L)) * (L + reward)

                # clamp
                if n is not None:
                    x[:] -= n
                if clamp is not None:
                    x.clamp(clamp[0], clamp[1])
                x.grad.data.zero_()

                grad_x_sum = torch.nan_to_num(grad_x_sum)
                x_dot_grad_x_sum = torch.nan_to_num(x_dot_grad_x_sum)
                for i in range(n_samples):
                    x[i] = torch.nan_to_num(x[i], torch.randn(1).item())

            x.requires_grad = True
        x.requires_grad = False
        if store:
            return x, all_x, timer
        else:
            return x


class CoinLAWGD:
    """
    Coin Laplacian Adjusted Wasserstein Gradient Descent and Laplacian Adjusted Wasserstein Gradient Descent
    Adapted from https://github.com/twmaunu/LAWGD
    """
    def __init__(self):
        pass

    @staticmethod
    def x_grid_func(x_min=-10, x_max=10, n_grid=128):
        return np.linspace(x_min, x_max, n_grid)

    @staticmethod
    def x_grid_func_2d(x_min=-10, x_max=10, y_min=-10, y_max=10, n_grid_x=32, n_grid_y=32):
        n_grid = n_grid_x * n_grid_y
        xx = np.linspace(x_min, x_max, n_grid_x)
        yy = np.linspace(y_min, y_max, n_grid_y)
        dx = (x_max - x_min) / (n_grid_x - 1)
        dy = (y_max - y_min) / (n_grid_y - 1)
        m = dict(zip(product(range(n_grid_x), range(n_grid_y)), range(n_grid)))
        xx2 = np.outer(xx, np.ones(n_grid_x, ))
        yy2 = np.outer(np.ones(n_grid_y, ), yy)
        xx2 = xx2.flatten()
        yy2 = yy2.flatten()
        xx_yy = np.column_stack((xx2, yy2))
        return m, xx, dx, yy, dy, xx_yy

    @staticmethod
    def diff_one(m):
        d = np.zeros((m+1, m))
        d[:m, :] = np.eye(m)
        d[1:, :] -= np.diag(np.ones(m), 0)
        return d

    def lap_two_d(self, n_grid_x, n_grid_y):
        d_x = self.diff_one(n_grid_x)
        d_y = self.diff_one(n_grid_y)
        a_x = d_x.T.dot(d_x)
        a_y = d_y.T.dot(d_y)
        a_x[0, :] = 0
        a_x[-1, :] = 0
        a_y[0, :] = 0
        a_y[-1, :] = 0
        n_grid_x_id = np.eye(n_grid_x)
        n_grid_y_id = np.eye(n_grid_y)
        n_grid_x_id[0, 0] = 0
        n_grid_x_id[-1, -1] = 0
        n_grid_y_id[0, 0] = 0
        n_grid_y_id[-1, -1] = 0
        return sparse.kron(a_y, n_grid_x_id) + sparse.kron(n_grid_y_id, a_x)

    def lawgd_1d_gaussian_kernel(self, x_min=-10, x_max=10, n_grid=128, n_eig=150):
        """
        1D LAWGD gaussian kernel

        Inputs:
            x_min: minimum value of x
            x_max: maximum value of x
            n_grid: the number of grid points
            n_eig: the number of eigen-values

        Outputs:
            kx: the kernel (np array)
        """

        x_grid = self.x_grid_func(x_min, x_max, n_grid)
        psi = np.zeros((n_grid, n_eig + 1))

        for i in range(n_eig+1):
            c = np.zeros(i + 1, )
            c[i] = 1
            psi[:, i] = (hermeval(x_grid, c) / math.factorial(i))

        psi_prime = np.zeros((n_grid, n_eig+1))
        for i in range(n_eig+1):
            if i == 0:
                c = np.zeros(i + 1, )
                c[i] = 1
                psi_prime[:, i] = 2 * i
            else:
                c = np.zeros(i + 1, )
                c[i] = 1
                c2 = np.zeros(i + 1, )
                c2[i - 1] = 1
                psi_prime[:, i] = 2 * hermeval(x_grid, c2) / math.factorial(i)

        # remove first (0) eigenfunction
        psi = psi[:, 1:]
        psi_prime = psi_prime[:, 1:]

        return psi_prime.dot(np.diag(1 / (np.arange(n_eig)+1))).dot(psi.T)

    def lawgd_1d_mixture_gaussian_kernel(self, x_min=-10, x_max=10, n_grid=128, mixture_model=None):
        """
        1D LAWGD mixture gaussian kernel

        Inputs:
            x_min: minimum value of x grid
            x_max: maximum value of x grid
            n_grid: the number of grid points
            mixture_model: target mixture model

        Outputs:
            kx: the kernel (np array)
        """

        x_grid = self.x_grid_func(x_min, x_max, n_grid).reshape((n_grid, 1))
        dx = (x_max-x_min)/(n_grid-1)

        neg_lap = np.zeros([n_grid, n_grid])
        for i in range(n_grid):
            if i > 0:
                neg_lap[i-1, i] = -1/(dx**2)
            neg_lap[i, i] = 2/(dx**2)
            if i+1 < n_grid:
                neg_lap[i+1, i] = -1/(dx**2)

        potential_op = np.diag(mixture_model.v_s(x_grid))
        l = neg_lap + potential_op

        e, psi = np.linalg.eigh(l)

        psi = psi * np.outer(np.exp(-mixture_model.ln_prob(x_grid) / 2), np.ones(n_grid, ))
        psi_prime = (psi[1:, ] - psi[:-1, ]) / dx

        e_inv = e
        e_inv = 1 / e_inv

        return psi_prime.dot(np.diag(e_inv)).dot(psi[:-1, :].T)

    def lawgd_1d(self, x0, step=2, n_iter=5000, kernel="gaussian", n_grid=128, n_eig=150, x_min=-6, x_max=6,
                 every_iter=1, model=None):
        """
        LAWGD on a 1d grid

        Inputs:
            x0: initial samples
            step: stepsize
            n_iter: number of iterations
            kernel: which kernel to use
            n_grid: number of grid values
            n_eig: number of eigenvalues
            x_min, x_max: grid min and max
            every_iter: how often to append particle positions
            model: target distribution

        Outputs:
            x: particle positions
        """

        x_grid = self.x_grid_func(x_min, x_max, n_grid)
        x_grid = x_grid.reshape((x_grid.size, 1))

        if kernel == "gaussian":
            kx = self.lawgd_1d_gaussian_kernel(x_min, x_max, n_grid, n_eig)

        if kernel == "mixture_gaussian":
            kx = self.lawgd_1d_mixture_gaussian_kernel(x_min, x_max, n_grid, model)
            x_grid = x_grid[:-1]

        x = x0.copy()
        x = x.reshape((x.size, 1))
        N = x0.size

        if every_iter:
            sample_seq = list()
            sample_seq.append(x.copy())
            counter = 1

        for i in range(n_iter):
            grad = np.zeros((np.size(x), 1))
            # snap x values to the grid
            idx = np.argmin(np.abs(x - x_grid.T), axis=1)
            # calculate gradient values
            for j in range(N):
                grad[j] = np.sum(kx[idx[j], idx]) / N
            x -= step * grad / N
            x[x < x_min] = x_min
            x[x > x_max] = x_max
            if every_iter:
                if (i % every_iter) == 0:
                    sample_seq.append(x.copy())
                    counter += 1
        if every_iter:
            return x, sample_seq
        else:
            return x

    def coin_lawgd_1d(self, x0, n_iter=5000, kernel="gaussian", n_grid=128, n_eig=150, x_min=-6, x_max=6,
                      every_iter=1, model=None, L=0.01):
        """
        Coin LAWGD on a 1d grid

        Inputs:
            x0: initial samples
            n_iter: number of iterations
            kernel: which kernel to use
            n_grid: number of grid values
            n_eig: number of eigenvalues
            x_min, x_max: grid min and max
            every_iter: how often to append particle positions
            model: target distribution
            L: initial value of L (should be ~0)

        Outputs:
            x: particle positions
        """


        x_grid = self.x_grid_func(x_min, x_max, n_grid)
        x_grid = x_grid.reshape((x_grid.size, 1))

        if kernel == "gaussian":
            kx = self.lawgd_1d_gaussian_kernel(x_min, x_max, n_grid, n_eig)

        if kernel == "mixture_gaussian":
            kx = self.lawgd_1d_mixture_gaussian_kernel(x_min, x_max, n_grid, model)
            x_grid = x_grid[:-1]

        x = x0.copy()
        x = x.reshape((x.size, 1))
        N = x0.size

        if every_iter:
            sample_seq = list()
            sample_seq.append(x.copy())

        counter = 1
        grad_sum = 0
        abs_grad_sum = 0
        reward = 0
        x_dot_grad_sum = np.zeros(N)

        for i in range(n_iter):
            grad = np.zeros((np.size(x), 1))
            # snap x values to the grid
            idx = np.argmin(np.abs(x - x_grid.T), axis=1)

            # gradient
            for j in range(N):
                grad[j] = np.sum(kx[idx[j], idx]) / N

            # |gradient|
            abs_grad = abs(grad)

            # constant
            L = np.maximum(abs_grad, L)

            # sum of gradients
            grad_sum += (-grad)
            abs_grad_sum += abs_grad
            abs_grad_sum_max = np.amax(abs_grad_sum, 1)

            # 'reward'
            reward = np.maximum(reward + np.multiply(x - x0, (-grad)), 0)

            # x dot gradient
            x_dot_grad = np.einsum('ij,ji->i', x, -grad.T)

            # sum of x dot gradient
            x_dot_grad_sum += x_dot_grad

            # x update
            x = x0 + grad_sum / (L * (abs_grad_sum + L)) * (L + reward)

            # clamp
            x[x < x_min] = x_min
            x[x > x_max] = x_max

            # update all x
            if every_iter:
                if (i % every_iter) == 0:
                    sample_seq.append(x.copy())
                    counter += 1

        # output
        if every_iter:
            return x, sample_seq
        else:
            return x


class CoinSGLD:
    """
    Stochastic Gradient Langevin Dynamics.
    """
    def __init__(self, batch=True):
        self.batch = batch

    def sgld_update(self, theta0, score, dt, n_iter):

        # check input
        if theta0 is None or score is None:
            raise ValueError('theta0 or ln_prob_grad cannot be None')

        # initial theta
        theta = np.copy(theta0)

        # all theta
        all_theta = list()
        all_theta.append(np.copy(theta))

        # noise
        w = np.random.normal(loc=0.0, scale=np.sqrt(2*dt), size=(n_iter,) + theta.shape)

        for t in range(n_iter):

            # grad log density
            if self.batch:
                grad_theta = score(theta)
            else:
                grad_theta = np.zeros_like(theta)
                for k in range(theta.shape[0]):
                    grad_theta[k, :] = score(theta[k, :])

            theta = theta + dt * grad_theta + w[t, :, :]

            all_theta.append(np.copy(theta))

        return all_theta

    def coin_update(self, theta0, score, n_iter):

        # check input
        if theta0 is None or score is None:
            raise ValueError('theta0 or ln_prob_grad cannot be None')

        # initial theta
        theta = np.copy(theta0)
        theta0 = np.copy(theta0)

        # all theta
        all_theta = list()
        all_theta.append(theta0)

        # initialise other vars
        L = 1e-10
        grad_theta_sum = 0
        reward = 0
        abs_grad_theta_sum = 0

        for t in range(n_iter):

            # calculate grad log density
            if self.batch:
                ln_p_grad = score(theta)
            else:
                ln_p_grad = np.zeros_like(theta)
                for k in range(theta.shape[0]):
                    ln_p_grad[k, :] = score(theta[k, :])

            # gradient
            grad_theta = ln_p_grad

            # |gradient|
            abs_grad_theta = abs(grad_theta)

            # constant
            L = np.maximum(abs_grad_theta, L)

            # sum of gradients
            grad_theta_sum += grad_theta
            abs_grad_theta_sum += abs_grad_theta

            # 'reward'
            reward = np.maximum(reward + np.multiply(theta - theta0, grad_theta), 0)

            # theta update
            theta_tmp = theta0 + grad_theta_sum / (L * (abs_grad_theta_sum + L)) * (L + reward)

            # effective lr
            eps = (theta_tmp - theta)/(ln_p_grad+1e-16)

            # add noise
            theta = theta + eps * ln_p_grad + np.sqrt(2*abs(eps)) * np.random.normal(0, 1, theta.shape)

            all_theta.append(np.copy(theta))

        return all_theta
