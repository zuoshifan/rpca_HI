import numpy as np
from scipy import linalg as la


def rpca(M, lmbda=None, mu=0.5, rho=1.1, gamma=1.0e-2, tol=1.0e-3, niter=500, norm_type='21'):
    """A python implementation of the non-convex robust PCA method.

    The method is described in the paper Kang, et al., 2015, Robust PCA via
    Noneconvex Rank Approximation.

    """
    if lmbda is None:
        lmbda = 1.0 / np.sqrt(np.max(M.shape))

    S = np.zeros_like(M)
    Y = np.zeros_like(M)
    sigma = np.zeros(min(M.shape), dtype=M.real.dtype)

    for i in range(niter):
        # update L
        A = M - S - Y/mu
        U, s, VT = la.svd(A, full_matrices=False)
        for j in range(100):
            w = gamma*(1.0 + gamma) / (gamma + sigma)**2
            sigmak = np.maximum(s - w/mu, 0.0)
            if np.sum((sigmak - sigma)**2) < 1.0e-6:
                sigma = sigmak
                break
            sigma = sigmak
        L = np.dot(U*sigma, VT)

        # update S
        Q = M - L - Y/mu
        lmu = lmbda/mu
        if norm_type == '1':
            S = np.maximum(np.abs(Q) - lmu, 0.0) * np.sign(Q)
        elif norm_type == '21':
            for c in range(S.shape[1]):
                # Qc2 = np.sum(Q[:, c]**2)**0.5
                Qc2 = la.norm(Q[:, c])
                if Qc2 > lmu:
                    S[:, c] = (Qc2 - lmu) * Q[:, c] / Qc2
                else:
                    S[:, c] = 0
        else:
            raise ValueError('Unknown norm_type %s' % norm_type)

        # update Y and mu
        Y = Y + mu*(L + S - M)
        mu = rho * mu

        # compute residuals
        res = la.norm(M - S - L, ord='fro') / la.norm(M, ord='fro')
        if res < tol:
            break

    return L, S, res


if __name__ == '__main__':
    from scipy.io import loadmat
    from numpy.linalg import matrix_rank

    data_name = '/home/zuoshifan/programming/matlab/noncvx-PRCA/subject5.mat'
    data = loadmat(data_name)
    # print data
    # print data['X'].shape
    M = data['X']
    L, S, res = rpca(M, lmbda=1.0e-3, norm_type='21')
    print res
    print matrix_rank(L)
    print matrix_rank(S)