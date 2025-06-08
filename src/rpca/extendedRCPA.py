"""
    扩展RPAC
"""
import numpy as np
from scipy.sparse.linalg import svds

from .choosvd import choosed


def extendedRPCA(D, omega, _lambda, tol, maxIter):
    m, n = D.shape
    # 把pandas转成numpy
    D = np.array(D)
    Y = D

    u, norm_two, v = np.linalg.svd(Y, full_matrices=False)  # 得到特征向量
    norm_two = norm_two[0]  # 得到第1个元素
    # matlab Y(:)，相当于把Y变成nx1的形式
    norm_inf = np.linalg.norm(Y.reshape((-1, 1)), axis=1, keepdims=True) / _lambda
    norm_inf = norm_inf[0]
    # 585.4379 63.2456
    dual_norm = max(norm_two, norm_inf)
    Y = Y / dual_norm

    A1 = np.zeros((m, n))
    E1 = np.zeros((m, n))
    mu = 1.25 / norm_two
    mu_bar = mu * 1e7
    r = 1.1
    d_norm = np.linalg.norm(D, "fro")  # 771.6923

    iter = 0
    total_svd = 0
    converged = False
    stopCon = 1
    sv = 10

    while converged is False:
        iter = iter + 1
        temp_T = D - A1 + (1 / mu) * Y  # 1000x500
        E1 = temp_T
        # temp_T - _lambda/mu，小于0为0.
        EE = temp_T - _lambda / mu
        EE[EE < 0] = 0

        temp = temp_T + _lambda / mu
        temp[temp > 0] = 0
        EE = EE + temp

        for x, y in zip(omega[0], omega[1]):
            E1[x, y] = EE[x, y]
        if choosed(n, sv) == 1:
            # 得到sv个元素
            # 输入参数D,E1,(1/MU)*Y全部正确
            U, S, V = np.linalg.svd(D - E1 + (1 / mu) * Y, full_matrices=True)  # 得到特征向量
            U = U[:, :sv]
            V = (V.T)[:, :sv]
            S = S[:sv]
        else:
            U, S, V = np.linalg.svd(D - E1 + (1 / mu) * Y, full_matrices=False)
            V = V.T
            # print('U shape {} S shape {} V shape {}'.format(U.shape, S.shape, V.shape))
        lenS = len(np.where(S > 1 / mu)[0])
        diagS = S
        if lenS < sv:
            sv = min(lenS + 1, n)
        else:
            sv = min(lenS + round(0.05 * n), n)

        A1 = np.dot(
            np.dot(U[:, 0:lenS], np.diag((diagS[0:lenS] - 1 / mu))), V[:, 0:lenS].T
        )

        total_svd = total_svd + 1
        Z = D - A1 - E1
        Y = Y + mu * Z
        mu = min(mu * r, mu_bar)

        stopCon = np.linalg.norm(Z, "fro") / d_norm
        # print('Z {}, mu {}, stopCon {} lenS {} diagS {}'.format(Z, mu, stopCon, lenS, diagS[:lenS]))

        if stopCon < tol:
            converged = True
        if ~converged and iter > maxIter:
            print("Reach maximum iterations")
            converged = True
    # 变成0
    return A1, E1, iter - 1, stopCon
