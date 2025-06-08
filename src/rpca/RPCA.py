"""
扩展RPAC
"""
import math

import numpy as np
from choosvd import choosed


def RPCA(D, _lambda=None, tol=None, maxIter=None):
    m, n = D.shape
    D = np.array(D)
    Y = D
    if _lambda is None:
        _lambda = 1 / math.sqrt(max(m, n))
    if tol is None or tol == -1:
        tol = 1e-7
    if maxIter is None or maxIter == -1:
        maxIter = 1000

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
    r = 1.5
    d_norm = np.linalg.norm(D, "fro")  # 771.6923

    iter = 0
    total_svd = 0
    converged = False
    stopCon = 1
    sv = 10

    while converged is False:
        iter = iter + 1
        temp_T = D - A1 + (1 / mu) * Y  # 1000x500
        EE = temp_T - _lambda / mu
        EE[EE < 0] = 0
        temp = temp_T + _lambda / mu
        temp[temp > 0] = 0
        E1 = EE + temp
        if choosed(n, sv) == 1:
            # 输入参数D,E1,(1/MU)*Y全部正确 该函数计算最大的K个特征值
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
        # TODO 把diagS转换为对角向量
        A1 = np.dot(
            np.dot(U[:, 0:lenS], np.diag((diagS[0:lenS] - 1 / mu))), V[:, 0:lenS].T
        )
        total_svd = total_svd + 1
        Z = D - A1 - E1
        Y = Y + mu * Z
        mu = min(mu * r, mu_bar)
        stopCon = np.linalg.norm(Z, "fro") / d_norm

        if stopCon < tol:
            converged = True
        if ~converged and iter > maxIter:
            print("Reach maximum iterations")
            converged = True
        print("iter", iter, "\n")
    return A1, E1, iter - 1, stopCon
