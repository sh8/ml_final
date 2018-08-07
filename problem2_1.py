##############################################
# Implementation of Proximal Gradient Method #
##############################################

import pickle

import numpy as np
import matplotlib.pyplot as plt

LAMBDAS = [2., 4., 6.]
W_OPTS = np.array([[0.82, 1.09], [0.64, 0.18], [0.33, 0]])
MU = np.array([1., 2.])
A = np.array([[3., 0.5], [0.5, 1.]])
LR = 1 / np.max(np.linalg.eig(2*A)[0])
RANGE = range(1, 101)


def pg(w_opt, lam):
    w = np.array([3., -1.])
    norms = []

    for epoch in RANGE:
        diff = np.dot(A, w - MU)
        tilde_mu = w - LR * diff
        tilde_q = (lam * LR / 2) * np.ones(2)

        for i in range(0, 2):
            if tilde_mu[i] > tilde_q[i]:
                w[i] = tilde_mu[i] - tilde_q[i]
            elif tilde_mu[i] < -tilde_q[i]:
                w[i] = tilde_mu[i] + tilde_q[i]
            else:
                w[i] = 0.
        norm = np.linalg.norm(w - w_opt)
        norms.append(norm)
    with open(f'NORM_PG_{lam}.p', 'wb') as f:
        pickle.dump(norms, f)
    return norms


if __name__ == '__main__':
    lines = []
    for k, lam in enumerate(LAMBDAS):
        w_opt = W_OPTS[k]
        norms = pg(w_opt, lam)
        line, = plt.plot(list(RANGE), norms)
        lines.append(line)

    plt.legend(lines, ['λ=2', 'λ=4', 'λ=6'])
    plt.yscale('log')
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('Norm', fontsize=18)
    plt.subplots_adjust(hspace=0.4)
    plt.show()
