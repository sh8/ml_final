##########################################################
# Implementation of Accelerated Proximal Gradient Method #
##########################################################

import pickle

import numpy as np
import matplotlib.pyplot as plt

LAMBDAS = [2., 4., 6.]
W_OPTS = np.array([[0.82, 1.09], [0.64, 0.18], [0.33, 0]])
w = np.array([3., -1.])
w_old = None
v = w
mu = np.array([1., 2.])
A = np.array([[3., 0.5], [0.5, 1.]])
lr = 1 / np.max(np.linalg.eig(2*A)[0])


def fn(weight):
    return np.dot(np.dot(np.transpose(weight-mu), A), weight-mu)


if __name__ == '__main__':
    lines = []
    for k, LAMBDA in enumerate(LAMBDAS):
        norms = []
        r = range(1, 101)

        for epoch in r:
            w_old = np.copy(w)
            diff = np.dot(A, v - mu)
            tilde_mu = v - lr * diff
            tilde_q = (LAMBDA * lr / 2) * np.ones(2)

            for i in range(0, 2):
                if tilde_mu[i] > tilde_q[i]:
                    w[i] = tilde_mu[i] - tilde_q[i]
                elif tilde_mu[i] < -tilde_q[i]:
                    w[i] = tilde_mu[i] + tilde_q[i]
                else:
                    w[i] = 0.
            qt = (epoch - 1) / (epoch + 2)
            v = w + qt * (w - w_old)
            norm = np.linalg.norm(v - W_OPTS[k])
            norms.append(norm)

        line, = plt.plot(list(r), norms)
        with open(f'NORM_APG_{LAMBDA}.p', 'wb') as f:
            pickle.dump(norms, f)
        lines.append(line)

    plt.legend(lines, ['λ=2', 'λ=4', 'λ=6'])
    plt.yscale('log')
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('Norm', fontsize=18)
    plt.subplots_adjust(hspace=0.4)
    plt.savefig('problem2_2.png')
