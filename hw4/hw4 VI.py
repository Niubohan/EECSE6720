import numpy as np
from scipy.special import digamma
from scipy.misc import comb
from scipy.special import betaln, gammaln
import matplotlib.pyplot as plt

x = []
with open('x.csv') as file:
    lines = file.readlines()
    for line in lines:
        x.append(int(line.strip()))
    n = len(x)
    x = np.array(x).reshape([n, 1])


def update_phi(x, a, b, alpha, K):
    t1 = digamma(a)
    t2 = digamma(b)
    t3 = digamma(a + b)
    t4 = digamma(alpha)
    t5 = digamma(np.sum(alpha))
    phi = np.zeros((n, K))
    for i in range(K):
        phi[:, i] = np.exp(x[:, 0] * (t1[i] - t3[i]) + (20 - x[:, 0]) * (t2[i] - t3[i]) + t4[i] - t5)
    temp = np.sum(phi, axis=1)
    phi /= temp.reshape([n,1])
    return phi


def update_nj(phi):
    return np.sum(phi, axis=0)


def update_q_pi(alpha, nj):
    return alpha + nj.reshape((len(nj), 1))


def update_q_theta(phi, a, b):
    a_p = np.sum(phi * x, axis=0).reshape((len(a), 1)) + a
    b_p = np.sum(phi * (20 - x), axis=0).reshape((len(a), 1)) + b
    return a_p, b_p


def L1(x, a_p, b_p, alpha, phi, K):
    sum_j = np.zeros((n,1))
    t1 = digamma(alpha) - digamma(np.sum(alpha))
    t2 = digamma(a_p + b_p)
    t3 = np.log(comb(20, x))
    for j in range(K):
        sum_j += phi[:, j].reshape((n, 1)) * (t3 + x * (digamma(a_p) - t2)[j] + (20 - x) * (digamma(b_p) - t2)[j] + t1[j])
    return np.sum(sum_j)


def L2(a, b, a_p, b_p):
    t = digamma(a_p + b_p)
    return np.sum((a - 1) * (digamma(a_p) - t) + (b - 1) * (digamma(b_p) - t) - betaln(a, b))


def L3(alpha, alpha_p, k):
    sum_alpha = sum(alpha)
    t1 = sum(map(lambda i: gammaln(alpha[i]), range(k)))
    t2 = gammaln(sum_alpha)
    t3 = (k - sum_alpha) * digamma(np.sum(alpha_p))
    t4 = sum(map(lambda i: (alpha[i] - 1) * digamma(alpha_p[i]), range(k)))
    return (t1 - t2 - t3 - t4)[0]


def L4(alpha, k):
    sum_alpha = sum(alpha)
    t1 = sum(map(lambda i: gammaln(alpha[i]), range(k)))
    t2 = gammaln(sum_alpha)
    t3 = (k - sum_alpha) * digamma(sum_alpha)
    t4 = sum(map(lambda i: (alpha[i] - 1) * digamma(alpha[i]), range(k)))
    return (t1 - t2 - t3 - t4)[0]


def L5(a_p, b_p):
    t1 = digamma(a_p + b_p)
    t2 = (a_p - 1) * (digamma(a_p) - t1) + (b_p - 1) * (digamma(b_p) - t1) - betaln(a_p, b_p)
    return np.sum(t2)


def L6(phi, k):
    return np.sum(phi * np.log(phi))


def VI_BMM(K):
    alpha = np.random.rand(K, 1).reshape((K, 1))
    a = b = np.ones((K, 1)) * 0.5
    phi = update_phi(x, a, b, alpha, K)
    cluster = []
    loss = []
    for i in range(1000):
        nj = update_nj(phi)
        alpha_p = update_q_pi(alpha, nj)
        a_p, b_p = update_q_theta(phi, a, b)
        l1 = L1(x, a_p, b_p, alpha_p, phi, K)
        l2 = L2(a, b, a_p, b_p)
        l3 = L3(alpha, alpha_p, K)
        l4 = L4(alpha_p, K)
        l5 = L5(a_p, b_p)
        l6 = L6(phi, K)
        phi = update_phi(x, a_p, b_p, alpha_p, K)
        loss.append(l1 + l2 - l3 + l4 - l5 - l6)
    for i in range(n):
        cluster.append(np.argmax(phi[i, :]))
    return loss, cluster


if __name__ == '__main__':
    K = [3, 15, 50]
    loss = []
    cluster = []
    for k in K:
        res_loss, res_clus = VI_BMM(k)
        loss.append(res_loss)
        cluster.append(res_clus)
    plt.plot(range(1000), loss[0], label='K=3')
    plt.plot(range(1000), loss[1], label='K=15')
    plt.plot(range(1000), loss[2], label='K=50')
    plt.legend(loc='center right')
    plt.xlabel('Iterations')
    plt.ylabel('VI Objective')
    plt.title('VI Objective against Iterations')
    plt.show()

    plt.scatter(x, cluster[0], c='g', label='scatter')
    plt.legend()
    plt.title('K=3')
    plt.show()

    plt.scatter(x, cluster[1], c='g', label='scatter')
    plt.legend()
    plt.title('K=15')
    plt.show()

    plt.scatter(x, cluster[2], c='g', label='scatter')
    plt.legend()
    plt.title('K=50')
    plt.show()
