import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt

x = []
with open('x.csv') as file:
    lines = file.readlines()
    for line in lines:
        x.append(int(line.strip()))
    n = len(x)
    # x = np.array(x)
    x = np.array(x).reshape([n, 1])


def update_phi(x, pi, theta, phi, K):
    for i in range(K):
        phi[:, i] = binom.pmf(x[:, 0], 20, theta[i]) * pi[i]
    temp = np.sum(phi, axis=1)
    phi /= temp.reshape([n,1])


def update_nj(phi):
    return np.sum(phi, axis=0)


def update_theta(phi, x, nj):
    t = np.sum(phi * x, axis= 0)
    t2 = t / (20 * nj)
    return t2


def update_pi(n, nj):
    return nj / n


def loss(pi, theta, phi, K):
    sum_j = np.zeros([n,1])
    for i in range(K):
        sum_j += phi[:, i].reshape((n, 1)) * (x * np.log(theta[i]) + (20 - x) * np.log(1 - theta[i]) + np.log(pi[i]))
    return np.sum(sum_j)


def margin(x, pi, theta, K):
    sum_j = np.zeros((n,1))
    # sum_j = x * np.log(theta) + (20 - x) * np.log(1 - theta) + np.log(pi)
    '''for i in range(K):
        sum_j += x * np.log(theta[i]) + (20 - x) * np.log(1 - theta[i]) + np.log(pi[i])'''
    for i in range(K):
        sum_j += pi[i] * binom.pmf(x, 20, theta[i])
    return np.sum(np.log(sum_j))


def EM_BMM(K):
    pi = np.ones([K, 1]) / K
    theta = np.random.rand(K,1).reshape((K, 1))
    '''theta = []
    for i in range(K):
        theta.append((i + 1) / (K + 1.0))
    theta = np.array(theta).reshape((K, 1))'''
    phi = np.zeros([n, K])
    margin_value = []
    cluster = []
    for i in range(50):
        update_phi(x, pi, theta, phi, k)
        nj = update_nj(phi)
        theta = update_theta(phi, x, nj)
        pi = update_pi(n, nj)
        margin_value.append(margin(x, pi, theta, k))
    for i in range(n):
        cluster.append(np.argmax(phi[i, :]))
    return margin_value, cluster


if __name__ == '__main__':
    K = [3, 9, 15]
    likelihood = []
    cluster = []
    for k in K:
        res_like, res_clus = EM_BMM(k)
        likelihood.append(res_like)
        cluster.append(res_clus)

    plt.plot(range(50), likelihood[0], label='K=3')
    plt.plot(range(50), likelihood[1], label='K=9')
    plt.plot(range(50), likelihood[2], label='K=15')
    plt.legend(loc='center right')
    plt.xlabel('Iterations')
    plt.ylabel('EM Objective')
    plt.title('EM Objective against Iterations')
    plt.show()

    plt.scatter(x, cluster[0], c='g', label='scatter')
    plt.legend()
    plt.title('K = 3')
    plt.show()

    plt.scatter(x, cluster[1], c='g', label='scatter')
    plt.legend()
    plt.title('K = 9')
    plt.show()

    plt.scatter(x, cluster[2], c='g', label='scatter')
    plt.legend()
    plt.title('K = 15')
    plt.show()

