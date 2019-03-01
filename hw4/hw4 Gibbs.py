import numpy as np
from scipy.stats import binom
from scipy.misc import comb
from scipy.special import beta
import random
import matplotlib.pyplot as plt

x = []
alpha, a, b = 0.75, 0.5, 0.5
cluster = {}
theta = {}
c = []
phi_p = []
cluster_num = []
cluster_len = []
max_val = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}
with open('x.csv') as file:
    lines = file.readlines()
    for line in lines:
        x.append(int(line.strip()))
    n = len(x)
    x = np.array(x).reshape([n, 1])

for i in range(30):
    cluster[i] = []
    theta[i] = np.random.beta(a, b)

for i in range(n):
    ind = random.randint(0, 29)
    c.append(ind)
    cluster[ind].append(i)


def prob_new(x, alpha):
    t1 = alpha / (alpha + n - 1)
    t2 = comb(20, x)
    t3 = beta(a + x, b + 20 - x) / beta(a, b)
    return t1 * t2 * t3


for i in range(n):
    phi_p.append(prob_new(x[i, 0], alpha))

for iter in range(1000):
    print iter
    print len(cluster)
    cluster_len = [len(cluster[i]) for i in cluster]
    cluster_len.sort(reverse=True)
    if len(cluster) < 6:
        for i in range(len(cluster)):
            max_val[i].append(cluster_len[i])
        for i in range(len(cluster), 6):
            max_val[i].append(0)
    else:
        for i in range(6):
            max_val[i].append(cluster_len[i])
    cluster_num.append(len(cluster))
    for i in range(n):
        f = 0
        phi = []
        for clu in cluster:
            if c[i] == clu:
                cluster[clu].remove(i)
            phi.append(binom.pmf(x[i, 0], 20, theta[clu]) * len(cluster[clu]) / (alpha + n - 1))
        phi.append(phi_p[i])
        phi = np.array(phi)
        phi /= np.sum(phi)
        c[i] = int(np.random.choice(len(phi), 1, p=phi))
        try:
            cluster[c[i]].append(i)
        except KeyError:
            cluster[c[i]] = [i]
        if c[i] == len(phi) - 1:
            theta[c[i]] = beta(a + x[i, 0], b + 20 - x[i, 0])
        for key in cluster.keys():
            if len(cluster[key]) == 0:
                f = 1
                del cluster[key]
        if f == 1:
            remain = cluster.keys()
            theta_new = {}
            for j in range(len(remain)):
                theta_new[j] = theta[remain[j]]
            theta = theta_new
            c_new = []
            cluster.clear()
            for m in range(n):
                for j in range(len(remain)):
                    if c[m] == remain[j]:
                        c_new.append(j)
                        try:
                            cluster[j].append(m)
                        except KeyError:
                            cluster[j] = [m]
            c = c_new

        for clu in cluster:
            sum1 = np.sum(x[cluster[clu], :])
            sum2 = np.sum(20 - x[cluster[clu], :])
            theta[clu] = np.random.beta(a + sum1, b + sum2)

plt.plot(range(1000), cluster_num)
plt.xlabel('Iterations')
plt.ylabel('Number of Clusters')
plt.title('Number of Clusters per Iterations')
plt.show()

plt.figure(figsize=(10,10))
for i in max_val:
    plt.plot(range(1000), max_val[i], label=str(i)+'th Largest')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Iterations')
plt.ylabel('Points in Cluster')
plt.title('Points in Cluster per Iterations')
plt.show()
