import numpy as np
from scipy.stats import norm
import numpy.linalg as lg
import matplotlib.pyplot as plt
from scipy.stats import binom

N, M = 0, 0
ratings = {1:[], -1:[]}
test = []
label = []
t = []
x = [i for i in range(20, 101)]

def UV(U, V):
    return np.dot(U, V.transpose())


def Ephi(UV, Rp, Rn):
    cdf = norm.cdf(-UV)
    pdf = norm.pdf(-UV)
    return (UV + pdf / (1 - cdf)) * Rp + (UV + (-pdf / cdf)) * Rn


def update(U, V, E, R):
    for i in range((len(U))):
        numer = np.dot(V.transpose(), E[i].reshape((len(V), 1)))
        Vomg = V.transpose() * R[i]
        denom = lg.inv(np.eye(5) + np.dot(Vomg, Vomg.transpose()))
        U[i] = np.dot(denom, numer)[0]


def bern(UV, Rp, Rn):
    cdf = norm.cdf(UV)
    return np.sum(np.log(cdf) * Rp + np.log(1 - cdf) * Rn)


with open('ratings.csv') as file:
    lines = file.readlines()
    for line in lines:
        [uid, mid, rating] = line.strip().split(',')
        N = max(N, int(uid))
        M = max(M, int(mid))
        if int(rating) == 1:
            ratings[1].append([int(uid), int(mid)])
        else:
            ratings[-1].append([int(uid), int(mid)])
with open('ratings_test.csv') as file:
    lines = file.readlines()
    for line in lines:
        [uid, mid, rating] = line.strip().split(',')
        test.append([int(uid), int(mid)])
        label.append(int(rating))

for a in range(5):
    U = np.random.normal(0, 0.1, (N, 5))
    V = np.random.normal(0, 0.1, (M, 5))
    Rp = np.zeros((N, M))
    Rn = np.zeros((N, M))
    E = np.zeros((N, M))
    hit1 = 0
    hit2 = 0
    tn = []

    for item in ratings[1]:
        Rp[item[0] - 1][item[1] - 1] = 1
    for item in ratings[-1]:
        Rn[item[0] - 1][item[1] - 1] = 1

    R = Rp + Rn
    UtV = UV(U, V)
    for i in range(100):
        E = Ephi(UtV, Rp, Rn)
        update(U, V, E, R)
        UtV = UV(U, V)
        E = Ephi(UtV, Rp, Rn)
        update(V, U, E.transpose(), R.transpose())
        UtV = UV(U, V)
        Bern = bern(UtV, Rp, Rn)
        tn.append(2.5 * (M + N) * np.log(1 / (2 * np.pi)) - 0.5 * (np.sum(U * U) + np.sum(V * V)) + Bern)
        print 2.5 * (M + N) * np.log(1 / (2 * np.pi)) - 0.5 * (np.sum(U * U) + np.sum(V * V)) + Bern
    t.append(tn)

    for i in range(len(test)):
        [uid, mid] = test[i]
        if norm.cdf(UtV[uid - 1][mid - 1]) >= 0.5:
            res = 1
        else: res = -1
        if res == label[i]:
            if res == 1:
                hit1 += 1
            else: hit2 += 1.
    print hit1, hit2
    print (hit1 + hit2) / len(test)

plt.plot(x,t[0][19:],'r')
plt.plot(x,t[1][19:],'g')
plt.plot(x,t[2][19:],'b')
plt.plot(x,t[3][19:],'y')
plt.plot(x,t[4][19:],'c')
plt.show()
