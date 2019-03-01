import numpy as np
import numpy.linalg as lg
from scipy.special import gammaln, digamma
import matplotlib.pyplot as plt

a0 = b0 = 10e-16
ft = e0 = f0 = 1
x, y, z, L = [], [], [], []

with open('X_set3.csv') as file:
    lines = file.readlines()
    for line in lines:
        t = []
        for item in line.strip().split(','):
            t.append(float(item))
        x.append(t)
x = np.array(x)

with open('Y_set3.csv') as file:
    lines = file.readlines()
    for line in lines:
        y.append(float(line.strip().split(',')[0]))
y = np.array(y)

with open('Z_set3.csv') as file:
    lines = file.readlines()
    for line in lines:
        z.append(float(line.strip().split(',')[0]))
z = np.array(z)

d = x.shape[1]
N = x.shape[0]
y = y.reshape((1,N))
at = 0.5 + a0 + np.zeros((d,1))
bt = b0 * np.ones((d,1))
et = N / 2.0 + e0
mu = np.zeros((d, 1))
sigma = np.diag(np.ones(d, dtype='float64'))
t = np.array([[2,3],[4,5],[6,7]])
c = np.square(t)
print 1


def update_q_lam(f, x, y, mu, sigma):
    t1 = 0.5 * np.sum(np.square(y.transpose() - np.dot(x, mu)))
    t2 = 0.5 * np.trace(np.dot(np.dot(x, sigma), x.transpose()))
    f_pr = f + t1 + t2
    return f_pr


def update_q_alph(b0, mu, sigma, d):
   mu_sqr = np.dot(mu, mu.transpose())
   sig_mu = np.dot((sigma + mu_sqr) * np.eye(d), np.ones((d, 1)))
   b_pr = 0.5 * sig_mu + b0
   return b_pr


def A(a,b):
    return a / b * np.eye(len(a))


def update_q_w(e, f, A, x, y, d):
    E_lam = e / f
    sigma = lg.inv(E_lam * np.dot(x.transpose(), x) + A)
    yx = np.sum(y.transpose() * x, axis=0).transpose()
    mu = np.dot(sigma, E_lam * yx).reshape((d,1))
    return sigma, mu


def L1(e0, f0, e, f):
    return e0 * np.log(f0) - gammaln(e0) + (e0 - 1) * (digamma(e) - np.log(f)) - f0 * (e / f)


def L2(a0, b0, a, b, d):
    return (d - 1) * (a0 * np.log(b0) - gammaln(a0)) + (a0 - 1) * np.sum(digamma(a) - np.log(b)) - b0 * np.sum(a / b)


def L3(a, b, At, mu, sigma, d):
    return - d / 2 * np.log(2 * np.pi) + 0.5 * np.sum(digamma(a) - np.log(b)) \
           - 0.5 * np.trace((np.dot(np.dot(mu, mu.transpose()) + sigma, At)))


def L4(e, f, x, y, N):
    s = np.sum(np.square(y.transpose() - np.dot(x, mu))) + np.trace(np.dot(np.dot(x, sigma), x.transpose()))
    return - N / 2 * np.log(2 * np.pi) + N / 2 * (digamma(e) - np.log(f)) - 0.5 * e / f * s


def L5(e, f):
    return -e + np.log(f) - gammaln(e) - (1 - e) * digamma(e)


def L6(sigma, N):
    sign, detsig = np.linalg.slogdet(sigma)
    return -0.5 * (sign * detsig)


def L7(a, b):
    return np.sum( - a + np.log(b) - gammaln(a) - (1 - a) * digamma(a))

for i in range(500):
    ft = update_q_lam(f0, x, y, mu, sigma)
    bt = update_q_alph(b0, mu, sigma, d)
    At = A(at, bt)
    sigma, mu = update_q_w(et, ft, At, x, y, d)
    l1 = L1(e0, f0, et, ft)
    l2 = L2(a0, b0, at, bt, d)
    l3 = L3(at, bt, At, mu, sigma, d)
    l4 = L4(et, ft, x, y, N)
    l5 = L5(et, ft)
    l6 = L6(sigma, N)
    l7 = L7(at, bt)
    l = l1 + l2 + l3 + l4 - l5 - l6 - l7
    print l
    L.append(l)

y_hat = np.dot(x, mu).transpose()
plt.plot(z, y_hat[0], 'r', label='y hat')
plt.scatter(z, y[0], c='g', label='scatter')
plt.plot(z, 10 * np.sinc(z), 'b', label='function')
plt.legend()
plt.title('Dataset3')
plt.show()
