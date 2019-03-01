import math
import numpy as np
import matplotlib.pyplot as plt


def gamma(i):
    sum = 0
    while i > 1:
        sum += math.log(i)
        i -= 1
    return sum

raw_x = []
test_x = []
spm_y = []
nspm_y = []
res_y = []
lam1 = []
lam0 = []
spmSum_x = [0 for i in range(0,54)]
nspmSum_x = [0 for a in range(0,54)]

with open("X_train.csv") as file:
    lines = file.readlines()
    temp = []
    for line in lines:
        raw_x.append(map(int, line.strip().split(',')))

with open("X_test.csv") as file:
    lines = file.readlines()
    temp = []
    for line in lines:
        test_x.append(map(int, line.strip().split(',')))

with open("label_train.csv") as file:
    lines = file.readlines()
    cnt = 0
    for line in lines:
        if int(line):
            spm_y.append(cnt)
        else:
            nspm_y.append(cnt)
        cnt += 1

with open("label_test.csv") as file:
    lines = file.readlines()
    for line in lines:
        res_y.append(int(line.strip()))

N = len(spm_y) + len(nspm_y)
pyIs1 = (1.0 + len(spm_y)) / (2.0 + N)
pyIs0 = (1.0 + len(nspm_y)) / (2.0 + N)

for x in spm_y:
    for i in range(0,54):
        spmSum_x[i] += raw_x[x][i]

for x in nspm_y:
    for i in range(0, 54):
        nspmSum_x[i] += raw_x[x][i]

nspm_right = 0
spm_right = 0
cnt = 0
wrong_x = [test_x[240]]

for i in range(0, 54):
    lam1.append((spmSum_x[i] + 1) / (N + 1))
    lam0.append((nspmSum_x[i] + 1) / (N + 1))


for x in test_x:
    p0,p1 = 0.0, 0.0
    for i in range(0,54):
        p0 += gamma(x[i] + nspmSum_x[i] + 1) - gamma(nspmSum_x[i] + 1) - gamma(x[i] + 1) + \
              (nspmSum_x[i]) * (math.log((N + 1)) - math.log(N + 2)) - x[i] * math.log(N + 2)
        p1 += gamma(x[i] + spmSum_x[i] + 1) - gamma(spmSum_x[i] + 1) - gamma(x[i] + 1) + \
              (spmSum_x[i]) * (math.log((N + 1)) - math.log(N + 2)) - x[i] * math.log(N + 2)

    p0 += math.log(pyIs0)
    p1 += math.log(pyIs1)

    res0 = math.exp(p0) / (math.exp(p1) + math.exp(p0))
    res1 = math.exp(p1) / (math.exp(p1) + math.exp(p0))

    print cnt
    print res0
    print res1

    if p0 > p1 and res_y[cnt]:
        print cnt
        print math.exp(p0) / (math.exp(p1) + math.exp(p0))
        print math.exp(p1) / (math.exp(p1) + math.exp(p0))
    elif p1 > p0 and not res_y[cnt]:
        print cnt
        print math.exp(p0) / (math.exp(p1) + math.exp(p0))
        print math.exp(p1) / (math.exp(p1) + math.exp(p0))
    cnt += 1


x = np.arange(54)
y1 = test_x[431]
fig, ax = plt.subplots()
plt.bar(x, y1, width = 0.35, label='No.432 email')
plt.bar(x + 0.35, lam1, width = 0.35, label='lambda 1')
plt.bar(x - 0.35, lam0, width = 0.35, label='lambda 0')
plt.xticks(x, ('make', 'address', 'all', '3d', 'our', 'over', 'remove', 'internet', 'order', 'mail', 'receive', 'will',
               'people', 'report', 'addresses', 'free', 'business', 'email', 'you', 'credit', 'your', 'font', '000',
               'money', 'hp', 'hpl', 'george', '650', 'lab', 'labs', 'telnet', '857', 'data', '415', '85', 'technology',
               '1999', 'parts', 'pm', 'direct', 'cs', 'meeting', 'original', 'project', 're', 'edu', 'table',
               'conference', ';', '(', '[', '!', '$', '#'), fontsize=5)
plt.legend(loc="upper left")
plt.show()

