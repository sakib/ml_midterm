#!/ilab/users/sfj19/python/Python-2.7.11/python
from random import randint
import numpy as np
from math import log
from scipy import io
from copy import deepcopy
from collections import defaultdict
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.stats import norm, uniform

#RW = {'log': 'w', 'q0': 'r', 'q1': 'r'}
RW = {'log': 'r', 'q0': 'r', 'q1': 'r'}

# Read data
dataset = io.loadmat('em_problem.mat')['y'][0].tolist()

# Initialize matrix
logs = np.zeros((40, 40))

# summation of log likelihoods. aka l(\theta)
def l(th):
    s = 0.
    for x in dataset: s += log(p(x,th))
    print('l. {}: {}'.format(th, s))
    return s

# 1st derivative of l(\theta). aka vector \nabla l
def dl(th):
    s = [0., 0.]
    for x in dataset:
        dpx = dp(x, th)
        s[0] += dpx[0]/p(x, th)
        s[1] += dpx[1]/p(x, th)
    #print('dl1. {}: {}'.format(th, s))
    return s

# 2nd derivative of l(\theta). aka matrix \nabla^2 l
def dl2(th):
    s = np.zeros((2, 2))
    for x in dataset:
        px, dpx, dp2x = p(x, th), dp(x, th), dp2(x, th)
        s[0][0] += -(dpx[0]/px)**2 + dp2x.item((0,0))/px
        s[0][1] += -dpx[0] * dpx[1] / px**2
        s[1][0] += -dpx[1] * dpx[0] / px**2
        s[1][1] += -(dpx[1]/px)**2 + dp2x.item((1,1))/px
    #print('dl2. {}: {}'.format(th, s))
    return s

# mixture of gaussians model. aka p(x|\theta)
def p(x, th):
    c1 = 0.9 * 0.7 * norm(th[0], 2).pdf(x)
    c2 = 0.9 * 0.3 * norm(th[1], 2).pdf(x)
    c3 = 0.1 * uniform(-20, 40).pdf(x)
    return c1 + c2 + c3

# 1st derivative of p(x|\theta). aka vector \nabla p
def dp(x, th):
    c1 = 0.9 * 0.7 * 0.5 * (x-th[0]) * norm(th[0], 2).pdf(x)
    c2 = 0.9 * 0.3 * 0.5 * (x-th[1]) * norm(th[1], 2).pdf(x)
    return [c1, c2]

# 2nd derivative of p(x|\theta). aka matrix \nabla^2 p
def dp2(x, th):
    c2, c3 = 0, 0
    c1 = 0.9 * 0.7 * 0.5 * (0.5*((x-th[0])**2) - 1) * norm(th[0], 2).pdf(x)
    c4 = 0.9 * 0.3 * 0.5 * (0.5*((x-th[1])**2) - 1) * norm(th[1], 2).pdf(x)
    return np.matrix([[c1, c2], [c3, c4]])

# posterior distribution. aka p(z|x,theta)
def q(z, x, th):
    if z == 0: return 0.9 * 0.7 * norm(th[0], 2).pdf(x) / p(x, th)
    elif z == 1: return 0.9 * 0.3 * norm(th[1], 2).pdf(x) / p(x, th)
    else: return 0.1 * uniform(-20, 40).pdf(x)

# auxiliary function. aka Q(\theta, \theta^t)
def Q(th, th_t):
    s = 0
    for x in dataset:
        for z in range(3):
            ratio = q(z, x, th) * p(x, th) / q(z, x, th_t)
            s += q(z, x, th_t) * log(ratio, 2)
    #print('{}, {}: Q = {}'.format(th, th_t, s))
    return s

# for some reason, matplotlib is plotting points as the inverse of their actual value
# as in, [-5, 5]'s value is ~ -300, but it's plotted as -650. so I invert for plotting purposes
def c(val, logs):
    return logs.max()+logs.min()-val

# Map log_likelihoods
with open('one/logs.txt', 'r+') as lf:
    if RW['log'] == 'w':
        for mu_1 in range(-20, 21, 1):
            for mu_2 in range(-20, 21, 1):
                logs[mu_1][mu_2] = l([mu_1, mu_2])
                lf.write('{}\n'.format(str(logs[mu_1][mu_2])))
    elif RW['log'] == 'r':
        values = [float(line.strip()) for line in lf.readlines()]
        for mu_1 in range(-20, 21, 1):
            for mu_2 in range(-20, 21, 1):
                logs[mu_1][mu_2] = float(values[(mu_1+20)*41 + (mu_2+20)])

# Display graph
nmap = mpl.colors.Normalize(vmin=logs.min(), vmax=logs.max())
cmap = mpl.cm.get_cmap('jet')

plt.imshow(c(logs, logs), cmap=cmap, extent=[-20, 20, -20, 20], interpolation='none')
plt.xlabel(r'$\mu_1$')
plt.ylabel(r'$\mu_2$')
plt.colorbar()

plt.title(r'Log likelihoods of Dataset $\mathbb{D}$')
plt.show() #1a.png

# Initialize params
thetas = [[-0.1, 0.1]]

# EM algorithm
for t in range(11):
    thetas.append([])
    for z in range(2):
        q_l = [q(z, x, thetas[-2]) for x in dataset]
        numer = sum([q_l[i]*dataset[i] for i in range(len(dataset))])
        denom = sum(q_l)
        thetas[-1].append(numer/denom)

plt.xlabel(r'$\mu_1$')
plt.ylabel(r'$\mu_2$')
plt.imshow(c(logs, logs), cmap=cmap, extent=[-20, 20, -20, 20], interpolation='none')
plt.plot([th[0] for th in thetas], [th[1] for th in thetas], 'xb-', color='black')
plt.title(r'Log likelihoods of Dataset $\mathbb{D}$ with EM steps')
plt.colorbar()
plt.show() #1c.png

# Auxiliary Function Calculations
Qs = []
with open('one/aux0.txt', 'r+') as qf0:
    if RW['q0'] == 'w':
        for theta in thetas: # vs. theta_0
            Qs.append(Q(theta, thetas[0]))
            qf0.write('{}\n'.format(str(Qs[-1])))
    elif RW['q0'] == 'r':
        Qs = [float(line.strip()) for line in qf0.readlines()]

for i in range(len(Qs)):
    plt.plot(thetas[i][0], thetas[i][1], '*', color=cmap(nmap(c(Qs[i], logs))), markersize=10)

plt.xlabel(r'$\mu_1$')
plt.ylabel(r'$\mu_2$')
plt.imshow(c(logs, logs), cmap=cmap, extent=[-20, 20, -20, 20], interpolation='none')
plt.title(r'Log likelihoods of Dataset $\mathbb{D}$ with EM steps, $Q(\theta, \theta^0)$')
plt.colorbar()
plt.show() #1e0.png

Qs = []
with open('one/aux1.txt', 'r+') as qf1:
    if RW['q1'] == 'w':
        for theta in thetas: # vs. theta_1
            Qs.append(Q(theta, thetas[1]))
            qf1.write('{}\n'.format(str(Qs[-1])))
    elif RW['q1'] == 'r':
        Qs = [float(line.strip()) for line in qf1.readlines()]

for i in range(len(Qs)):
    plt.plot(thetas[i][0], thetas[i][1], '*', color=cmap(nmap(c(Qs[i], logs))), markersize=10)

plt.xlabel(r'$\mu_1$')
plt.ylabel(r'$\mu_2$')
plt.imshow(c(logs, logs), cmap=cmap, extent=[-20, 20, -20, 20], interpolation='none')
plt.title(r'Log likelihoods of Dataset $\mathbb{D}$ with EM steps, $Q(\theta, \theta^1)$')
plt.colorbar()
plt.show() #1e1.png

# Gradient Ascent method
for alpha in [0.10, 0.05, 0.01]:
    plt.imshow(c(logs, logs), cmap=cmap, extent=[-20, 20, -20, 20])
    plt.xlabel(r'$\mu_1$')
    plt.ylabel(r'$\mu_2$')
    thetas = [[-0.1, 0.1]]
    for t in range(11):
        thetas.append([0., 0.])
        gdl = dl(thetas[-2])
        for i in range(2):
            thetas[-1][i] = thetas[-2][i] + alpha*gdl[i]
    plt.title(r'Gradient Ascent Method with $\alpha = {}$'.format(alpha))
    plt.plot([th[0] for th in thetas], [th[1] for th in thetas], '*b-', color='black')
    plt.colorbar()
    plt.show()

# Newton's Ascent method
plt.imshow(c(logs, logs), cmap=cmap, extent=[-20, 20, -20, 20])
plt.xlabel(r'$\mu_1$')
plt.ylabel(r'$\mu_2$')
plt.colorbar()
thetas = [np.array([-0.1, 0.1])]
for t in range(11):
    thetas.append(np.array([0., 0.]))
    gdl = dl(thetas[-2])
    gdl2 = dl2(thetas[-2])
    change = inv(gdl2).dot(gdl)
    for i in range(2):
        thetas[-1][i] = thetas[-2][i] - change[i]
plt.title(r'Newton Ascent Method')
plt.plot([th[0] for th in thetas], [th[1] for th in thetas], '*b-', color='black')
plt.show()


plt.imshow(c(logs, logs), cmap=cmap, extent=[-20, 20, -20, 20])
alpha = 1.
s_vals = defaultdict()
thetas = [np.array([-0.1, 0.1])]
a, b, c, d = [randint(-20, 20) for k in range(4)]
s_vals[0] = [np.array([a, b])]
s_vals[1] = [np.array([c, d])]
plt.xlabel(r'$\mu_1$')
plt.ylabel(r'$\mu_2$')
print a,b,c,d

for x in dataset:
    thetas.append(np.array([0., 0.]))
    for z in range(2):
        alpha /= 1.1
        posterior = q(z, x, thetas[-2]) # q(z|x_{n+1},\theta^n)
        s_vals[z].append(s_vals[z][-1] + alpha*(np.array([posterior, posterior*x]) - s_vals[z][-1]))
        thetas[-1][z] = s_vals[z][-1][1]/s_vals[z][-1][0] # theta_z^{n+1}

plt.title(r'Online EM algorithm over Dataset $\mathbb{D}$ with Model M1')
plt.plot([th[0] for th in thetas], [th[1] for th in thetas], '*b-', color='black')
plt.show()

