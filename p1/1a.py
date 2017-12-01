#!/ilab/users/sfj19/python/Python-2.7.11/python
import numpy as np
from scipy import io
import matplotlib.pyplot as plt

# Read data
data = io.loadmat('em_problem.mat')['y']

# Log likelihood calculation
def get_log_likelihoods(mu_1, mu_2):
    return 5

# Luls
logs = np.zeros(40, 40))

# Map log_likelihoods
for mu_1 in range(-20, 21, 1):
    for mu_2 in range(-20, 21, 1):
        logs[mu_1][mu_2] = get_log_likelihoods(mu_1, mu_2)

# Display graph
plt.imshow(logs, interpolation='nearest', extent=[-20, 20, -20, 20])
plt.title(r'Log likelihoods of Dataset $\mathbb{D}$')
plt.xlabel(r'$\mu_1$')
plt.ylabel(r'$\mu_2$')
plt.colorbar()
plt.show()

# Expect maximum likelihood at -5, 5
