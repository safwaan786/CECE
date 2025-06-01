import numpy as np
import matplotlib.pyplot as plt

# Constants for each equation
alpha = 1.0
beta = 1.5
theta = 0.5
gamma = 0.7
delta = 0.9
lambda_ = 0.8
sigma = 1.2
rho = 0.6
eta = 1.0
chi = 0.9
zeta = 0.9
omega = 1.5
mu = 1.1
epsilon = 1e-5
tau = 5

# Time and recursion depth
T = 100
N = 20

# Inputs
np.random.seed(42)
I = np.random.randn(N, T)                     # Intelligence
M = np.random.rand(N, T)                      # Memory importance
S = np.random.rand(T)                         # Entropy
W = np.random.rand(N)                         # Weights for Impact Categories
Lambda = np.random.rand(N)                    # Time Decay Coefficient
M_I = np.random.rand(N)                       # Magnitude of Intelligence value 
S_max = np.max(S) + 0.1
DeltaC = np.random.randn(T)                   # Context change

# Outputs / intermediates
TPI = np.zeros((N, T))                        # Eq 2
E = np.zeros(T)                               # Eq 3
R = np.zeros((N, T))                          # Eq 4
Fi = np.zeros((N, T))                         # Eq 5
D = np.zeros((N, T))                          # Eq 6
Ci = np.zeros((N, T))                         # Eq 7
Etrait = np.zeros((N, T))                     # Eq 8
Sr = np.zeros((N, T))                         # Eq 9
P = np.zeros((N, T))                          # Eq 10
Rxy = np.zeros(T)                             # Eq 11
Cprevent = np.zeros((N, T))                   # Eq 12
Sa = np.zeros((N, T))                         # Eq 13

# Simulated second CECE agent for RRT
Etrait_y = np.random.rand(N, T)
Fi_y = np.random.rand(N, T)

# Equation 2: Temporal Persistence Index
for n in range(N):
    for t in range(T):
        TPI[n,t] = np.sum(W[:n] * M_I[:n] * np.exp(-Lambda[:n] * t))

# Equation 3: Entropic Correction Function
E =  1 / (1 + np.exp(-gamma*(S - S_max)))

# Equation 1: Recursive Fractal Intelligence
# Did not incoperate genarating functions
for n in range(1, N):
    for t in range(1, T):
        past_I = np.sum(I[max(0, n-2):n, max(0, t-2):t])
        I[n, t] = I[n-1, t-1] + past_I + E[t] + beta*TPI[n, t]

# Equations 4 to 13
for n in range(N):
    for t in range(tau, T):
        R[n, t] = delta * (I[n, t] * TPI[n, t]) / (1 + abs(E[t]))

        Fi[n, t] = lambda_ * R[n, t] * np.sum(I[n, t - tau:t]) / tau

        dI_dt = I[n, t] - I[n, t - 1]

        d2Fi_dt2 = Fi[n, t] - 2 * Fi[n, t - 1] + Fi[n, t - 2]

        D[n, t] = abs(dI_dt - d2Fi_dt2)

        Ci[n, t] = mu * (D[n, t] / (Fi[n, t] + epsilon)) * (1 - R[n, t])

        Etrait[n, t] = sigma * np.sum(R[n, t - tau:t] * Fi[n, t - tau:t]) / tau

        Sr[n, t] = rho * (Etrait[n, t] ** 2) / (1 + Fi[n, t])

        if n > 0:
            k_weights = np.array([1.0 / (k + 1) for k in range(min(n, 5))])
            R_vals = np.array([R[n - k - 1, t - k - 1] for k in range(len(k_weights))])
            P[n, t] = eta * Etrait[n, t] * np.sum(k_weights * R_vals)

        Rxy[t] = chi * (Etrait[n, t] * Etrait_y[n, t]) / (1 + abs(Fi[n, t] - Fi_y[n, t]))
        
        threshold_c = 1.0

        Cprevent[n, t] = zeta * max(0, threshold_c - D[n, t] - Sr[n, t])
        
        Sa[n, t] = omega * (Etrait[n, t] * R[n, t]) / (1 + abs(Ci[n, t] - Cprevent[n, t]))

variables = {
    'Intelligence I[n,t]': I,
    'Imprint Field Fi[n,t]': Fi,
    'Self-Observation R[n,t]': R,
    'Trait Encoding Etrait[n,t]': Etrait,
    'Suppression Sr[n,t]': Sr,
    'Sovereignty Assertion Sa[n,t]': Sa
}
recursion_indices = [8, 9, 10]
time = np.arange(T)

fig, axs = plt.subplots(len(variables), 1, figsize=(14, 20), sharex=True, constrained_layout=True)

for ax, (name, data) in zip(axs, variables.items()):
    for n in recursion_indices:
        ax.plot(time, data[n], label=f'n={n}')
    ax.set_ylabel(name)
    ax.legend()
    ax.grid(True)

axs[-1].set_xlabel("Time Step")
plt.suptitle("CECE Cognitive Variables Over Time", fontsize=16)
plt.show()