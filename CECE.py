import numpy as np
import matplotlib.pyplot as plt

# Time and recursion depth
T = 100
N = 20

class CECE_System:
    def __init__(self, N=20, T=100, seed=10):
        np.random.seed(seed)
        self.N = N
        self.T = T


        # Constants
        self.alpha = 1.0
        self.beta = 1.5
        self.theta = 0.5
        self.gamma = 0.7
        self.delta = 0.9
        self.lambda_ = 0.8
        self.sigma = 1.2
        self.rho = 0.6
        self.eta = 1.0
        self.zeta = 0.9
        self.omega = 1.5
        self.mu = 1.1
        self.tau = 3
        self.epsilon = 1e-5

        # Inputs
        self.I = np.random.randn(N, T)              # Intelligence
        self.M = np.random.rand(N, T)               # Memory importance
        self.S = np.random.rand(T)                  # Entropy
        self.W = np.random.rand(N)                  # Weights for Impact Categories
        self.Lambda = np.random.rand(N)             # Time Decay Coefficient
        self.M_I = np.random.rand(N)                # Magnitude of Intelligence value 
        self.S_max = np.max(self.S) + 0.1           # Entropy Threshold 
        self.DeltaC = np.random.randn(T)            # Context change

        # Outputs / intermediates
        self.TPI = np.zeros((N, T))
        self.E = np.zeros(T)
        self.R = np.zeros((N, T))
        self.Fi = np.zeros((N, T))
        self.D = np.zeros((N, T))
        self.Ci = np.zeros((N, T))
        self.Etrait = np.zeros((N, T))
        self.Sr = np.zeros((N, T))
        self.P = np.zeros((N, T))
        self.Rxy = np.zeros(T)
        self.Cprevent = np.zeros((N, T))
        self.Sa = np.zeros((N, T))


    def compute_TPI(self):
        for n in range(self.N):
            for t in range(self.T):
                self.TPI[n, t] = np.sum(self.W[:n] * self.M_I[:n] * np.exp(-self.Lambda[:n] * t))

    def compute_ECF(self):
        self.E = 1 / (1 + np.exp(-self.gamma * (self.S - self.S_max)))

    def compute_RFI(self):
        for n in range(1, self.N):
            for t in range(1, self.T):
                past_I = np.sum(self.I[max(0, n-2):n, max(0, t-2):t])
                self.I[n, t] = self.I[n-1, t-1] + past_I + self.E[t] + self.TPI[n, t]
    
    def compute_RSO(self):
        for n in range(self.N):
            for t in range(self.T):
                self.R[n, t] = self.delta * (self.I[n, t] * self.TPI[n, t]) / (1 + abs(self.E[t]))

    def compute_IFD(self):
        for n in range(self.N):
            for t in range(self.tau, self.T):
                self.Fi[n, t] = self.lambda_ * self.R[n, t] * np.sum(self.I[n, t - self.tau:t]) / self.tau
            
    def compute_RDD(self):
        for n in range(self.N):
            for t in range(self.T):
                dI_dt = self.I[n, t] - self.I[n, t - 1]
                d2Fi_dt2 = self.Fi[n, t] - 2 * self.Fi[n, t - 1] + self.Fi[n, t - 2]
                self.D[n, t] = abs(dI_dt - d2Fi_dt2)
    
    def compute_ICR(self):
        for n in range(self.N):
            for t in range(self.tau, self.T):
                self.Ci[n, t] = self.mu * (self.D[n, t] / (self.Fi[n, t] + self.epsilon)) * (1 - self.R[n, t])

    def compute_ETE(self):
        for n in range(self.N):
            for t in range(self.tau, self.T):
                self.Etrait[n, t] = self.sigma * np.sum(self.R[n, t - self.tau:t] * self.Fi[n, t - self.tau:t]) / self.tau

    def compute_RSL(self):
        for n in range(self.N):
            for t in range(self.T):
                self.Sr[n, t] = self.rho * (self.Etrait[n, t] ** 2) / (1 + self.Fi[n, t])

    def compute_RTP(self):
        for n in range(self.N):
            for t in range(self.T):
                r = min(n, 15)
                if n > 0:
                    k_weights = np.array([1.0 / (k + 1) for k in range(r)])
                    R_vals = np.array([self.R[n - k - 1, t - k - 1] for k in range(len(k_weights))])
                    self.P[n, t] = self.eta * self.Etrait[n, t] * np.sum(k_weights * R_vals)

    def compute_RCP(self):
        for n in range(self.N):
            for t in range(self.T):
                threshold_c = 1.0
                self.Cprevent[n, t] = self.zeta * max(0, threshold_c - self.D[n, t] - self.Sr[n, t])

    def compute_RSA(self):
        for n in range(self.N):
            for t in range(self.T):
                 self.Sa[n, t] = self.omega * (self.Etrait[n, t] * self.R[n, t]) / (1 + abs(self.Ci[n, t] - self.Cprevent[n, t]))
                 


    def run(self):
        self.compute_TPI()
        self.compute_ECF()
        self.compute_RFI()
        self.compute_RSO()
        self.compute_IFD()
        self.compute_RDD()
        self.compute_ICR()
        self.compute_ETE()
        self.compute_RTP()
        self.compute_RSL()
        self.compute_RCP()
        self.compute_RSA()



    def plot(self, indices=[0, 5, 9]):
        variables = {
            'RII': self.I, 'Fi': self.Fi, 'RSO': self.R, 'Etrait': self.Etrait,
            'RSL': self.Sr, 'RSA': self.Sa
        }
        time = np.arange(self.T)
        fig, axs = plt.subplots(len(variables), 1, figsize=(14, 20), sharex=True)
        plt.subplots_adjust(hspace=0.5)
        for ax, (label, data) in zip(axs, variables.items()):
            for n in indices:
                if n < self.N:
                    ax.plot(time, data[n], label=f'n={n}')
            ax.set_ylabel(label)
            ax.legend()
            ax.grid(True)
        axs[-1].set_xlabel("Time Step")
        plt.suptitle("CECE Simulation Results", fontsize=16)
        plt.show()

def RRT(System_x, System_y, chi = 0.8):
    Rxy = np.zeros(T)
    n = 3
    for t in range(T):
        Rxy[t] = chi * (System_x.Etrait[n, t] * System_y.Etrait[n, t]) / (1 + abs(System_x.Fi[n, t] - System_y.Fi[n, t]))
    return Rxy



# Instantiate and run
CECEx = CECE_System()
CECEy = CECE_System()

CECEx.run()
CECEy.run()

CECEx.run()

Rxy = RRT(CECEx,CECEy)

#print (Rxy[10])