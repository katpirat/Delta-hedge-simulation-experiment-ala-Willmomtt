import matplotlib.pyplot as plt 
import numpy as np
from scipy.stats import norm

class GBM: 
    def __init__(self, n_steps, n_sims, mu, sigma, r, T = 1, S0 = 100, K = 120) -> None:
        self.n_steps = n_steps 
        self.n_sims = n_sims
        self.mu = mu
        self.sigma = sigma 
        self.implied_volatility = self.sigma + 0.1
        self.r = r 
        self.S0 = S0
        self.T = T
        self.K = K
        self.generated_path = self.generate_path()
    
    def generate_path(self): 
        dt = self.T/self.n_steps
        sim_test = np.random.normal(0, np.sqrt(dt), size = (self.n_sims, self.n_steps-1))

        asset_path = np.exp(
            (self.mu - self.sigma**2/2) * dt + self.sigma * sim_test
        )

        l = np.full(shape = (self.n_sims, 1), fill_value=self.S0)

        res = self.S0 * asset_path.cumprod(axis = 1)
        res = np.hstack((l, res))
        return res 
    

    def p_fct(self): 
        plt.plot(self.path)
        plt.show()
    

    def C_pricing(self, S, tau, sigma = None, single = True):
        if sigma is None: 
            sigma = self.sigma 
        if single: 
            d1 = self.d1_calc(S, tau, sigma)
            d2 = d1 - sigma * np.sqrt(tau)
            price = norm.cdf(d1) * S - self.K * np.exp(-self.r * tau) * norm.cdf(d2)
            return price
        else: 
            d1 = np.array([self.d1_calc(i, tau) for i in S])
            print(d1)
            d2 = d1 - sigma * np.sqrt(tau)
            print(d2)
            price = norm.cdf(d1) * S - self.K *np.exp(-self.r * tau) * norm.cdf(d2)
            print(price)
            return price

    def d1_calc(self, S, tau, sigma): 
        return 1/(sigma * (np.sqrt(tau))) * (np.log(S/self.K) + (self.r + sigma ** 2 /2) * tau)

    def C_delta(self, S: list, tau: list, sigma = None, single = True):
        if sigma is None: 
            sigma = self.sigma 
        if single: 
            res = norm.cdf(self.d1_calc(S = S, tau = tau, sigma = sigma))
            return res 
        else: 
            res = [norm.cdf(self.d1_calc(S = i, tau = tau, sigma=sigma)) for i in S]
            return np.array(res)
    
    def payoff_call(self, upper = 50, lower = 150): 
        S = np.linspace(lower, upper, 100)
        res = [max(0, i-self.K) for i in S]
        return S, np.array(res)

    def delta_hedge(self): 
        dt = 1/self.n_steps
        tau = self.T - np.linspace(0, self.T-dt, self.n_steps) 
        pf_value = np.zeros(shape = (self.n_sims, self.n_steps))
        pf_hedge_value = np.zeros(shape = (self.n_sims, self.n_steps))

        pf_value[:, 0] = np.full(fill_value = self.C_pricing(self.S0, tau = self.T), shape = n_sims)
        pf_hedge_value[:, 0] = self.C_pricing(self.S0, tau = self.T)
        a = self.C_delta(self.generated_path[:, 0], tau=T)
        b = pf_value[:, 0] - a * self.generated_path[:, 0] 
        for i in range(1, self.n_steps): 
            pf_value[:, i] = a * self.generated_path[:, i] + b * np.exp(dt * r)
            a = self.C_delta(self.generated_path[:, i], tau = tau[i])
            b = pf_value[:, i] - a * self.generated_path[:, i]
            pf_hedge_value[:, i] = pf_value[:, i] - self.C_pricing(self.generated_path[:, i], tau = tau[i])

        return pf_value, pf_hedge_value
    
    def ftodt(self, hedge_sigma): 
        dt = 1/self.n_steps
        t = 0
        pf_value = np.zeros(shape = (self.n_sims, self.n_steps))
        Pi = np.zeros(shape = (self.n_sims, self.n_steps))

        # Initial setup: 
        pf_value[:, 0] = np.full(fill_value = self.C_pricing(self.S0, tau = self.T, sigma = self.implied_volatility), shape = n_sims)
        a = self.C_delta(self.generated_path[:, 0], tau=self.T, sigma = hedge_sigma)
        b = pf_value[:, 0] - a * self.generated_path[:, 0] 

        for i in range(1, self.n_steps):
            t += dt
            pf_value[:, i] = a * self.generated_path[:, i] + b * np.exp(dt * r)
            a = self.C_delta(self.generated_path[:, i], tau=self.T - t, sigma = hedge_sigma)
            b = pf_value[:, i] - a * self.generated_path[:, i]
            Pi[:, i] = pf_value[:, i] - self.C_pricing(self.generated_path[:, i], tau = self.T-t, sigma = self.implied_volatility)
        return Pi


    def plot_hedge_payoff(self): 
        hedge_val = self.delta_hedge()[0][:, -1]
        S_N = self.generated_path[:, -1]
        true_vals = self.payoff_call(lower = min(S_N), upper = max(S_N))
        plt.plot(S_N, hedge_val, 'bo')
        plt.plot(true_vals[0], true_vals[1], 'r-')
        plt.show()

        
# Parameters, change at will; 

n_steps = 1000 # Nr of steps pr. simulated path. The higher the number, the better the hedge. 
# However in real life the transaction costs would be too high. 
n_sims = 10 # number of stock paths to simulate 
S0 = 100 # Initial spot price 
K = 120 # Strike 
mu = 0.05 # Drift of the proess 
sigma = 0.2 # True volatiliy in the model 
sigma_imp = 0.3 # Implied volatility
r = 0.03 # Interest rate 
T = 1 # Duration of call option. 
dt = 1/n_steps

obj1 = GBM(n_steps = n_steps,
            n_sims = n_sims, 
            mu = mu,
            sigma = sigma,
            r=r, 
            S0 = S0, 
            K = K) # Create an 
obj1.plot_hedge_payoff()

k = obj1.ftodt(sigma)
plt.plot(k.T)
plt.show()

k = obj1.ftodt(sigma_imp)
plt.plot(k.T)
plt.show()

