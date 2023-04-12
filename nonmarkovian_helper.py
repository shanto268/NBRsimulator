from scipy.stats import rv_discrete
from scipy.special import factorial
from numpy import exp

class poisson_mem(rv_discrete):
    "Poisson distribution altered by some memory"
    def _pmf(self, k, mu, delay):
        mu2 = mu*(1-exp(-delay))
        return exp(-mu2) * mu2**k / factorial(k)
    

class poisson_var(rv_discrete):
    "Poisson distribution which can alter mu on the fly in calling .rvs()"
    def _pmf(self, k, mu):
        return exp(-mu) * mu**k / factorial(k)