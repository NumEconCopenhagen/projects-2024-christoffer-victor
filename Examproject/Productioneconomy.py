import numpy as np
from scipy import optimize
from types import SimpleNamespace

class ProductionEconomyClass:

    def __init__(self):

        par = self.par = SimpleNamespace()

        # firms
        par.A = 1.0
        par.gamma = 0.5

        # households
        par.alpha = 0.3
        par.nu = 1.0
        par.epsilon = 2.0

        # government
        par.tau = 0.0
        par.T = 0.0

        # Question 3
        par.kappa = 0.1

    # Firms

    def l_firm(self, p1, p2):
        par = self.par

        return (p1*par.A*par.gamma)**(1/(1-par.gamma)), (p2*par.A*par.gamma)**(1/(1-par.gamma))  

    def y_firm(self, l1, p1, l2, p2):
        par = self.par
        l1, l2 = self.l_firm(p1, p2)

        return par.A*l1**par.gamma, par.A*l2**par.gamma

    def pi(self, p1, p2):
        par = self.par

        return (1-par.gamma)/par.gamma*(p1*par.A*par.gamma)**(1/(1-par.gamma)), (1-par.gamma)/par.gamma*(p2*par.A*par.gamma)**(1/(1-par.gamma))
    
    # Households

    def utility(self, p1, p2, l):
        par = self.par
        pi1 = self.pi1(p1)
        pi2 = self.pi2(p2)

        c1_l = par.alpha*(l+par.T+pi1+pi2)/p1
        c2_l = (1-par.alpha)*(l+par.T+pi1+pi2)/(p2*par.tau)

        return np.log(c1_l**par.alpha*c2_l**(1-par.alpha))-par.nu*l**(1+par.epsilon)/(1+par.epsilon)
    
    def l(self, p1, p2):
        par = self.par

        obj = lambda l: -self.utility(p1, p2, l)
        res = optimize.minimize_scalar(obj, bounds=(0,1), method='bounded')

        l = res.x
        c1 = par.alpha*(l+par.T+self.pi1(p1)+self.pi2(p2))/p1
        c2 = (1-par.alpha)*(l+par.T+self.pi1(p1)+self.pi2(p2))/(p2*par.tau)

        return l, c1, c2
    
    def check_market_clearing(self, p1, p2):
        par = self.par

        l, c1, c2 = self.l(p1, p2)
        l1, l2 = self.l_firm(p1, p2)
        y1, y2 = self.y_firm(l1, p1, l2, p2)

        eps1 = l1 + l2 - l
        eps2 = c1 - y1
        eps3 = c2 - y2

        return eps1, eps2, eps3
    