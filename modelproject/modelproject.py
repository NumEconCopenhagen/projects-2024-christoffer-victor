from types import SimpleNamespace
import sympy as sm
from scipy.optimize import brentq
from scipy.optimize import fsolve
import numpy as np

class ASADClass:

    def __init__(self):

        par = self.par = SimpleNamespace()

        # I. parameter for standard model 
        par.s = 0.101
        par.alpha = 0.33
        par.delta = 0.05
        par.phi = 0.5
        par.n = 0.02

        # II. parameter for extended model
        par.sk = 0.15
        par.sh = 0.2
        par.alpha_2 = 0.30
        par.gamma = 0.35 
        par.n_2 = 0.025
        par.phi_2 = 0.3
        # delta is the same as in the standard model

    def transition_equation(self, k):
        par = self.par

        return k - k/(1+par.n) * (par.s * k**(par.alpha - 1) + 1 - par.delta)**(1 - par.phi)
    
    def solve(self):
        par = self.par

        k = brentq(self.transition_equation, 0.01, 100)
        y = k**par.alpha

        return k, y
    
    def transition_equation_phi(self, k, phi):
        par = self.par

        return k - k / (1 + par.n) * (par.s * k**(par.alpha - 1) + 1 - par.delta)**(1 - phi)

    def solve_with_phi_vec(self, phi_vec):
        k_values = []
        y_values = []
        for phi_val in phi_vec:
            k_phi = brentq(self.transition_equation_phi, 0.01, 100, args=(phi_val,))
            y_phi = k_phi**self.par.alpha
            k_values.append(k_phi)
            y_values.append(y_phi)

        return k_values, y_values
    
    def transition_equations(self, var):
        par = self.par
        k, h = var
        k_trans = ((par.sk * k**(par.alpha_2 - 1) * h**par.gamma + (1 - par.delta))**(1 - par.phi_2) / (1 + par.n_2) - 1)
        h_trans = ((par.sh * h**(par.gamma - 1) * k**par.alpha_2 + (1 - par.delta)) * (par.sk * k**(par.alpha_2 - 1) * h**par.gamma + (1 - par.delta))**(-par.phi_2) / (1 + par.n_2) - 1)

        return k_trans, h_trans

    def solve_steady_state(self):
        initial_guesses = [(0, 0), (1, 1), (10, 10), (100, 100)]
        for initial_guess in initial_guesses:
            with np.errstate(all='ignore'):
                k, h = fsolve(self.transition_equations, initial_guess)
                k_trans, h_trans = self.transition_equations((k, h))
                print(f"Initial guess: {initial_guess}: steady state value of capital and human capital is {k:.3f} and {h:.3f} where k_trans is {k_trans:.3f} and h_trans is {h_trans:.3f}")
    
        return k, h

