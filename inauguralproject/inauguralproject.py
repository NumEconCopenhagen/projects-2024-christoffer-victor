from types import SimpleNamespace

class ExchangeEconomyClass:

    def __init__(self):

        par = self.par = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3

        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3

    def utility_A(self,x1A,x2A):
        par = self.par
        return (x1A ** par.alpha) * (x2A ** par.beta)

    def utility_B(self,x1B,x2B):
        par = self.par
        return (x1B ** par.beta) * (x2B ** par.alpha)

    def demand_A(self,p1):
        par = self.par
        return par.alpha * ( par.w1A + par.w2A / p1), par.beta * (p1 * par.w1A + par.w2A)

    def demand_B(self,p1):
        par = self.par
        return par.beta * (( 1 - par.w1A) +  (1 - par.w2A ) / p1), par.alpha * (p1 * (1 - par.w1A) + (1 - par.w2A))

    def check_market_clearing(self,p1):

        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1,eps2








































def square(x):
    """ square numpy array
    
    Args:
    
        x (ndarray): input array
        
    Returns:
    
        y (ndarray): output array
    
    """
    
    y = x**2
    return y

