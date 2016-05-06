import numpy as np
import types
from utils import cast_init_to_fun


class CNModel(object):
    '''
1) dt - time discretization step (mandatory)
2) dx, dy - coordinate discretization on a plane for Ox and Oy consequently (mandatory)
3) lambd - factor in Cranc-Nicilson model (mandatory)
4) T,M,N - number of nodes for time, Ox, Oy (int expected)

init_cond could be eiter callable or numerical; if float(int) given casting to constant function occurs
    '''

    def __init__(self, lambd, dt, dx, dy, T, M, N, sigma=1., alpha=1., 
                beta=1., m_0=1., init_cond=None):
        self.lambd = lambd
        self.dt = float(dt)
        self.dx = float(dx)
        self.dy = float(dy)
        if (type(T) is not int or type(M) is not int or type(N) is not int):
            raise ValueError("Nodes number (T,M,N): int expected")
        self.T = T
        self.M = M
        self.N = N
        self.domain = (self.dt * self.T, self.dx * (self.M - 1), self.dy * 
                        (self.N - 1))
        print "Model on (0,%s) time interval, [0,%s]x[0,%s] plane created" % self.domain
        self.alpha = alpha
        self.beta = beta
        self.m_0 = m_0
        if callable(init_cond):
            self.init_cond = init_cond
        else:
            self.init_cond = cast_init_to_fun(init_cond)
        self.sigma = sigma

    def _get_footprint(self):
        return {'up': -self.lambd * self.S_x,
                'down': -self.lambd * self.S_x,
                'left': -self.lambd * self.S_y,
                'right': -self.lambd * self.S_y,
                'center': 1. - self.lambd * self.dt * self.m_0 * self.alpha + 2. * self.lambd * self.S
                }

    def prepare(self):
        self.n_0 = float(self.m_0) / self.beta
        self.S_x = (self.sigma * self.dt) / (self.dx)**2
        self.S_y = (self.sigma * self.dt) / (self.dy)**2
        self.S = self.S_x + self.S_y
        self.footprint = self._get_footprint()
        self.initial_state = self.generate_initial_state()

    def generate_initial_state(self):
        init_grid = []
        for i in range(self.M):
            for j in range(self.N):
                init_grid.append(self.init_cond([i*self.dx,j*self.dy]))
        return np.asarray(init_grid, dtype=float)
    
    def stability_factor(self,i,j):
        return 2.*(self.S_x*(1-np.cos(i*self.dx)) + self.S_y*(1-np.cos(j*self.dy)))
