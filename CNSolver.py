import scipy.sparse as sps
import numpy as np
from scipy.sparse.linalg.dsolve import linsolve


class CNSolver(object):
    def __init__(self):
        self.buff = None
        self.solutions = []

    def fit(self, mod):
        self.mod = mod
        self.mod.prepare()
        self.initial_state = self.mod.initial_state
        self.buff = self.initial_state
        self.solutions.append(self.buff)

    def _decompose_element_position(self, index):
        j = index % self.mod.M
        i = (index - j) / self.mod.M
        return i, j
    
    def is_stable(self, sol):
        l = self.mod.lambd
        dt = self.mod.dt
        a = self.mod.alpha
        b = self.mod.beta
        m = self.mod.m_0
        max_stable = 0
        for point in xrange(len(sol)):
            i, j = self._decompose_element_position(point)
            stable = (1.-(1.-l)*(dt*a*m + self.mod.stability_factor(i,j)) - (1-2.*l)*dt*a*b*sol[point])/\
                        (1.-l*(dt*m*a-2.*sol[point]*dt*a*b-self.mod.stability_factor(i,j)))
            if np.abs(stable)>max_stable:
                max_stable = np.abs(stable)
        if max_stable < 1:
            return True
        else:
            return False

    def _generate_LHS(self):
        print "Generating LHS"
        m = self.mod.M
        n = self.mod.N
        num_of_eqs = m * n
        LHS = sps.lil_matrix((num_of_eqs, num_of_eqs), dtype=np.float64)
        for q in xrange(num_of_eqs):
            i, j = self._decompose_element_position(q)
            if (i == 0 or i == (n - 1) or j == 0 or j == (m - 1)):
                LHS[q, q] = 1.
            else:
                LHS[q, q] = self.mod.footprint['center'] + 2. * self.buff[q] * \
                    self.mod.lambd * self.mod.dt * self.mod.alpha * self.mod.beta
                LHS[q, q - 1] = self.mod.footprint['left']
                LHS[q, q + 1] = self.mod.footprint['right']
                LHS[q, q - m] = self.mod.footprint['up']
                LHS[q, q + m] = self.mod.footprint['down']
        LHS = LHS.tobsr()
        return LHS

    def _generate_RHS(self):  # must be rewritten
        m = self.mod.M
        n = self.mod.N
        sx = self.mod.S_x
        sy = self.mod.S_y
        square_coef = self.mod.dt * self.mod.alpha * self.mod.beta
        center_coef = self.mod.dt*self.mod.alpha*self.mod.m_0 - 2. * self.mod.S
        rhs = []
        num_of_eqs = m * n
        for q in range(m):
            rhs.append(0.)

        for q in range(m, num_of_eqs - m):
            if q % m == 0 or q % m == m - 1:
                rhs.append(0.)
            else:
                rhs.append((center_coef - square_coef * self.buff[q]) * self.buff[q] + sx * (
                    self.buff[q - 1] + self.buff[q + 1]) + sy * (self.buff[q - m] + self.buff[q + m]))
        for q in range(num_of_eqs - m, num_of_eqs):
            rhs.append(0.)
        return rhs

    def solve(self):
        debug = True
        sol = None  # np.zeros((self.mod.M, self.mod.N), dtype=float)
        for k in range(self.mod.T):
            print "Generating RHS  # %s" % k
            LHS = self._generate_LHS()
            RHS = self._generate_RHS()
            print "Solving % s system" % k
            sol = linsolve.spsolve(LHS, RHS)
            sol = sol + self.buff
            stable = self.is_stable(sol)
            if not stable:
                print "No stability!"
                self.buff = self.initial_state
                break
            self.solutions.append(sol)
            self.buff = sol
        for i in range(len(self.solutions)):
            self.solutions[i] = self.solutions[i].reshape((self.mod.M, self.mod.N))
        self.buff = self.initial_state
