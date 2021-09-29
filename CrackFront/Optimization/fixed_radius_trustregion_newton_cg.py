#
# Copyright 2020 Antoine Sanner
#
# ### MIT license
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import math

import scipy
from scipy.optimize import OptimizeResult
from scipy.optimize._trustregion_ncg import BaseQuadraticSubproblem
import numpy as np


class CGSteihaugSubproblem(BaseQuadraticSubproblem):
    """Quadratic subproblem solved by a conjugate gradient method
    Adapted from scipy
    """
    def __init__(self, x, fun, jac, hess=None, hessp=None, cg_tolerance=None):
        super().__init__(x, fun, jac, hess, hessp)
        if cg_tolerance==None:
            cg_tolerance = lambda jac_mag: min(0.5, math.sqrt(jac_mag))
        self._cg_tolerance = cg_tolerance

    def solve(self, trust_radius):
        """
        Solve the subproblem using a conjugate gradient method.

        Parameters
        ----------
        trust_radius : float
            We are allowed to wander only this far away from the origin.

        Returns
        -------
        p : ndarray
            The proposed step.
        hits_boundary : bool
            True if the proposed step is on the boundary of the trust region.

        Notes
        -----
        This is algorithm (7.2) of Nocedal and Wright 2nd edition.
        Only the function that computes the Hessian-vector product is required.
        The Hessian itself is not required, and the Hessian does
        not need to be positive semidefinite.
        """

        # get the norm of jacobian and define the origin
        p_origin = np.zeros_like(self.jac)

        # define a default tolerance
        tolerance = self._cg_tolerance(self.jac_mag)

        # Stop the method if the search direction
        # is a direction of nonpositive curvature.
        if self.jac_mag < tolerance:
            hits_boundary = False
            return p_origin, hits_boundary

        # init the state for the first iteration
        z = p_origin
        r = self.jac
        d = -r

        # Search for the min of the approximation of the objective function.
        nit_CG = 1
        while True:

            # do an iteration
            Bd = self.hessp(d)
            dBd = np.dot(d, Bd)
            if dBd <= 0:
                # Look at the two boundary points.
                # Find both values of t to get the boundary points such that
                # ||z + t d|| == trust_radius
                # and then choose the one with the predicted min value.
                ta, tb = self.get_boundaries_intersections(z, d, trust_radius)
                pa = z + ta * d
                pb = z + tb * d
                if self(pa) < self(pb):
                    p_boundary = pa
                else:
                    p_boundary = pb
                hits_boundary = True
                return p_boundary, hits_boundary
            r_squared = np.dot(r, r)
            alpha = r_squared / dBd
            z_next = z + alpha * d
            if scipy.linalg.norm(z_next) >= trust_radius:
                # Find t >= 0 to get the boundary point such that
                # ||z + t d|| == trust_radius
                ta, tb = self.get_boundaries_intersections(z, d, trust_radius)
                p_boundary = z + tb * d
                hits_boundary = True
                return p_boundary, hits_boundary
            r_next = r + alpha * Bd
            r_next_squared = np.dot(r_next, r_next)
            #if math.sqrt(r_next_squared) < tolerance:
            if np.max(abs(r_next)) < tolerance:
                hits_boundary = False
                print("CG reaches tolerance: ", nit_CG)
                return z_next, hits_boundary
            beta_next = r_next_squared / r_squared
            d_next = -r_next + beta_next * d

            # update the state for the next iteration
            z = z_next
            r = r_next
            d = d_next
            nit_CG+=1


def trustregion_newton_cg(x0, gradient, hessian=None, hessian_product=None,
                          trust_radius=0.5, gtol=1e-6, maxiter=1000,
                          trust_radius_from_x=None, cg_tolerance=None):
    r"""
    minimizes the function having the given gradient and hessian
    In other words it finds only roots of gradient where the hessian
    is positive semi-definite

    This is the Newton algorithm using the Steihaug-CG (Nocedal algorithm 7.2)
    to determine the step.
    The step-length is limitted by trust-radius
    """
    x = x0

    # nfev = 0
    njev = 0
    nhev = 0

    def wrapped_gradient(*args):
        nonlocal njev
        njev += 1
        return gradient(*args)

    def wrapped_hessian(*args):
        nonlocal nhev
        nhev += 1
        return hessian(*args)

    def wrapped_hessian_product(*args):
        nonlocal nhev
        nhev += 1
        return hessian_product(*args)

    m = CGSteihaugSubproblem(x, fun=lambda a: 0, jac=wrapped_gradient,
                             hess=wrapped_hessian if hessian is not None
                             else None,
                             hessp=wrapped_hessian_product
                             if hessian_product is not None else None,
                             cg_tolerance=cg_tolerance)
    n_hits_boundary = 0
    nit = 1

    while nit <= maxiter:
        # print(f"####### it {nit}")
        if trust_radius_from_x is not None:
            # this allows to choose the trust radius
            # according to nonadmissible values of x
            trust_radius = trust_radius_from_x(x)
        # quadratic subproblem that uses the gradient = residual
        # and the hessian at a
        p, hits_boundary = m.solve(trust_radius=trust_radius)
        # solve with CG-Steihaug (Nocedal Algorithm 7.2.)
        # we choose trust_radis based on our knowledge of the nonlinear
        # part of the function

        x = x + p

        if hits_boundary:
            n_hits_boundary = n_hits_boundary + 1

        m = CGSteihaugSubproblem(x, fun=lambda x: 0,
                                 jac=wrapped_gradient,
                                 hess=wrapped_hessian
                                 if hessian is not None else None,
                                 hessp=wrapped_hessian_product
                                 if hessian_product is not None else None,
                                 cg_tolerance=cg_tolerance)
        max_r = np.max(abs(m.jac))
        print(f"max(|r|)= {max_r}")
        if max_r < gtol:
            result = OptimizeResult(
                {
                    'success': True,
                    'x': x,
                    'jac': m.jac,
                    'nit': nit,
                    'nfev': 0,
                    'njev': njev,
                    'nhev': nhev,
                    'n_hits_boundary': n_hits_boundary,
                    'message': 'CONVERGENCE: CONVERGENCE: NORM_OF_GRADIENT_<=_GTOL', # noqa E501
                    })
            return result
        nit += 1

    result = OptimizeResult({
        'success': False,
        'x': x,
        'jac': m.jac,
        'nit': nit,
        'nfev': 0,
        'njev': njev,
        'nhev': nhev,
        'n_hits_boundary': n_hits_boundary,
        'message': 'MAXITER REACHED',
        })
    return result


# A little demo
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    figall, axall = plt.subplots()
    for n, marker in ((9, ".-"), (20, "+-"), (200, "x-")):
        # Parameters
        C = 0.1
        kappa = 0.05

        dK = 0.4

        sy = 2
        dy = sy / n
        y = np.arange(n) * dy
        q = 2 * np.pi * np.fft.rfftfreq(n, sy / n)

        fig, ax = plt.subplots()
        x = np.arange(2 * n) * dy
        ax.imshow(dK * np.cos(np.pi * x.reshape(1, -1))
                  * np.cos(np.pi * y.reshape(-1, 1)))

        # Defining residual and jacobian
        elastic_jac = np.zeros((n, n))
        v = np.fft.irfft(q / 2, n=n)
        for i in range(n):
            for j in range(n):
                elastic_jac[i, j] = v[i - j]
        elastic_jac *= C
        # check elastic jacobian
        a = np.random.normal(size=n)
        np.testing.assert_allclose(
            elastic_jac @ a,
            C * np.fft.irfft(q / 2 * np.fft.rfft(a), n=n))

        J_drive = kappa * np.eye(n)

        a0 = []
        K0 = []
        a = np.zeros(n)

        def hessian(a):
            J_dis = np.diag(
                - dK * np.pi * np.sin(np.pi * a) * np.cos(np.pi * y))
            J = elastic_jac + J_drive + J_dis
            return J

        for driving_a in np.concatenate((np.linspace(-7, 5, 50),
                                         np.linspace(5, -5, 50))):
            def gradient(a):
                return dK * np.cos(np.pi * a) * np.cos(np.pi * y) \
                       + (elastic_jac) @ a + kappa * (a - driving_a)

            sol = trustregion_newton_cg(a, gradient, hessian,
                                        trust_radius=0.5, gtol=1e-8,
                                        maxiter=1000)
            assert sol.success
            a = sol.x
            ax.plot(a / dy, y / dy)
            K0.append(np.mean(dK * np.cos(np.pi * a) * np.cos(np.pi * y)))
            a0.append(np.mean(a))

            plt.pause(0.0001)
        axall.plot(a0, K0, marker, label=f"n = {n}")
    axall.legend()
