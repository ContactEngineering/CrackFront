
from scipy.optimize import OptimizeResult
from scipy.optimize._trustregion_ncg import CGSteihaugSubproblem
import numpy as np

def trustregion_newton_cg(x0, gradient, hessian=None, hessian_product=None,
                                 trust_radius=0.5, gtol=1e-6, maxiter=1000):
    r"""
    minimizes the function having the given gradient and hessian
    In other words it finds only roots of gradient where the hessian is positive semi-definite

    This is the Newton algorithm using the Steihaug-CG (Nocedal algorithm 7.2) to
    determine the step.
    The step-length is limitted by trust-radius
    """
    x = x0
    m = CGSteihaugSubproblem(x, fun=lambda a: 0, jac=gradient, hess=hessian, hessp=hessian_product, )
    n_hits_boundary = 0
    nit = 1
    while nit <=maxiter:
        #print(f"####### it {nit}")

        # quadratic subproblem that uses the gradient = residual and the hessian at a

        p, hits_boundary  = m.solve(trust_radius=trust_radius) # solve with CG-Steihaug (Nocedal Algorithm 7.2.)
        # we choose trust_radis based on our knowledge of the nonlinear part of the function

        x = x + p

        if hits_boundary:
            n_hits_boundary = n_hits_boundary + 1

        m = CGSteihaugSubproblem(x, fun=lambda x: 0, jac=gradient, hess=hessian, hessp=hessian_product)

        max_r = np.max(abs(m.jac))
        #print(f"max(|r|)= {max_r}")
        if max_r < gtol:
            result = OptimizeResult({'success': True,
                                     'x': x,
                                     'jac': m.jac,
                                     'nit': nit,
                                     'n_hits_boundary': n_hits_boundary,
                                     'message': 'CONVERGENCE: CONVERGENCE: NORM_OF_GRADIENT_<=_GTOL',
                                     })
            return result
        nit+=1

    result = OptimizeResult({'success': False,
                                     'x': x,
                                     'jac': m.jac,
                                     'nit': nit,
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
        ax.imshow(dK  *  np.cos(np.pi * x.reshape(1,-1)) * np.cos(np.pi * y.reshape(-1, 1)))

        # Defining residual and jacobian
        elastic_jac = np.zeros((n,n))
        v = np.fft.irfft(q/2, n=n)
        for i in range(n):
            for j in range(n):
                elastic_jac[i, j] = v[i-j]
        elastic_jac *=C
        # check elastic jacobian
        a = np.random.normal(size = n)
        np.testing.assert_allclose(elastic_jac @ a, C * np.fft.irfft(q / 2 * np.fft.rfft(a), n=n))

        J_drive = kappa * np.eye(n)

        a0 = []
        K0 = []
        a = np.zeros(n)
        def hessian(a):
            J_dis = np.diag(- dK  *  np.pi * np.sin(np.pi * a) * np.cos(np.pi * y))
            J = elastic_jac + J_drive + J_dis
            return J

        for driving_a in np.concatenate((np.linspace(-7, 5, 50),
                                          np.linspace(5, -5, 50))):
            def gradient(a):
                return dK  *  np.cos(np.pi * a) * np.cos(np.pi * y) + (elastic_jac) @ a + kappa * (a - driving_a)

            sol = trustregion_newton_cg(a, gradient, hessian,
                                 trust_radius=0.5, gtol=1e-8, maxiter=1000)
            assert sol.success
            a = sol.x
            ax.plot(a / dy, y / dy)
            K0.append(np.mean(dK  *  np.cos(np.pi * a) * np.cos(np.pi * y)))
            a0.append(np.mean(a))

            plt.pause(0.0001)
        axall.plot(a0, K0, marker,label=f"n = {n}")
    axall.legend()