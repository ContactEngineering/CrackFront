
import math

import scipy
from scipy.optimize import OptimizeResult


from CrackFront.Optimization.fixed_radius_trustregion_newton_cg import CGSteihaugSubproblem
import numpy as np


def trustregion_newton_cg(x0, gradient, hessian=None, hessian_product=None,
                          trust_radius=0.5,
                          max_trust_radius=10,
                          gtol=1e-6, maxiter=1000,
                          reduce_threshold=0.5,
                          increase_threshold=0.05,
                          trust_threshold=0.6,
                          trust_radius_from_x=None, cg_tolerance=None, verbose=False):
    r"""
    minimizes the function having the given gradient and hessian
    In other words it finds only roots of gradient where the hessian
    is positive semi-definite

    This is the Newton algorithm using the Steihaug-CG (Nocedal algorithm 7.2)
    to determine the step.
    The step-length is limitted by trust-radius
    """

    if verbose:
        def log(*args):
            print(*args)
    else:
        def log(*args):
            pass

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
                             cg_tolerance=cg_tolerance,
                             verbose=verbose)
    n_hits_boundary = 0
    nit = 1
    n_CG = 0
    while nit <= maxiter:
        log(f"####### it {nit}")

        # quadratic subproblem that uses the gradient = residual
        # and the hessian at a
        p, hits_boundary = m.solve(trust_radius=trust_radius)
        n_CG += m.nit
        # solve with CG-Steihaug (Nocedal Algorithm 7.2.)
        # we choose trust_radis based on our knowledge of the nonlinear
        # part of the function

        x_proposed = x + p

        predicted_gradient = m.jac + m.hessp(p)
        actual_gradient = wrapped_gradient(x_proposed)

        rel_gradient_error = scipy.linalg.norm(predicted_gradient - actual_gradient) / scipy.linalg.norm(actual_gradient) # accuracy of prediction
        log("rel_gradient_error", rel_gradient_error)

        if rel_gradient_error > reduce_threshold:
            trust_radius *= 0.25
            log("trust_radius: ", trust_radius)
        elif rel_gradient_error < increase_threshold:
            if hits_boundary:
                trust_radius = min(2 * trust_radius, max_trust_radius)
                log("trust_radius: ", trust_radius)

        #print(trust_radius)
        if hits_boundary:
            n_hits_boundary = n_hits_boundary + 1

        if rel_gradient_error < trust_threshold:
            x = x_proposed
            m = CGSteihaugSubproblem(x, fun=lambda x: 0,
                                     jac=wrapped_gradient,
                                     hess=wrapped_hessian
                                     if hessian is not None else None,
                                     hessp=wrapped_hessian_product
                                     if hessian_product is not None else None,
                                     cg_tolerance=cg_tolerance, verbose=verbose)



            max_r = np.max(abs(m.jac))
            #print(f"max(|r|)= {max_r}")
            if max_r < gtol:
                result = OptimizeResult(
                    {
                        'success': True,
                        'x': x,
                        'jac': m.jac,
                        'nit': nit,
                        'nCG': n_CG,
                        'nfev': 0,
                        'njev': njev,
                        'nhev': nhev,
                        'n_hits_boundary': n_hits_boundary,
                        'message': 'CONVERGENCE: CONVERGENCE: NORM_OF_GRADIENT_<=_GTOL', # noqa E501
                        })
                return result
        else:
            log("step regected")
        nit += 1

    result = OptimizeResult({
        'success': False,
        'x': x,
        'jac': m.jac,
        'nit': nit,
        'nCG': n_CG,
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
    from CrackFront.GenericElasticLine import ElasticLine
    figall, axall = plt.subplots()
    for n, marker in ((9, ".-"), (20, "+-"), (200, "x-")):
        # Parameters
        C = 0.1
        kappa = 0.05

        dK = 0.4


        a = np.zeros(n)

        phases = np.random.uniform(size=n) * 2 * np.pi

        q_potential = 2 * np.pi / 3

        def potential(a, der="0"):
            if der == "0":
                return np.cos(a * q_potential + phases)
            elif der == "1":
                return - q_potential * np.sin(a * q_potential + phases)

        sy = 2
        dy = sy / n
        y = np.arange(n) * dy

        cf = ElasticLine(n, Lk = n/2, pinning_field=potential)
        fig, ax = plt.subplots()
        x = np.arange(2 * n) * dy

        #ax.imshow(potential(y.reshape(-1,1) * np.ones((1, len(x)))))
        K0=[]
        a0=[]
        for driving_a in np.concatenate((np.linspace(-7, 5, 50),
                                         np.linspace(5, -5, 50))):


            sol = trustregion_newton_cg(x0=a, gradient=lambda a: cf.gradient(a, a_forcing=driving_a),
                                         hessian_product=lambda a, p: cf.hessian_product(p,a),
                                        gtol=1e-14,
                                        maxiter=1000,
                                        verbose=True)
            assert sol.success
            a = sol.x
            ax.plot(a / dy, y / dy)
            K0.append(np.mean(dK * np.cos(np.pi * a) * np.cos(np.pi * y)))
            a0.append(np.mean(a))

            plt.pause(50)
        axall.plot(a0, K0, marker, label=f"n = {n}")
    axall.legend()
