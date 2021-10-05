


import time

import numpy as np
import matplotlib.pyplot as plt
from CrackFront.Optimization import trustregion_newton_cg
from NuMPI.IO.NetCDF import NCStructuredGrid

from CrackFront.GenericElasticLineWithEnergy import ElasticLinePotentialPreconditionned
import scipy.optimize
from matplotlib import animation
from matplotlib.animation import FuncAnimation

from scipy.optimize import minimize

npx = L = 256

np.random.seed(10)
phases = np.random.uniform(size=npx) * 2 * np.pi

q_potential = 2 * np.pi / np.random.randint(1, 1024, size=npx)
sigma = 0.5

def potential(a, der="0"):
    if der == "0":
        return np.cos(a * q_potential + phases) * sigma
    elif der == "1":
        return - q_potential * np.sin(a * q_potential + phases) *sigma
    elif der == "2":
        return - q_potential**2 * np.cos(a * q_potential + phases) *sigma

# %%
Lk=L/2

line = ElasticLinePotentialPreconditionned(L, Lk, pinning_potential=potential)

driving_positions = np.linspace(0, 16, 200)


force = []
gtol = 1e-5
disp = False
for solver_name, trust_fac  in [#("lbfgsb", ""),
                                #("lbfgsb_hc", ""),
                                ("trust_ncg_fixed_preconditioned", 4),
                                ("lbfgsb_preconditioned", ""),
                                ("trust_ncg_fixed", 4),

                                #("trust_ncg_fixed", 4),
                                #("trust_ncg_fixed", 8),
                                #("trust_ncg_fixed", 2),
                                ]:
    print(solver_name)
    a = np.zeros(L)
    a_hc = np.zeros(L)
    a_p = np.zeros(L)
    nc = NCStructuredGrid(solver_name + f"{trust_fac}" + ".nc", "w", (L,))
    time_solver = 0
    for i in range(len(driving_positions)):
        w = driving_positions[i]
        start = time.time()
        if solver_name == "lbfgsb":
            sol = scipy.optimize.minimize(
                fun=lambda a: line.potential(a, w),
                x0=a,
                jac=lambda a: line.gradient(a, w),
                method="l-bfgs-b",
                options=dict(maxcor=3, gtol=gtol, maxiter=4000, ftol=1e-14))

            assert sol.success, sol.message
            #print(sol.message)
            a = sol.x
            #print(sol.nfev)
            #print(sol.njev)
        elif solver_name == "lbfgb_p":

            sol = scipy.optimize.minimize(
                fun=lambda a_hc: line.halfcomplex_potential(a_hc, w),
                x0=a_hc,
                jac=lambda a_hc: line.halfcomplex_gradient(a_hc, w),
                method="l-bfgs-b",
                options=dict(maxcor=3, gtol=gtol, maxiter=4000, ftol=1e-14))

            assert sol.success, sol.message
            #print(sol.message)
            a_hc = sol.x
            a = line.halfcomplex_to_real(a_hc)

        elif solver_name == "lbfgsb_preconditioned":

            sol = scipy.optimize.minimize(
                fun=lambda a_p: line.preconditioned_potential(a_p, w),
                x0=a_p,
                jac=lambda a_p: line.preconditioned_gradient(a_p, w),
                method="l-bfgs-b",
                options=dict(maxcor=3, gtol=gtol, maxiter=4000, ftol=1e-14))

            assert sol.success, sol.message

            a_p = sol.x
            a = line.halfcomplex_to_real(a_p / line.preconditioner)

        elif solver_name == "trust_ncg_fixed":
            sol = trustregion_newton_cg(
                x0=a, gradient=lambda a: line.gradient(a, w),
                hessian_product=lambda a, p: line.hessian_product(p, a),
                trust_radius=2 * np.pi / np.max(q_potential) / trust_fac, #TODO: this might be to small
                maxiter=10000,
                gtol=gtol  # he has issues to reach the gtol at small values of a
                        )
            #print(sol.n_hits_boundary)
            #print(sol.nhev)
            #print(sol.njev)

            assert sol.success, sol.message
            a=sol.x

        elif solver_name == "trust_ncg_fixed_preconditioned":
            def gradient(a_p):
                grad_p = line.preconditioned_gradient(a_p, a_forcing=w)
                print(line.gradient_norm_from_preconditioned(grad_p))
                return grad_p
            sol = trustregion_newton_cg(
                x0=a_p, gradient=gradient,
                hessian_product=lambda a_p, p_p: line.preconditioned_hessian_product(a_p=a_p, p_p=p_p),
                trust_radius=2 * np.pi / np.max(q_potential) / trust_fac / np.sqrt(line.L), # to a very good approximation, the two
                maxiter=10000,
                gtol=1e-2 #gtol  # he has issues to reach the gtol at small values of a
                        )
            #print(sol.n_hits_boundary)
            #print(sol.nhev)
            #print(sol.njev)

            assert sol.success, sol.message
            a_p=sol.x
            a = line.halfcomplex_to_real(a_p / line.preconditioner)

        time_solver += time.time() - start
        line.dump(nc[i], w, a, False)

    nc.close()
    print(solver_name, time_solver)


# %% [markdown]
#
# Note that the performance has not to be taken too seriously yet because I am not sure the termination criteria are equivalent.


# %%



fig, ax = plt.subplots()

nc = NCStructuredGrid("lbfgsb.nc")
symbol = "s"
ax.plot(nc.driving_position, nc.driving_force, symbol, alpha=0.5, label="lbfgsb")
nc.close()

nc = NCStructuredGrid("lbfgsb_hc.nc")
symbol = "x"
ax.plot(nc.driving_position, nc.driving_force, symbol, alpha=0.5, label="lbfgsa_p")
nc.close()

nc = NCStructuredGrid("lbfgsb_p.nc")
symbol = "+"
ax.plot(nc.driving_position, nc.driving_force, symbol, alpha=0.5, label="lbfgsb_preconditioned")
nc.close()

nc = NCStructuredGrid("trust_ncg_fixed.nc")
symbol = "^"
ax.plot(nc.driving_position, nc.driving_force, symbol, alpha=0.5, label="trust_ncg_p")
nc.close()


nc = NCStructuredGrid("trust_ncg_fixed_p.nc")
symbol = "^"
ax.plot(nc.driving_position, nc.driving_force, symbol, alpha=0.5, label="trust_ncg_p")
nc.close()


