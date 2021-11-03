# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext snakeviz

# %%
import numpy as np

from CrackFront.GenericElasticLine import ElasticLine
from CrackFront.Optimization.fixed_radius_trustregion_newton_cg import trustregion_newton_cg
from CrackFront.Optimization.RossoKrauth import linear_interpolated_pinning_field, brute_rosso_krauth
import pytest

# %%
np.random.seed(0)

L = 8192
Lx = 256

random_forces = np.random.normal(size=(L, Lx)) * 1.  # *0.05
pinning_field = linear_interpolated_pinning_field(random_forces)

line = ElasticLine(L, L / 16, pinning_field=pinning_field)

gtol = 1e-10
w = 0
# Do an initial configuration
sol = trustregion_newton_cg(
    x0=np.zeros(L), gradient=lambda a: line.gradient(a, w),
    hessian_product=lambda a, p: line.hessian_product(p, a),
    trust_radius=1 / 8,
    gtol=gtol,
    maxiter=10000000)
assert sol.success, sol.message
a_init = sol.x

a_forcings = np.linspace(0, 100, 300)[1:]
a_forcings = np.concatenate([a_forcings, a_forcings[:-1][::-1]])

# %%
# %%snakeviz
# %%
mean_a_RK = []
a = a_init.copy()
a_forcing_prev = 0
for a_forcing in a_forcings:
    # print(a_forcing)
    dir = 1 if a_forcing > a_forcing_prev else -1
    # print(dir)
    sol = brute_rosso_krauth(a, a_forcing, line, gtol=gtol, maxit=100000, direction=dir)
    assert sol.success
    a = sol.x
    mean_a_RK.append(np.mean(a))
    a_forcing_prev = a_forcing

print("RK done")

# %%

# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(a_forcings, a_forcings-mean_a_RK)

# %%
