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

# %% [markdown]
# Circular, linearized versus not linearised

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
from Adhesion.ReferenceSolutions import JKR

from CrackFront.Circular import cart2pol, pol2cart, SphereCrackFrontPenetration
from CrackFront.Optimization import trustregion_newton_cg

# %%
w = 1 / np.pi
Es = 3. / 4
mean_Kc = np.sqrt(2 * Es * w)

# %%
penetration = -0.4
for n_rays in [1, 2, 8]:
    for dK in 0.1, 0.4, 0.8: 

        def kc(radius, angle):
            return  (1 + dK * np.cos(angle * n_rays) )  * mean_Kc

        def dkc(radius, angle):
            return np.zeros_like(radius)

        n = 256 # discretisation


        a = np.ones(cf.npx) * JKR.contact_radius(penetration=penetration)

        cf = SphereCrackFrontPenetration(n,
                                         kc=kc,
                                         dkc=dkc,
                                         lin=False)

        sol = trustregion_newton_cg(
                    x0=a, gradient=lambda a: cf.gradient(a, penetration),
                    hessian=lambda a: cf.hessian(a, penetration),
                    trust_radius=0.25 * np.min(a),
                    maxiter=3000,
                    gtol=1e-11)
        assert sol.success

        radii_nonlin = sol.x



        cf = SphereCrackFrontPenetration(n,
                                         kc=kc,
                                         dkc=dkc,
                                         lin=True)

        sol = trustregion_newton_cg(
                    x0=a, gradient=lambda a: cf.gradient(a, penetration),
                    hessian=lambda a: cf.hessian(a, penetration),
                    trust_radius=0.25 * np.min(a),
                    maxiter=3000,
                    gtol=1e-11)
        assert sol.success

        radii_lin = sol.x

        fig, ax = plt.subplots()

        ax.plot(radii_lin, label="linearised")
        ax.plot(radii_nonlin, label="normal")
        ax.set_xlabel("pixel")
        ax.set_ylabel("radius")
        ax.set_title("penetration={}, dK={}".format(penetration,dK))
        ax.legend()




# %% [markdown]
# So mainly, not linearizing the 0-mode term decreases the mean contact radius. For high dK and low number of rays it is also visible that the amplitude is increased.

# %% [markdown]
# Where does this come from ? From the convexity of the SIF with respect to the radius so that fluctuations in the radius will inncrease the mean stress intensity factor for a given mean radius. This has to be compensated by a raduction of the mean contact radius.

# %% [markdown]
# So is the stress intensity factor convex with $a$ ? 

# %%
fig, ax= plt.subplots()

main_radius = JKR.contact_radius(penetration=penetration)

a = np.linspace(0.9, 1.1) * main_radius

ax.plot(a, JKR.stress_intensity_factor(a, penetration), label="SIF(a, penetration)")
ax.plot(a, JKR.stress_intensity_factor(main_radius, penetration) + (a - main_radius) *JKR.stress_intensity_factor(main_radius, penetration, der="1_a"), "--k", label="linearized")

ax.axhline(mean_Kc, ls=":", c="k")
ax.axvline(main_radius,ls=":",c="k",  label="equilibrium")
ax.set_xlabel("contact radius")
ax.set_ylabel("stress intensity factor")
ax.set_title("penetration = {}".format(penetration))

# %% [markdown]
# ## Curvature of the stress intensity factor for different penetrations (equilibrium)
#

# %%
fig, ax = plt.subplots()

a = np.linspace(0.1, 4)

ax.plot(a, JKR.stress_intensity_factor(a, JKR.penetration(a)))

ax.set_xlabel(r"contact radius")
ax.set_ylabel(r"$\frac{\partial^2K}{\partial a^2}$")

# %% [markdown]
# OK, looks like the curvature is always positive and has the same value.
#
# TODO: show this analytically

# %%
