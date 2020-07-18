import numpy as np
import matplotlib.pyplot as plt
plt.style.use("presentation")

# +
from Adhesion.ReferenceSolutions import JKR

from CrackFront.Circular import cart2pol, pol2cart, SphereCrackFrontPenetration
from CrackFront.Optimization import trustregion_newton_cg

# +
# nondimensionalisation a la Maugis
w = 1 / np.pi
Es = 3. / 4
maugis_K=1.
R = 1.
mean_Kc = np.sqrt(2 * Es * w)

#smallest possible radius in the JKR contact: displacement controled  pulloff radius: 

puloff_radius = (np.pi * w * R**2 / 6 * maugis_K)**(1/3)
# -

puloff_radius

# \begin{equation}
# K(\theta, \{a\}, \Delta) = K^0(a_0, \Delta) + \sum_{n=-\infty\\n\neq 0}^{\infty} \left[ \left. \frac{\partial K^0(a_0, \Delta) }{\partial a_0} \right|_{\Delta} + \frac{|n|}{2 a_0} K^0(a_0, \Delta) \right] a_n e^{i n \theta}
# \end{equation}
#
# or we actually not necesseraly need to linearize the mode 0 components:
#
# \begin{equation}
# K(\theta, \{a\}, \Delta) = K^0(a, \Delta) + \sum_{n=-\infty\\n\neq 0}^{\infty} \frac{|n|}{2 a_0} K^0(a_0, \Delta) a_n e^{i n \theta}
# \end{equation}
#
#
# For the stress intensity factor distribution
# \begin{equation}
#     K_c(\theta) = \bar K_c (1 + dK \cos(n \theta))
# \end{equation}
# Using the linearized equation, the contact radius takes the form
# \begin{equation}
#     a(\theta) = a_0 + da \cos(\theta)
# \end{equation}
# with
# \begin{equation}
#     da = \frac{\bar K_c dK}{
#     \frac{\partial K^0}{\partial a}(a_0, \Delta) +
#     \frac{|n| K^0(a_0, \Delta)}{2 a_0}
#     }
# \end{equation}
# and :math:`a_0` the solution of
# \begin{equation}
#     \bar K_c = K^0(a_0, \Delta)
# \end{equation}
#
# We want to understand how much the $\frac{\partial K^0}{\partial a}(a_0, \Delta)$ term, 
# vs. the $\frac{|n| K^0(a_0, \Delta)}{2 a_0}$ term contribute to the total stiffness of the crack front. 
# The first term corresponds to describing each point along the perimeter of the contact as independent JKR contacts. 
# The second term is the stiffness of the crack front itself. 
#

# +
n = 64

angles = np.arange(n) / n * 2 * np.pi
penetration = -0.4
n_rays = 1
for dK in 0.1, 0.4, 0.8: 

    def kc(radius, angle):
        return  (1 + dK * np.cos(angle * n_rays) )  * mean_Kc

    def dkc(radius, angle):
        return np.zeros_like(radius)

    fig, ax = plt.subplots()
    a0 = JKR.contact_radius(penetration=penetration)
    radii_lin_by_hand = dK * mean_Kc / (JKR.stress_intensity_factor(a0, penetration, der="1_a") 
                              + JKR.stress_intensity_factor(a0, penetration) / (2*a0)) * np.cos(angles) + a0
    ax.plot(radii_lin_by_hand, label="all contributions")
    radii_front_only = dK * mean_Kc / (+ JKR.stress_intensity_factor(a0, penetration) / (2*a0)) * np.cos(angles) + a0
    ax.plot(radii_front_only, label="front only")
    radii_jkr_only = [JKR.contact_radius(penetration=penetration, 
                      work_of_adhesion= kc(1, theta)**2 / (2 * Es) ) for theta in angles]
    ax.plot(radii_jkr_only, label="independent JKR contacts")
    ax.set_xlabel("pixel")
    ax.set_ylabel("radius")
    ax.set_title("penetration={}, dK={}".format(penetration,dK))
    ax.legend()




# +

n_rays = 1
for penetration in -0.4, 1.:
    for dK in 0.1, 0.4, 0.8: 
        
        def kc(radius, angle):
            return  (1 + dK * np.cos(angle * n_rays) )  * mean_Kc

        def dkc(radius, angle):
            return np.zeros_like(radius)

        fig, ax = plt.subplots()
        a0 = JKR.contact_radius(penetration=penetration)
        radii_lin_by_hand = dK * mean_Kc / (JKR.stress_intensity_factor(a0, penetration, der="1_a") 
                                  + JKR.stress_intensity_factor(a0, penetration) / (2*a0)) * np.cos(angles) + a0
        ax.plot(*pol2cart(radii_lin_by_hand, angles), label="all contributions")
        radii_front_only = dK * mean_Kc / (+ JKR.stress_intensity_factor(a0, penetration) / (2*a0)) * np.cos(angles) + a0
        ax.plot(*pol2cart(radii_front_only, angles), label="front only")

        radii_jkr_only = [JKR.contact_radius(penetration=penetration, work_of_adhesion= kc(1, theta)**2 / (2 * Es) ) for theta in angles]
        ax.plot(*pol2cart(radii_jkr_only, angles), ".",label="q=0 term only only")
        ax.set_xlabel("pixel")
        ax.set_ylabel("radius")
        ax.set_title("penetration={}, dK={}".format(penetration,dK))
        ax.legend(bbox_to_anchor=(1.,1.05))
        ax.set_aspect(1)
        #ax.plot(0,0, "+k", mar)
        ax.axvline(0, c="gray")
        ax.axhline(0, c="gray")
        


# -

# Ok so actually the cosinusoidal radius corresponds to a simple translation only to smaller order.

# ## As a function of penetration, i.e. mean contact radius

# +
dK = 0.4

a0 = np.linspace(puloff_radius, 10)[1:]
penetration = JKR.penetration(contact_radius=a0)

fig, ax = plt.subplots()
ax.plot( a0 ,  dK / (JKR.stress_intensity_factor(a0, penetration, der="1_a") / mean_Kc + 1 / (2*a0)), label="linearized equation")
ax.plot( a0 ,  dK / (1 / (2*a0)), label="only crack front elasticity")
ax.plot( a0 ,  dK / (JKR.stress_intensity_factor(a0, penetration, der="1_a") / mean_Kc), label="independent JKR contacts")


ax.set_ylabel("da")
ax.set_xlabel("contact radius")
ax.set_xlim(left=0)

# -

# Let's compare the two terms in the stiffness of the crack front
#
# The first line is the q=0 mode, and the other is the stiffness of the crack front itself for a single period of work of adhesion fluctuation.
# Since the crack front gets longer and longer, the stiffness of this mode gets softer and softer.

# +
fig, ax = plt.subplots()

a0 = np.linspace(puloff_radius, 10)
penetration = JKR.penetration(contact_radius=a0)
ax.plot(a0, JKR.stress_intensity_factor(a0, penetration, der="1_a") / mean_Kc, label=r"$\frac{\partial K^0 / \partial a}{K^0}$")
ax.plot(a0, 1 / (2*a0), label=r"$\frac{1}{2 a_0}$")

ax.legend()
ax.set_ylabel("stiffness")
ax.set_xlabel("mean contact radius $a_0$")

# -

# In reality, the heterogeneities have a characteristic correlation length that is independent of the correlation.
# Hence, it makes more sense to compare the mode 0 stiffness with the stiffness of the crackfront for a fixed wavelength, rather than a fixed number of periods.
# We can compute the wavelength at which the $q=0$ stiffness equals the "crack-front-stiffness". 
#
#
#
# The corresponding wavelength $L$ is 
#
# $L = \pi \frac{K^0}{\partial K^0 / \partial a}$
#

fig, ax = plt.subplots()
a0 = np.linspace(puloff_radius, 10, 100)[1:]
penetration = JKR.penetration(contact_radius=a0)
ax.plot(a0, np.pi * mean_Kc / JKR.stress_intensity_factor(a0, penetration, der="1_a"))
ax.plot([0,11], 2 * np.pi * np.array([0,11]), c="k", label= "nominal contact perimeter $2 \pi a_0$")
ax.set_xlabel(r"mean contact radius $a_0$")
ax.set_ylabel(r"$\pi \frac{K^0}{\partial K^0 / \partial a}$")
ax.margins(0,0)
ax.legend()
ax.set_ylim(top=20)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(bottom=0.1)
ax.set_xlim(left=puloff_radius)
fig

fig, ax = plt.subplots()
a0 = np.linspace(puloff_radius, 2, 100)[1:]
penetration = JKR.penetration(contact_radius=a0)
ax.plot(penetration, np.pi * mean_Kc / JKR.stress_intensity_factor(a0, penetration, der="1_a"))
ax.plot(penetration, 2 * np.pi * a0, c="k", label= "nominal contact perimeter $2 \pi a_0$")
ax.set_xlabel(r"penetration")
ax.set_ylabel(r"$\pi \frac{K^0}{\partial K^0 / \partial a}$")
ax.margins(0,0)
ax.legend()
ax.set_ylim(top=20)
ax.set_ylim(bottom=0)

#ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(bottom=0.1)
#ax.set_xlim(left=puloff_radius)
fig

# By definition, at the pulloff radius, $\partial K^0 / \partial a$ is zero and hence the mode 0 stiffness completely disappears. Then the limitting length is simply the contact perimeter. 
#
# We see that the $\partial K^0 / \partial a$ contribution is only negligible at penetrations very close to the pulloffs.
#

