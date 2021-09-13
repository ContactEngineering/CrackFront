# %% [markdown]
# # Sphere against one dimensional waviness
#
# Comparing the crack front and BEM for the contact

# %%
import matplotlib as mpl
import scipy as sp
from Adhesion.Interactions import PowerLaw
from Adhesion.ReferenceSolutions import JKR
from Adhesion.System import BoundedSmoothContactSystem
from ContactMechanics import FreeFFTElasticHalfSpace
from ContactMechanics.Tools.Logger import Logger, screen
from CrackFront.Circular import Interpolator, pol2cart
from CrackFront.CircularEnergyReleaseRate import SphereCrackFrontERRPenetrationEnergyConstGc
from CrackFront.Optimization import trustregion_newton_cg
from matplotlib.colors import LinearSegmentedColormap
from scipy import interpolate

from CrackFront.Roughness import circular_crack_sif_from_roughness
from SurfaceTopography import Topography, make_sphere
import scipy.interpolate

import numpy as np
import matplotlib.pyplot as plt

import pytest

# %% [markdown]
# Nondimensionalisation

# %%
R = 1
Es = 3/4
w = 1 / np.pi
Kc = np.sqrt(2 * Es * w)

# %% [markdown]
# System size and discretization for the crack front

# %%
sx = 5
sy = 5

nx = 128

sphere = make_sphere(R, (nx, nx), (sx, sy),
        centre=(sx / 2, sy / 2), kind="paraboloid") # This topo is just used to generate pixel positions

x, y = sphere.positions()

x -= sx / 2
y -= sy /2
radius = np.sqrt(x ** 2 + y ** 2)

sinewave_period = 1.
roughness_amplitude = 0.1 * sinewave_period

roughness = Topography(roughness_amplitude * np.cos(2 * np.pi * y / sinewave_period), physical_sizes=(sx, sy))

n_angles = 128
n_radii = 128
cf_angles = (np.arange(n_angles) / n_angles * 2 * np.pi).reshape(-1, 1)
cf_radii = np.linspace(0.2, 2.5, n_radii).reshape(1, -1)

SIF = circular_crack_sif_from_roughness(roughness, cf_radii, cf_angles, Es=Es)

# %% [markdown]
# How the SIF looks like

# %%
from mpl_toolkits.mplot3d import Axes3D

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes
axes[0].set_title("waviness")
axes[1].set_title("SIF, real")
axes[2].set_title("SIF,imag")

for a in axes:
        a.set_aspect(1)

plt.colorbar(axes[0].pcolormesh(x, y, roughness.heights()), ax=axes[0])
plt.colorbar(axes[1].pcolormesh(*pol2cart(cf_radii, cf_angles), SIF.real), ax=axes[1])
plt.colorbar(axes[2].pcolormesh(*pol2cart(cf_radii, cf_angles), SIF.imag), ax=axes[2])

fig.show()

assert np.max(abs(SIF.imag)) < 1e-10, np.max(abs(SIF.imag))

SIF = SIF.real
sif_from_roughness = SIF

print(f"expected amplitude of SIF: {Es *  roughness_amplitude * np.sqrt(np.pi / sinewave_period) }")

# %% [markdown]
# Compute the effective work of adhesion from the stress intensity factor

# %%
w_field = (Kc + SIF) ** 2 / (2 * Es)

# interpolate. 
# one 1D-spline for each angle
splines = [interpolate.InterpolatedUnivariateSpline(cf_radii.reshape(-1), w_field[i, :]) for i in range(len(cf_angles)) ]

def interpolated_w(radius, angle):
    angle_index = np.array(angle / (2 * np.pi / n_angles), dtype=int)
    return np.array([splines[i](radius[i]) for i in angle_index])

def interpolated_dw(radius, angle):
    angle_index = np.array(angle / (2 * np.pi / n_angles), dtype=int)
    return np.array([splines[i].derivative(1)(radius[i]) for i in angle_index])


cf = SphereCrackFrontERRPenetrationEnergyConstGc(npx=n_angles,
                                                 w=interpolated_w,
                                                 dw=interpolated_dw)

# %% [markdown]
# ## Comparison of contact shapes

# %%
penetration = 0.5

# %% [markdown]
# ### Crack front

# %% Solve for the crack shape

sol = trustregion_newton_cg(
    x0=np.ones(n_angles) * JKR.contact_radius(penetration=penetration),
    gradient=lambda radius: cf.gradient(radius, penetration),
    hessian_product=lambda a, p: cf.hessian_product(p,
                                                    radius=a,
                                                    penetration=penetration),
    gtol=1e-6,
 )
assert sol.success

radii = sol.x

fig, ax = plt.subplots()
#plt.colorbar(ax.pcolormesh(x, y, roughness.heights()))
plt.colorbar(plt.pcolormesh(*pol2cart(cf_radii, cf_angles), w_field), label="Effective work of adhesion")
ax.plot(*pol2cart(radii, cf_angles.reshape(-1)))
ax.set_aspect(1)

# %% [markdown]
# ### BEM
#
# we use a finer discretisation in order to use reasonably short interaction ranges.

# %% BEM reference simulation
sx = 4.8
sy = 4.8

nx = 512

interaction_range = 0.16

sphere = make_sphere(R, (nx, nx), (sx, sy),
        centre=(sx / 2, sy / 2), kind="paraboloid")

x, y = sphere.positions()

x -= sx / 2
y -= sy /2
radius = np.sqrt(x ** 2 + y ** 2)

roughness = Topography(roughness_amplitude * np.cos(2 * np.pi * y / sinewave_period), physical_sizes=(sx, sy))

substrate = FreeFFTElasticHalfSpace((nx, nx), Es, (sx, sy),
                            check_boundaries=False,)

print("substrate initialized")
topography = make_sphere(R,
    nb_grid_pts=(nx, nx),
    physical_sizes=(sx, sy),
    nb_subdomain_grid_pts=substrate.topography_nb_subdomain_grid_pts,
    subdomain_locations=substrate.topography_subdomain_locations,
    centre=(sx / 2, sy / 2),
    kind="paraboloid",)

combined_topography = Topography((topography.heights() + roughness.heights()), physical_sizes=topography.physical_sizes)



interaction = PowerLaw(w,
                       # v that way the max stress is still w / rho
                       3 * interaction_range,
                       3,
                       )



system = BoundedSmoothContactSystem(substrate=substrate,
                                    interaction=interaction,
                                    surface=combined_topography)

gtol = 1e-6

sol = system.minimize_proxy(
    #initial_displacements=u_initial_guess,
    lbounds="auto",
    options=dict(gtol=gtol * (topography.area_per_pt * abs(interaction.max_tensile)),
                 ftol=np.finfo(float).eps * 10, maxcor=3,
                 maxiter=1000,
                 maxfun=3000,
                 ),
    logger=Logger("evaluations.log"),
    #logger=screen,
    offset=penetration,
)

u = system.displacements = sol.x
system.compute_gap(u, penetration)


# %% [markdown]
# ## Comparing crack shapes

# %%

force = - system.substrate.evaluate_force(u)
pressures = force[system.substrate.local_topography_subdomain_slices] \
            / system.surface.area_per_pt


class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return sp.ma.masked_array(sp.interp(value, x, y))

    def inverse(self, value):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        return np.interp(value, [normalized_min, normalized_mid,
                                 normalized_max], [self.vmin, self.midpoint, self.vmax])

max_stress = np.abs(interaction.max_tensile)
pnorm = MidpointNormalize(vmin=-max_stress, vmax=max_stress)
pressurecmap = LinearSegmentedColormap.from_list('testCmap', (
    (0.15294117647058825, 0.39215686274509803, 0.09803921568627451, 1.),
    (1, 1, 1, 0.),
    (0.5568627450980392, 0.00392156862745098, 0.3215686274509804, 1)), N=256)

fig, ax = plt.subplots()

plt.colorbar(ax.imshow(roughness.heights().T,
             cmap="coolwarm", extent=(-sx / 2, sx / 2, -sy / 2, sy / 2), rasterized=True),
             ax=ax, label=r"Heights $h^*$")
ax.imshow(pressures.T, cmap=pressurecmap, extent=(-sx / 2, sx / 2, -sy / 2, sy / 2), rasterized=True)

ax.plot(*pol2cart(radii, cf_angles.reshape(-1)), c="pink")

ax.set_xlabel("$x^*$")
ax.set_ylabel("$y^*$")

# %% compare [markdown]
#
