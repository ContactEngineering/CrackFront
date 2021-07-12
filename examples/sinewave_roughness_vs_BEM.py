import numpy as np
from Adhesion.Interactions import PowerLaw
from Adhesion.System import BoundedSmoothContactSystem
from ContactMechanics import PeriodicFFTElasticHalfSpace
from ContactMechanics.Tools.Logger import Logger
from SurfaceTopography import Topography

import matplotlib as mpl
import scipy as sp

from CrackFront.Optimization import trustregion_newton_cg
from CrackFront.StraightForRoughness import SinewaveCrackFrontLoadEnergyConstK


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


# nondimensional units
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

Es = 1 / np.pi
h = 1.  # amplitude (half peak to valley) of the sinewave
sinewave_lambda = 1.
sx = 1.

# discretisation
nx = 16384
ny = 256
dx = sx / nx
sy = dx * ny

amplitude_roughness = 0.00625

Kc = 0.1
work_of_adhesion = Kc ** 2 / (2 * Es)
interaction_fac = 3  # factor of safety for the interaction range. On the sphere I found empirically that 2 is sufficient.


def heights_roughness(x, z):
    return amplitude_roughness * np.cos(z * 2 * np.pi / sy)


def kr(a, y, der="0"):
    if der == "0":
        return - Es * amplitude_roughness * np.sqrt(np.pi / sy) * np.cos(y * 2 * np.pi / sy)
    else:
        return np.zeros_like(y)


def make_topography_from_function(fun, physical_sizes,
                                  nb_grid_pts=None,
                                  subdomain_locations=None,
                                  nb_subdomain_grid_pts=None,
                                  origin=(0, 0),
                                  **kwargs
                                  ):
    """
    See also Topography

    Parameters
    ----------
    fun: callable(X, Y)
    origin: tuple or "center"
        (cx, cy)
        h(X,Y) =  fun(X-cx, Y-cy)
        default (0,0)
    physical_sizes
    subdomain_locations
    nb_subdomain_grid_pts
    nb_grid_pts

    Other Parameters
    ----------------

    other kwargs for PyCo.Topography

    Returns
    -------

    SurfaceTopography.topography

    """
    if isinstance(origin, str) and origin == "center":
        origin = tuple([s / 2 for s in physical_sizes])

    if nb_subdomain_grid_pts is None:
        # serial code
        nb_subdomain_grid_pts = nb_grid_pts
        nb_grid_pts = None
    topography = Topography(heights=np.zeros(nb_subdomain_grid_pts),
                            physical_sizes=physical_sizes,
                            subdomain_locations=subdomain_locations,
                            nb_grid_pts=nb_grid_pts,
                            **kwargs,
                            )

    cx, cy = origin
    X, Y = topography.positions()

    topography._heights = fun(X - cx, Y - cy)

    return topography


# %%

halfspace = PeriodicFFTElasticHalfSpace((nx, ny), Es, (sx, sy), )

topography = make_topography_from_function(
    lambda x,y: (-1 + np.cos(2 * np.pi * x / sx)) + heights_roughness(x, y),
    (sx, sy),
    # subdomain_locations=halfspace.topography_subdomain_locations,
    # nb_subdomain_grid_pts=halfspace.topography_nb_subdomain_grid_pts,
    nb_grid_pts=(nx, ny),
    periodic=True,
    origin=(sx / 2, 0)
    )

interaction = PowerLaw(work_of_adhesion, 3 * interaction_fac * np.sqrt(work_of_adhesion / Es * dx))
system = BoundedSmoothContactSystem(halfspace, interaction, topography)

displacement = 0.1
gtol = 1e-5
sol = system.minimize_proxy(
    # disp0=disp0,
    lbounds="auto",
    options=dict(gtol=gtol * abs(interaction.max_tensile) * system.surface.area_per_pt,
                 ftol=0, maxcor=3, maxiter=80000),
    logger=Logger("laststep.log"),
    offset=displacement,
    callback=None
    )
assert sol.success
u = disp0 = sol.x
mean_deformation = np.sum(u) / np.prod(halfspace.nb_domain_grid_pts)

force = - halfspace.evaluate_force(u)
#
#
contacting_points = np.where(system.gap == 0., 1, 0)
pressures = force[halfspace.local_topography_subdomain_slices] / topography.area_per_pt
#
#
contact_area = np.sum(contacting_points) * halfspace.area_per_pt
fractional_contact_area = contact_area / (sx * sy)
normal_force = force.sum()
mean_pressure = normal_force / np.prod(topography.physical_sizes)
# ncfile[i].displacement = displacement
# ncfile[i].mean_deformation = mean_deformation
# ncfile[i].elastic_energy = elastic_energy = system.substrate.energy
# ncfile[i].interaction_energy = interaction_energy = system.interaction.energy
# ncfile[i].energy = energy = system.energy

n = 32
def kr_left(a, y, der="0"):
    return kr(-a, y, der)
cf = SinewaveCrackFrontLoadEnergyConstK(n, sy, kr, kr_left, w=work_of_adhesion)

a = np.ones(2 * n) * 0.25

sol = trustregion_newton_cg(
    x0=a,
    gradient=lambda a: cf.gradient(a, mean_pressure),
    hessian_product=lambda a, p: cf.hessian_product(p, a, mean_pressure),
    trust_radius=0.25 * np.min(a),
    maxiter=3000,
    gtol=1e-11)

al = sol.x[:n]
ar = sol.x[n:]

fig, ax = plt.subplots()

x, y = topography.positions()
x -= 0.5

max_stress = np.abs(interaction.max_tensile)
pnorm = MidpointNormalize(vmin=- max_stress, vmax=max_stress)
pressurecmap = LinearSegmentedColormap.from_list('testCmap', (
    (0.15294117647058825, 0.39215686274509803, 0.09803921568627451, 1.),
    (1, 1, 1, 0.6),
    (0.5568627450980392, 0.00392156862745098, 0.3215686274509804, 1)), N=256)

pnorm = MidpointNormalize(vmin=- max_stress, vmax=max_stress)

heightcmap = plt.get_cmap("coolwarm")
ax.imshow(heights_roughness(x, y).T, cmap=heightcmap, rasterized=True)
plt.colorbar(ax.imshow(pressures.T, norm=pnorm, cmap=pressurecmap, rasterized=True))
#ax.imshow(contacting_points.T)
#ax.imshow(system.gap.T)

ax.invert_yaxis()

ticks = np.linspace(-sx / 2, (sx / 2), 10)

ax.set_xticks(ticks / sx * nx + nx * 0.5)
ax.set_xticklabels([f"{v:.2f}" for v in ticks])

ticks = np.linspace(-sy / 2, sy / 2, 5)
ax.set_yticks(ticks / sy * ny + ny * 0.5)
ax.set_yticklabels([f"{v:.2f}" for v in ticks])

y_cf = np.arange(len(al)) * sy / len(al)
ax.plot((0.5 - al) / sx * nx, y_cf  / sy * ny, "--k")
ax.plot((0.5 + ar) / sx * nx, y_cf / sy * ny, "--k")

ax.set_xlim((0.5 - 1.2 * np.max(al)) / sx * nx, (0.5 + 1.2 * np.max(ar)) / sx * nx)
fig.savefig(f"{nx}x{ny}_Kc{Kc}_hr{amplitude_roughness}_inter{interaction_fac}.svg")