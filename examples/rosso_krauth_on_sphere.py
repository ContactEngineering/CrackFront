import os

import numpy as np
from ContactMechanics.Tools.Logger import screen, Logger
from NuMPI.IO.NetCDF import NCStructuredGrid
from SurfaceTopography.Generation import fourier_synthesis
from matplotlib import pyplot as plt

from CrackFront.CircularEnergyReleaseRate import SphereCrackFrontERRPenetrationEnergyConstGc
from CrackFront.Circular import Interpolator, RadiusTooLowError
from CrackFront.Optimization.RossoKrauth import linear_interpolated_pinning_field_equaly_spaced, brute_rosso_krauth_other_spacing

w=1/np.pi
Es=.75
R=1

# %%

params = {'hurst_exponent': 0,
          'rolloff_wavelength': 0.2,
          'shortcut_wavelength': 0.2,
          'rms': 0.2,
          'seed': 0,
          'penetration_increment': 0.01,
          'max_penetration': 1.0,
          'n_pixels_random': 128,
          'physical_size': 5.12, }
s = params["physical_size"]


# %%

from CrackFront.Optimization import trustregion_newton_cg


def simulate_crack_front_trust_region(
        cf,
        penetrations=np.concatenate((
        np.linspace(0, 1., 200, endpoint=False),
        np.linspace(1., -2., 600)
        )),
        filename="CF.nc",
        pulloff_radius=0.01,
        initial_radius=None,
        trust_radius=0.05,
        dump_fields=True,
        ):
    """

    Parameters:
    -----------
    pulloff_radius: radius at which  the pulloff certainly happend and hence
    the iterations stop

    """
    n = cf.npx

    nc_CF = NCStructuredGrid(filename, "w", (n,))

    penetration = 0

    # initial guess
    if initial_radius is None:
        a = np.ones(n) * pulloff_radius
    elif not hasattr(initial_radius, "len"):
        a = np.ones(n) * initial_radius
    else:
        a = initial_radius

    j = 0

    def trust_radius_from_x(radius):
        if np.max(radius) < pulloff_radius:
                raise RadiusTooLowError
        return np.min((trust_radius, 0.9 * np.min(radius)))

    try:
        for penetration in penetrations:
            print(f"penetration: {penetration}")
            try:
                sol = trustregion_newton_cg(
                    x0=a, gradient=lambda radius: cf.gradient(radius, penetration),
                    #hessian=lambda a: cf.hessian(a, penetration),
                    hessian_product=lambda a, p: cf.hessian_product(p,
                                                                    radius=a,
                                                                    penetration=penetration),
                    trust_radius_from_x=trust_radius_from_x,
                    maxiter=10000,
                    gtol=1e-6  # he has issues to reach the gtol at small values of a
                    )

            except RadiusTooLowError:
                print("lost contact")
                break
            print(sol.message)
            assert sol.success
            print("nit, njev: {}, {}".format(sol.nit, sol.njev))
            a = sol.x
            cf.dump(nc_CF[j], penetration, a, dump_fields)
                    # infos on convergence
            nc_CF[j].nit = sol.nit
            nc_CF[j].n_hits_boundary = sol.n_hits_boundary
            nc_CF[j].njev = sol.njev
            nc_CF[j].nhev = sol.nhev
            nc_CF.sync()
            j = j + 1
    finally:
        nc_CF.close()

def generate_roughness(
    physical_size,
    n_pixels_random,
    rms,
    shortcut_wavelength,
    rolloff_wavelength,
    hurst_exponent,
    seed,**kwargs):
    sx = sy = physical_size

    np.random.seed(seed)
    roughness = fourier_synthesis((n_pixels_random, n_pixels_random), (sx, sx),
             hurst = hurst_exponent,
             c0=1.,
             short_cutoff=shortcut_wavelength,
             long_cutoff=rolloff_wavelength,
             )
    roughness = roughness.scale(rms / roughness.rms_height_from_area())
    return roughness

# %%

topography = generate_roughness(**params).squeeze()
topography._heights += w

# %%

plt.set_cmap('coolwarm')
topography.plot()


# %%

def penetrations(dpen, max_pen):
    i = 0 # integer penetration value
    pen = dpen * i
    yield pen
    while pen < max_pen:
        i += 1
        pen = dpen * i
        yield pen
    while True:
        i -= 1
        pen = dpen * i
        yield pen


n_pixels = 256
interpolator = Interpolator(topography)

cf = SphereCrackFrontERRPenetrationEnergyConstGc(npx=n_pixels,
                                                 w=interpolator.field_polar,
                                                 dw=interpolator.dfield_dr_polar, wm=w)
filename = "trust_region_simulation.nc"
if not os.path.exists(filename):
    simulate_crack_front_trust_region(
            cf,
            penetrations=penetrations(params["penetration_increment"], params["max_penetration"]),
            filename=filename,
            pulloff_radius= (np.pi * w * (1 - 0.4) * R**2 / 6 * 1)**(1/3),
            initial_radius=.5,
            trust_radius=0.1 * 0.1,)


# %% [markdown]
#
# # Using the Rosso-Krauth algorithm
# ## sample the random field on rays

sample_radii = np.linspace(0.1, params["physical_size"] / 2, int(4 * params["physical_size"] / params["shortcut_wavelength"]))

values = - interpolator.field_polar(sample_radii.reshape(1, -1), cf.angles.reshape(-1, 1))

pinning_field = linear_interpolated_pinning_field_equaly_spaced(values * sample_radii * 2 * np.pi / cf.npx , sample_radii)

# %%

cf.pinning_field = pinning_field

filename = "KR.nc"
nc_CF = NCStructuredGrid(filename, "w", (n_pixels,))

penetration_prev = - 10
a = np.ones(n_pixels) * .1
for j, penetration in enumerate(penetrations(params["penetration_increment"], params["max_penetration"])):
    print(penetration)
    try:
        sol = cf.rosso_krauth(a, penetration, gtol=1e-6, maxit = 10000000, dir = 1 if penetration > penetration_prev else -1, logger=Logger(outevery=100) )
    except RadiusTooLowError:
        print("lost contact")
        break
    assert sol.success
    a = sol.x
    penetration_prev = penetration
    cf.dump(nc_CF[j], penetration, a, False)
    nc_CF[j].nit = sol.nit
    nc_CF.sync()


# %%


# %%

nc = NCStructuredGrid("trust_region_simulation.nc")

fig, ax = plt.subplots()
ax.plot(nc.penetration, nc.force, c="k", label="TR simulation")
nc = NCStructuredGrid("KR.nc")
ax.plot(nc.penetration, nc.force, "+-b", label="RK")

ax.legend()

# %% [markdown]
#
# There are some little differences that are due to the linear versus bicubic interpolation
#
# This should however disappear if I increase the number of collocation points.
#
