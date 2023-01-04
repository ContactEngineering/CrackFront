#
# Copyright 2021 Antoine Sanner
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
import numpy as np
import pytest
from Adhesion.ReferenceSolutions import JKR
from ContactMechanics.Tools.Logger import Logger
from NuMPI.IO.NetCDF import NCStructuredGrid
from SurfaceTopography import Topography

from CrackFront.Circular import Interpolator
from CrackFront.CircularEnergyReleaseRate import (
    SphereCrackFrontERRPenetrationEnergyConstGc,
    SphereCFPenetrationEnergyConstGcPiecewiseLinearField, SphereCrackFrontERRPenetrationEnergy,
    SphereCrackFrontERRPenetrationLin
    )
from CrackFront.Optimization.RossoKrauth import linear_interpolated_pinning_field_equaly_spaced
from CrackFront.Optimization.propagate_sphere_trust_region import penetrations_generator, simulate_crack_front

# nondimensional units following Maugis Book:
Es = 3 / 4
w = 1 / np.pi
R = 1.
maugis_K = 1.
mean_Kc = np.sqrt(2 * Es * w)


def test_propagate_sphere_trust_region_vs_JKR():
    npx = 64

    cf = SphereCrackFrontERRPenetrationEnergyConstGc(npx,
                                                     w=lambda radius, angle: np.ones_like(radius) * w,
                                                     dw=lambda radius, angle: np.zeros_like(radius) * w, )

    pulloff_radius = (np.pi * w * R ** 2 / 6 * maugis_K) ** (1 / 3)

    simulate_crack_front(
        cf,
        penetrations=penetrations_generator(0.1, 1.),
        filename="trust.nc",
        pulloff_radius=pulloff_radius * 0.5,
        initial_radius=JKR.contact_radius(penetration=0, work_of_adhesion=w) * 0.5,
        trust_radius=0.5,
        dump_fields=False,
        gtol=1e-08
        )

    nc = NCStructuredGrid("trust.nc")
    np.testing.assert_allclose(nc.force, [JKR.force(penetration=p) for p in nc.penetration[:]], rtol=1e-06)
    np.testing.assert_allclose(nc.mean_radius,
                               [JKR.contact_radius(penetration=p) for p in nc.penetration[:]],
                               rtol=1e-06)


def make_w_field_interpolator(params_w_field):
    dw = params_w_field["rms"]
    highcut_wavelength = params_w_field["shortcut_wavelength"]
    sx = sy = params_w_field["physical_size"]
    # resolution of the fourier interpolated grid
    nx = ny = params_w_field["n_pixels_fourier_interpolation"]

    # coarsest grid that can fit the whole nonzero spectrum
    # we generate the field on this grid and subsequently fourier_interpolate
    # on the finer grid.
    # In this way, the random spectrum is independent of the fourier
    # interpolation grid
    nx_sub = ny_sub = params_w_field["n_pixels_random"]

    assert highcut_wavelength > 2 * sx / nx_sub

    np.random.seed(params_w_field["seed"])
    w_fluct_topo = Topography(
        np.random.uniform(size=(nx_sub, ny_sub)),
        physical_sizes=(sx, sy),
        periodic=True
        ).shortcut(cutoff_wavelength=highcut_wavelength
                   ).detrend("center"
                             ).interpolate_fourier((nx, ny))

    w_topo = Topography((w_fluct_topo.scale(dw / w_fluct_topo.rms_height_from_area()
                                            ).heights()
                         + 1) * w,
                        w_fluct_topo.physical_sizes, periodic=True)
    w_topo_interpolator = Interpolator(w_topo)
    return w_topo_interpolator


@pytest.mark.parametrize("cf_class", [
    SphereCrackFrontERRPenetrationEnergy,
    pytest.param(SphereCrackFrontERRPenetrationLin, marks=pytest.mark.skip("TODO: update API")),
    SphereCrackFrontERRPenetrationEnergyConstGc])
def test_propagate_sphere_trust_region_random_spline(cf_class):
    """
    Just asserts that it is converging at all
    """
    params = dict(
        pixel_size_radial=0.1,  # starting from 8000 pix numpy starts to slower then cuda
        n_pixels_front=256,  # 64
        n_pixels_random=128,
        n_pixels_fourier_interpolation=128,
        physical_size=5.12,
        shortcut_wavelength=0.1,
        rms=.5,
        max_penetration=1.,
        penetration_increment=0.01,
        # randomness:
        seed=0,
        # numerics:
        gtol=1e-8,
        maxit=10000000,
        )

    interpolator = make_w_field_interpolator(params)

    dw = params["rms"]
    shortcut_wavelength = params["shortcut_wavelength"]

    pulloff_radius = (np.pi * w * (1 - dw) * R ** 2 / 6 * maugis_K) ** (1 / 3)

    cf = cf_class(npx=params["n_pixels_front"],
                  w=interpolator.field_polar,
                  dw=interpolator.dfield_dr_polar)


    simulate_crack_front(
        cf,
        penetrations=penetrations_generator(params["penetration_increment"], params["max_penetration"]),
        filename="trust.nc",
        pulloff_radius=pulloff_radius * 0.5,
        initial_radius=JKR.contact_radius(penetration=0, work_of_adhesion=w * (1 - dw)) * 0.5,
        trust_radius=0.05 * shortcut_wavelength,
        dump_fields=False,
        gtol=1e-08,
        logger=Logger("trust.log")
        )

    # nc = NCStructuredGrid("trust.nc")
    # np.testing.assert_allclose(nc.force, [JKR.force(penetration=p) for p in nc.penetration[:]], rtol=1e-06)
    # np.testing.assert_allclose(nc.mean_radius,
    #                           [JKR.contact_radius(penetration=p) for p in nc.penetration[:]],
    #                           rtol=1e-06)

    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(nc.penetration, nc.force)

        a = np.linspace(0, 2)
        ax.plot(JKR.penetration(contact_radius=a), JKR.force(contact_radius=a), "--k")
        plt.show(block=True)

@pytest.mark.skip("just plotting")
def test_linear_interpolated_pinning_field_derivative():
    npx_front = 16
    n_radii = 100

    sample_radii = np.linspace(0.5, 6, n_radii)
    w_values = np.ones((npx_front, 1)) * sample_radii.reshape(1, -1)
    w_values = np.random.normal(size=(npx_front, n_radii))

    piecewise_linear_w = linear_interpolated_pinning_field_equaly_spaced(
        w_values * sample_radii * 2 * np.pi / npx_front, sample_radii)
    cf = SphereCFPenetrationEnergyConstGcPiecewiseLinearField(piecewise_linear_w)

    a = 3 + np.random.uniform(0, 1, size=npx_front)
    da = np.random.uniform(0, 1, size=npx_front)

    w = cf.piecewise_linear_w(a)

    if True:
        hs = np.array([1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5,
                       1e-6, 1e-7])
        rms_errors = []
        for h in hs:
            dw = cf.piecewise_linear_w(a + h * da) - w
            dw_from_derivative = cf.piecewise_linear_w(a, der="1") * h * da
            rms_errors.append(np.sqrt(np.mean((dw_from_derivative - dw) ** 2)))

        # Visualize the quadratic convergence of the taylor expansion
        # What to expect:
        # Taylor expansion: g(x + h ∆x) - g(x) = Hessian * h * ∆x + O(h^2)
        # We should see quadratic convergence as long as h^2 > g epsmach,
        # the precision with which we are able to determine ∆g.
        # What is the precision with which the hessian product is made ?
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(hs, rms_errors, "+-")
        print(rms_errors)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True)
        plt.show()


def test_linear_interpolated_pinning_field_vs_JKR():
    """
    Here again we simply compute the JKR contact,
    but we check that the linearly interpolated pinning field is not buggy
    """

    npx_front = 64
    n_radii = 23

    pulloff_radius = (np.pi * w * R ** 2 / 6 * maugis_K) ** (1 / 3)

    sample_radii = np.linspace(0.5 * pulloff_radius, 6, n_radii)
    w_values = np.ones((npx_front, n_radii)) * w
    piecewise_linear_w = linear_interpolated_pinning_field_equaly_spaced(
        w_values * sample_radii * 2 * np.pi / npx_front, sample_radii)
    cf = SphereCFPenetrationEnergyConstGcPiecewiseLinearField(piecewise_linear_w, wm=w)

    simulate_crack_front(
        cf,
        penetrations=penetrations_generator(0.1, 1.),
        filename="trust.nc",
        pulloff_radius=pulloff_radius * 0.5,
        initial_radius=JKR.contact_radius(penetration=0, work_of_adhesion=w) * 0.5,
        trust_radius=0.5,
        dump_fields=False,
        gtol=1e-08
        )

    nc = NCStructuredGrid("trust.nc")
    np.testing.assert_allclose(nc.force, [JKR.force(penetration=p) for p in nc.penetration[:]], rtol=1e-06)
    np.testing.assert_allclose(nc.mean_radius,
                               [JKR.contact_radius(penetration=p) for p in nc.penetration[:]],
                               rtol=1e-06)
