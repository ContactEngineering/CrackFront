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
import threading
import signal
import os
import time

import torch
from ContactMechanics.Tools.Logger import Logger
from NuMPI.IO.NetCDF import NCStructuredGrid
from Adhesion.ReferenceSolutions import JKR

from CrackFront.Circular import Interpolator
from CrackFront.CircularEnergyReleaseRate import SphereCFPenetrationEnergyConstGcPiecewiseLinearField
from CrackFront.Optimization.RossoKrauth import linear_interpolated_pinning_field_equaly_spaced
from CrackFront.Optimization.propagate_sphere_pytorch import (
    propagate_rosso_krauth,
    LinearInterpolatedPinningFieldUniformFromFile
    )
from CrackFront.CircularEnergyReleaseRate import Es, w, R, maugis_K, generate_random_work_of_adhesion


def test_JKR_curve():
    npx_front = 100
    # npx_front can't be too low because there is a nonlinear part in the elasticity (coming from the JKR solution)
    # that we ignore
    #
    # Having more pixels increases the stiffness associated with moving one pixel at a time, so we make more
    # careful steps where the nonlinearity is not a problem

    sample_radii = np.linspace(0.1, 2.5, 20)

    values = w * np.ones((len(sample_radii), npx_front))

    w_radius_values = np.ascontiguousarray(values * sample_radii.reshape(-1, 1) * 2 * np.pi / npx_front)

    grid_spacing = sample_radii[1] - sample_radii[0]
    LinearInterpolatedPinningFieldUniformFromFile.save_values_and_slopes_to_file(
        values=w_radius_values,
        grid_spacing=grid_spacing,
        filename="values_and_slopes.npy")

    cf = SphereCFPenetrationEnergyConstGcPiecewiseLinearField(
        piecewise_linear_w_radius=LinearInterpolatedPinningFieldUniformFromFile(
            filename="values_and_slopes.npy",
            min_radius=sample_radii[0],
            grid_spacing=grid_spacing,
            accelerator=torch.device("cpu")
            ),
        wm=w)

    propagate_rosso_krauth(
        cf,
        penetration_increment=0.1,
        max_penetration=0.5,
        initial_a=np.ones(cf.npx) * cf.piecewise_linear_w_radius.kinks[0],
        gtol=1e-10,
        dump_fields=False,
        maxit=10000,
        filename="RK_torch.nc",
        logger=Logger("RK_pytorch.log", outevery=1),
        )

    nc = NCStructuredGrid("RK_torch.nc")
    np.testing.assert_allclose(nc.mean_radius,
                               [JKR.contact_radius(penetration=nc.penetration[i]) for i in range(len(nc))],
                               rtol=1e-06)


def test_restart():
    params = dict(
        # pixel_size_radial=0.1,
        n_pixels_front=512,
        rms=.5,
        max_penetration=1.,
        penetration_increment=0.2,
        shortcut_wavelength=0.08,
        # randomness:
        seed=0,
        # numerics:
        gtol=1e-8,
        maxit=10000,
        n_pixels=256,
        # n_pixels_fourier_interpolation=128,
        pixel_size=0.02,
        )
    npx_front = params["n_pixels_front"]
    assert params["shortcut_wavelength"] > 2 * params["pixel_size"]
    params.update(dict(pixel_size_radial=params["shortcut_wavelength"] / 16))
    pulloff_radius = (np.pi * w * R ** 2 / 6 * maugis_K) ** (1 / 3)

    minimum_radius = pulloff_radius / 10

    # maximum radius
    physical_sizes = params["pixel_size"] * params["n_pixels"]

    maximum_radius = physical_sizes / 2

    n_pixels_radial = np.floor((maximum_radius - minimum_radius) / params["pixel_size_radial"])

    sample_radii = np.arange(n_pixels_radial) * params["pixel_size_radial"] + minimum_radius
    cf_angles = np.arange(params["n_pixels_front"]) * 2 * np.pi / npx_front

    w_topography = generate_random_work_of_adhesion(**params)

    interpolator = Interpolator(w_topography)
    w_radius_values = np.ascontiguousarray(interpolator.field_polar(sample_radii.reshape(-1, 1), cf_angles.reshape(1, -1)) \
                      * sample_radii.reshape(-1, 1) * 2 * np.pi / npx_front)

    LinearInterpolatedPinningFieldUniformFromFile.save_values_and_slopes_to_file(
        w_radius_values,
        params["pixel_size_radial"],
        filename="values_and_slopes.npy")

    cf = SphereCFPenetrationEnergyConstGcPiecewiseLinearField(
        piecewise_linear_w_radius=LinearInterpolatedPinningFieldUniformFromFile(
            filename="values_and_slopes.npy",
            min_radius=sample_radii[0],
            grid_spacing=params["pixel_size_radial"],
            accelerator=torch.device("cpu")
            ),
        wm=w)

    propagate_rosso_krauth(
        cf,
        initial_a=np.ones(cf.npx) * cf.piecewise_linear_w_radius.min_radius,
        dump_fields=False,
        filename="uninterupted.nc",
        **params,
    )

    pid = os.getpid()

    # https://stackoverflow.com/questions/26158373/how-to-really-test-signal-handling-in-python
    def trigger_signal():
        # You could do something more robust, e.g. wait until port is listening
        time.sleep(0.5)
        os.kill(pid, signal.SIGUSR1)

    thread = threading.Thread(target=trigger_signal)
    thread.daemon = True
    thread.start()

    # simulate until interrupted by signal
    assert not propagate_rosso_krauth(
        cf,
        initial_a=np.ones(cf.npx) * cf.piecewise_linear_w_radius.min_radius,
        dump_fields=False,
        filename="interupted.nc",
        handle_signals=True,
        **params,
    ), "Simulation finished successfully before being interupted"

    # I send the signal several times, so I am sure it works for both directions.
    not_yet_finished = True
    while not_yet_finished:
        # plan to send the signal in a second
        thread = threading.Thread(target=trigger_signal)
        thread.daemon = True
        thread.start()

        # restart and finish simulation
        not_yet_finished = not propagate_rosso_krauth(
            cf,
            dump_fields=False,
            initial_a=np.load("restart_position.npy"),
            filename="interupted.nc",
            handle_signals=True,
            restart=True,
            **params,
            )

    # Now we assert the result was unaffected by the restart
    nc_uninterupted = NCStructuredGrid("uninterupted.nc")
    nc_interupted = NCStructuredGrid("interupted.nc")

    np.testing.assert_allclose(nc_interupted.mean_radius, nc_uninterupted.mean_radius)
    # np.testing.assert_allclose(nc_interupted.position, nc_uninterupted.position, atol=1e-5)
    # 1e-5 is already very small compared to the heterogeneity spacing
    np.testing.assert_allclose(nc_interupted.rms_radius, nc_uninterupted.rms_radius, )
    np.testing.assert_allclose(nc_interupted.force, nc_uninterupted.force, )
    np.testing.assert_allclose(nc_interupted.nit, nc_uninterupted.nit, )
