
import numpy as np
import matplotlib.pyplot as plt

from CrackFront.Roughness import (
    straight_crack_sif_from_roughness,
    circular_crack_sif_from_roughness,
    circular_crack_sif_from_roughness_memory_friendly
    )
from SurfaceTopography import Topography

import pytest


@pytest.mark.parametrize("s", (1., 2., 3., 4.))
@pytest.mark.parametrize("Es", (1., 2.))
def test_straight_crack_sif_from_roughness_tangential(s, Es):
    sx = sy = s
    nx, ny = 128, 128

    dx = sx / nx

    x, y = np.mgrid[:nx, :ny] * dx

    roughness = Topography(np.cos(2 * np.pi * y / sy), (sx, sy))

    K = straight_crack_sif_from_roughness(roughness, Es=Es)

    K_simple_analytical = - Es * np.sqrt(np.pi / sy) * np.cos(
        2 * np.pi * y / sy)

    # fig, ax = plt.subplots()
    # ax.plot(K[0, :], label="fourier") # factor of pi  too big for wavelength 1.
    # ax.plot(K_simple_analytical[0, :], label="analytical")
    # ax.legend()
    # plt.show()

    np.testing.assert_allclose(K, K_simple_analytical, atol=1e-13)


@pytest.mark.parametrize("s", (1., 2., 3., 4.))
def test_straight_crack_sif_from_roughness_perpendicular(s):
    Es = 1.
    sx = sy = s
    nx, ny = 128, 128

    dx = sx / nx

    x, y = np.mgrid[:nx, :ny] * dx

    roughness = Topography(np.cos(2 * np.pi * x / sx), (sx, sy))

    K = straight_crack_sif_from_roughness(roughness, Es=Es)

    K_simple_analytical = Es * np.sqrt(np.pi / sx) * np.cos(
        2 * np.pi * x / sx - 3 * np.pi / 4)

    # fig, ax = plt.subplots()
    # ax.plot(K[0, :], label="fourier") # factor of pi  too big for wavelength 1.
    # ax.plot(K_simple_analytical[0, :], label="analytical")
    # ax.legend()
    # plt.show()

    np.testing.assert_allclose(K, K_simple_analytical, atol=1e-13)


@pytest.mark.parametrize("circular_crack_sif_from_roughness",
                         [
                             circular_crack_sif_from_roughness,
                             circular_crack_sif_from_roughness_memory_friendly
                             ])
def test_circular_waviness_amplitude(circular_crack_sif_from_roughness):
    Es = 0.75
    R = 1
    w = 1 / np.pi

    # cartesian grid on which we define the waviness
    nx = ny = 128

    sx = sy = 5

    x = np.arange(nx) * sx / nx
    y = np.arange(ny) * sy / ny

    x -= sx / 2
    y -= sy / 2

    y, x = np.meshgrid(y, x)

    sinewave_period = 1.
    roughness_amplitude = 0.1 * sinewave_period

    roughness = Topography(roughness_amplitude * np.cos(2 * np.pi * x / sinewave_period), physical_sizes=(sx, sy))

    n_radii = 32
    cf_angles = np.array((0,), ).reshape((-1, 1))
    cf_radii = np.linspace(0.2, 2.5, n_radii).reshape(1, -1)

    sif = circular_crack_sif_from_roughness(roughness, cf_radii, cf_angles, Es=Es)

    sif_expected = Es * roughness_amplitude * np.sqrt(np.pi / sinewave_period) \
                   * np.cos(2 * np.pi * cf_radii / sinewave_period - 3 * np.pi / 4)

    if False:
        fig, ax = plt.subplots()
        ax.plot(cf_radii.reshape(-1), sif.reshape(-1), label="actual")
        ax.plot(cf_radii.reshape(-1), sif_expected.reshape(-1), label="expected")
        ax.legend()
        plt.show()

    np.testing.assert_allclose(sif, sif_expected, rtol=1e-12, atol=1e-15)

    # Other orientation

    cf_angles = np.array((np.pi / 2,), ).reshape((-1, 1))

    sif = circular_crack_sif_from_roughness(roughness, cf_radii, cf_angles, Es=Es)
    sif_expected = - Es * roughness_amplitude * np.sqrt(np.pi / sinewave_period)

    np.testing.assert_allclose(sif, sif_expected, rtol=1e-12, atol=1e-15)
