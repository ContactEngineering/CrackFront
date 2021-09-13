from CrackFront.Roughness import straight_crack_sif_from_roughness, circular_crack_sif_from_roughness
from SurfaceTopography import Topography

import numpy as np
import matplotlib.pyplot as plt

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

# %%
#
# TODO
# at 0 and 90° the amplitudes and phases of the sinewaves should match the corresponding straight crack front calculation

@pytest.mark.skip("Not Implemented")
def test_circular_waviness_amplitude():
    pass
