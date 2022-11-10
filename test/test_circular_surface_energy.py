import numpy as np

from Adhesion.ReferenceSolutions import JKR
from CrackFront.CircularEnergyReleaseRate import (
    SphereCrackFrontERRPenetrationLin,
    SphereCrackFrontERRPenetrationEnergy,
    SphereCrackFrontERRPenetrationEnergyConstGc, SphereCrackFrontERRPenetrationFull
    )

from CrackFront.Optimization.propagate_sphere_pytorch import LinearInterpolatedPinningFieldUniformFromFileWithEnergy



def test_kink_integral_values():
    n = 8
    w = 1 / np.pi
    Es = 3. / 4

    a = np.linspace

    a = np.linspace(0.1, 1.5).reshape(-1, 1) * np.ones((1, n))
    values =  a * w

    min_radius = a[0,0]
    grid_spacing = a[1, 0] - a[0,0]
    integ = LinearInterpolatedPinningFieldUniformFromFileWithEnergy\
        .compute_integral_values(values, min_radius, grid_spacing)

    np.testing.assert_allclose(integ, a**2 / 2 * w)
