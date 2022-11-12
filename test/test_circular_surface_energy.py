import numpy as np
import torch

from Adhesion.ReferenceSolutions import JKR
from CrackFront.CircularEnergyReleaseRate import (
    SphereCrackFrontERRPenetrationLin,
    SphereCrackFrontERRPenetrationEnergy,
    SphereCrackFrontERRPenetrationEnergyConstGc, SphereCrackFrontERRPenetrationFull
    )

from CrackFront.Optimization.propagate_sphere_pytorch import LinearInterpolatedPinningFieldUniformFromFile



def test_kink_integral_values():
    n = 8
    w = 1 / np.pi
    Es = 3. / 4

    a = np.linspace

    a = np.linspace(0.1, 1.5).reshape(-1, 1) * np.ones((1, n))
    values =  a * w

    min_radius = a[0,0]
    grid_spacing = a[1, 0] - a[0,0]
    integ = LinearInterpolatedPinningFieldUniformFromFile\
        .compute_integral_values(values, min_radius, grid_spacing)

    np.testing.assert_allclose(integ, a**2 / 2 * w)

def test_interpolation():
    # Test the interpolation
    #
    n = 8
    w = 1 / np.pi
    Es = 3. / 4

    a = np.linspace

    a = np.linspace(0.1, 1.5).reshape(-1, 1) * np.ones((1, n))
    values =  a * w

    min_radius = a[0,0]
    grid_spacing = a[1, 0] - a[0,0]
    LinearInterpolatedPinningFieldUniformFromFile\
        .save_integral_values_to_file(values, min_radius, grid_spacing)
    LinearInterpolatedPinningFieldUniformFromFile\
        .save_values_and_slopes_to_file(values, grid_spacing, filename="values_and_slopes.npy")

    accelerator = torch.device("cpu")



    interp = LinearInterpolatedPinningFieldUniformFromFile(
                    filename="values_and_slopes.npy",
                    min_radius=min_radius ,
                    grid_spacing=grid_spacing,
                    accelerator=accelerator,
                    data_device=accelerator,
                    )
    interp.load_data()
    interp.load_integral_values()

    np.testing.assert_allclose(interp._integral_values,  LinearInterpolatedPinningFieldUniformFromFile
        .compute_integral_values(values, min_radius, grid_spacing))

    interp_points=a[10,:]+ 0.00000001
    integ = interp(interp_points, der="-1")

    # Test on collocation points
    np.testing.assert_allclose(integ,interp._integral_values[10, :])
    np.testing.assert_allclose(integ, a[10,:]**2 / 2 * w)

    interp_points = 0.4 + np.random.uniform(size=n)
    integ = interp(interp_points, der="-1")

    np.testing.assert_allclose(integ, interp_points**2 / 2 * w)


def test_energy():
    # Test the interpolation
    #
    n = 8
    w = 1 / np.pi
    Es = 3. / 4

    a = np.linspace

    a = np.linspace(0.1, 1.5).reshape(-1, 1) * np.ones((1, n))
    values =  a * w

    min_radius = a[0,0]
    grid_spacing = a[1, 0] - a[0,0]
    LinearInterpolatedPinningFieldUniformFromFile\
        .save_integral_values_to_file(values, min_radius, grid_spacing)
    LinearInterpolatedPinningFieldUniformFromFile\
        .save_values_and_slopes_to_file(values, grid_spacing, filename="values_and_slopes.npy")

    accelerator = torch.device("cpu")



    interp = LinearInterpolatedPinningFieldUniformFromFile(
                    filename="values_and_slopes.npy",
                    min_radius=min_radius ,
                    grid_spacing=grid_spacing,
                    accelerator=accelerator,
                    data_device=accelerator,
                    )
    interp.load_data()
    interp.load_integral_values()
