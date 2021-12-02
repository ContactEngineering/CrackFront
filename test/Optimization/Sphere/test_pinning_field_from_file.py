import numpy as np
import torch
from matplotlib import pyplot as plt

from CrackFront.Optimization.propagate_sphere_pytorch import LinearInterpolatedPinningFieldUniformFromFile

disable_cuda = False

# TORCH code starts here
if torch.cuda.is_available() and not disable_cuda:
    accelerator = torch.device("cuda")
    print("CUDA detected, using CUDA")
else:
    if not disable_cuda:
        print("CUDA not available, fall back to torch on CPU")
    else:
        print("CUDA disabled, use CPU")
    accelerator = torch.device("cpu")


def test_pinning_field_from_file():
    npx_propagation = 25
    npx_front = 19

    values = np.random.normal(size=(npx_propagation, npx_front))

    grid_spacing = 4
    minimum_radius = 3

    # SAVE
    LinearInterpolatedPinningFieldUniformFromFile.save_values_and_slopes_to_file(
        values,
        grid_spacing=grid_spacing,
        filename="values_and_slopes.npy"
        )

    # Instantiate the file handler
    pf = LinearInterpolatedPinningFieldUniformFromFile(
        filename="values_and_slopes.npy",
        grid_spacing=grid_spacing,
        minimum_radius=minimum_radius,
        accelerator=accelerator
        )

    subdomain = [5, 21]
    collocation_points = torch.from_numpy(np.random.randint(*subdomain, size=npx_front))

    # Load all data
    pf.load_data(0, npx_propagation)

    values_and_slopes_from_full = pf.values_and_slopes(collocation_points).to(device="cpu").numpy()

    # Load part of the data
    pf.load_data(*subdomain)

    values_and_slopes_from_part = pf.values_and_slopes(collocation_points).to(device="cpu").numpy()

    # should yield the same values at the collocation point
    np.testing.assert_equal(values_and_slopes_from_part, values_and_slopes_from_full)

    # check the slopes are computed ok
    pf.load_data(0, npx_propagation)
    values_and_slopes_upper = pf.values_and_slopes(collocation_points + 1).to(device="cpu").numpy()
    np.testing.assert_equal(
        values_and_slopes_from_part[:, 1],
        (values_and_slopes_upper[:, 0] - values_and_slopes_from_part[:, 0]) / grid_spacing
        )


