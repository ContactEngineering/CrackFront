import matplotlib.pyplot as plt
import numpy as np

from CrackFront.Optimization.RossoKrauth import linear_interpolated_pinning_field_equaly_spaced

def test_integral():
    npx_propagation = 10
    npx_front = 1

    grid_spacing = 3
    start_kink = -0.5
    kinks = np.arange(npx_propagation) * grid_spacing + start_kink

    colloc_grads = np.random.uniform(0, 1, size=(npx_front, npx_propagation))

    interp = linear_interpolated_pinning_field_equaly_spaced(colloc_grads, kinks)

    # %%
    if True:
        fig, axes = plt.subplots(3, 1, sharex=True)

        x = np.linspace(kinks[0], kinks[-1], 1000)[1:-1]

        axes[0].plot(kinks, interp.integral_values.reshape(-1), "+", c="gray")

        axes[0].plot(x, [interp(x, der="-1") for x in x], label="interp")
        axes[1].plot(x, [interp(x, der="0") for x in x], label="interp")
        axes[2].plot(x, [interp(x, der="1") for x in x], label="interp")

        axes[1].plot(kinks, colloc_grads[0, :], "+k", label="collocation points")

        pot = np.array([interp(x, der="-1") for x in x])
        finite_difference_grad = (pot[1:] - pot[:-1]) / (x[1] - x[0])

        axes[1].plot((x[1:] + x[:-1]) / 2, finite_difference_grad, "--", label="collocation points")
    # %%
    # TODO: automatic test