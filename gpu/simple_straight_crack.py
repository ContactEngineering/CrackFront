import numpy as np
from ContactMechanics.Tools.Logger import Logger

L = npx_front = 256
Lx = npx_propagation = 256
q_front_rfft = 2 * np.pi * np.fft.rfftfreq(npx_front, L / npx_front)

rms = 1.

a_init = np.random.normal(size=L)

# pinning field and its interpolation
period = Lx
grid_spacing = 1
indexes = np.arange(L, dtype=int)
#
kinks = np.arange(npx_propagation)

np.random.seed(0)
values = random_forces = np.random.normal(size=(L, Lx)) * rms

Lk = L / 4
qk = 2 * np.pi / Lk

a_test = np.zeros(npx_front)
a_test[0] = 1
elastic_stiffness_individual = np.fft.irfft(q_front_rfft * np.fft.rfft(a_test), n=npx_front)[0] + qk

a_init = np.zeros(L) + 1e-14
a = a_init.copy()

# initialize
colloc_point_above = np.searchsorted(kinks, a, side="right")

gtol = 1e-6
maxit = 10000

#             |-  This should ensures that we will find a strictly advancing configuration from beginning
#            v Otherwise our algorithm has problems
a_forcings = Lk * rms ** 2 + np.linspace(0, npx_propagation * 1.5, 100)
a_forcings = np.concatenate([a_forcings[:-1], a_forcings[::-1]])

# %%

# %%
logger = Logger(outevery=100)
driving_a_prev = -10
mean_a_RK = []
a = np.zeros(npx_front)
for driving_a in a_forcings:
    # print(driving_a)
    direction = np.sign(driving_a - driving_a_prev)
    # print(direction)
    nit = 0
    while nit < maxit:
        # print(a)
        # print(colloc_point_above)
        ###
        # Nullify the force on each pixel, assuming each pixel moves individually
        pinning_field_slope = (values[indexes, colloc_point_above % npx_propagation] - values[
            indexes, (colloc_point_above - 1) % npx_propagation]) / grid_spacing
        grad = np.fft.irfft(q_front_rfft * np.fft.rfft(a), n=npx_front) \
               + qk * (a - driving_a) \
               + values[indexes, (colloc_point_above - 1) % npx_propagation] \
               + pinning_field_slope * (a - grid_spacing * (colloc_point_above - 1))

        if logger:
            logger.st(["it", "max. residual"], [nit, np.max(abs(grad))])
        if (np.max(abs(grad)) < gtol):
            break

        stiffness = pinning_field_slope + elastic_stiffness_individual
        increment = - grad / stiffness
        ###

        a_new = a + increment
        mask_negative_stiffness = stiffness <= 0

        if direction == 1:
            # We let the line advance only until the boundary to the next pixel.
            # This is because the step length was based on the pinning curvature
            # which is erroneous as soon as we meet the next pixel
            mask_new_pixel = np.logical_or(a_new >= colloc_point_above * grid_spacing, mask_negative_stiffness)
            a_new[mask_new_pixel] = grid_spacing * colloc_point_above[mask_new_pixel]
            colloc_point_above[mask_new_pixel] += 1
        elif direction == -1:
            mask_new_pixel = np.logical_or(a_new <= grid_spacing * (colloc_point_above - 1), mask_negative_stiffness)
            a_new[mask_new_pixel] = grid_spacing * (colloc_point_above[mask_new_pixel] - 1)
            colloc_point_above[mask_new_pixel] -= 1

        # because of numerical errors it can be that the gradient points in the wrong
        # direction on some pixels, but is very small.
        # We just make sure these points do not move backwards
        # The same could be achieved by a_new = np.maximum(a_new, a)
        a_new[grad * direction >= 0] = a[grad * direction >= 0]

        # if direction == 1:
        #     assert (a_new >= a).all()
        # elif
        a = a_new

        nit += 1

    driving_a_prev = driving_a

    mean_a_RK.append(np.mean(a))

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.plot(a_forcings,
        a_forcings - mean_a_RK, label="KR")
# ax.plot(a_forcings, a_forcings - mean_a_trust, "+", label="TR, safe")
# ax.plot(a_forcings, a_forcings - mean_a_trust_coarse, "x", label="TR")

ax.legend()
plt.show(block=True)
# %%
