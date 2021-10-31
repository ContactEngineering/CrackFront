# %%
import numpy as np
import torch
import time

# %%
######################## PARAMETERS
L = npx_front = 4096
Lx = npx_propagation = 256
rms = 1.
Lk = 256

# randomness:
seed = 0

# numerics:
gtol = 1e-6
maxit = 100000

# history:
#             |-  This should ensures that we will find a strictly advancing configuration from beginning
#            v Otherwise our algorithm has problems
a_drivings = Lk * rms ** 2 + np.linspace(0, npx_propagation * 0.1, 5)
a_drivings = np.concatenate([a_drivings[:-1], a_drivings[::-1]])

######################## K

q_front_rfft = 2 * np.pi * torch.fft.rfftfreq(npx_front, L / npx_front, device=torch.device('cuda'))

# pinning field and its interpolation
period = Lx
grid_spacing = torch.tensor(1., dtype=torch.double)
indexes = torch.arange(L, dtype=int)
#
np.random.seed(seed)
values = random_forces = np.random.normal(size=(L, Lx)) * rms

slopes =  np.roll(values, -1, axis=-1)- values # / grid _spacing ! 
values_and_slopes = torch.from_numpy(np.stack([values, slopes], axis=2)).cuda()

# %%
qk = 2 * np.pi / Lk
a_test = torch.zeros(npx_front, device=torch.device('cuda'))
a_test[0] = 1
elastic_stiffness_individual = torch.fft.irfft(q_front_rfft * torch.fft.rfft(a_test), n=npx_front)[0] + qk

a_init = np.zeros(L) + 1e-14
a = a_init.copy()

# initialize the collocation points that correspond to the crack front.
# of course for a = 0 it is trivial.....
kinks = np.arange(npx_propagation)
colloc_point_above = torch.from_numpy(np.searchsorted(kinks, a, side="right")).cuda()

# %%
torch.cuda.is_available()

# %%
elastic_stiffness_individual

# %%
a = torch.zeros(npx_front, dtype=torch.double, device=torch.device('cuda'))
driving_a = a_drivings[0]
driving_a = torch.tensor(driving_a).cuda()
direction = torch.tensor(1, device=torch.device('cuda'))


# %%
class ConvergenceError(Exception):
    pass


# %%

def simulate():
    logger= None
    #logger = Logger(outevery=100)
    driving_a_prev = -10
    mean_a_RK = []
    a = torch.zeros(npx_front, dtype=torch.double, device=torch.device('cuda'))
    for driving_a in a_drivings:
        driving_a = torch.tensor(driving_a).cuda()
        # whether the crack is expected to move forward or backward
        direction = torch.sign(driving_a - driving_a_prev)
        # print(direction)
        nit = 0
        while nit < maxit:
            ###
            # Nullify the force on each pixel, assuming each pixel moves individually
            
            current_value_and_slope = values_and_slopes[indexes, (colloc_point_above - 1) % npx_propagation, :]
            #print(current_value_and_slope.shape)
            #pinning_field_slope = (values[indexes, colloc_point_above % npx_propagation] - values[indexes, (colloc_point_above - 1) % npx_propagation]) / grid_spacing
            grad = torch.fft.irfft(q_front_rfft * torch.fft.rfft(a), n=npx_front)
            grad.add_(qk * (a - driving_a))
            grad.add_(current_value_and_slope[:,0])
            grad.add_(current_value_and_slope[:,1] * (a - grid_spacing * (colloc_point_above - 1)))

            if logger:
                logger.st(["it", "max. residual"], [nit, torch.max(abs(grad))])
            if (torch.max(torch.abs(grad)) < gtol):
                break

            stiffness = current_value_and_slope[:,1] + elastic_stiffness_individual
            increment = - grad / stiffness
            ###

            a_new = a + increment
            mask_negative_stiffness = stiffness <= 0

            if direction == 1:
                # We let the line advance only until the boundary to the next pixel.
                # This is because the step length was based on the pinning curvature
                # which is erroneous as soon as we meet the next pixel
                mask_new_pixel = torch.logical_or(a_new >= colloc_point_above * grid_spacing, mask_negative_stiffness)
                a_new[mask_new_pixel] = grid_spacing * colloc_point_above[mask_new_pixel]
                colloc_point_above[mask_new_pixel] += 1
                
                # instead of a_new[grad*direction >= 0]
                a_new = torch.maximum(a_new, a)
                
            elif direction == -1:
                mask_new_pixel = torch.logical_or(a_new <= grid_spacing * (colloc_point_above - 1), mask_negative_stiffness)
                a_new[mask_new_pixel] = grid_spacing * (colloc_point_above[mask_new_pixel] - 1)
                colloc_point_above[mask_new_pixel] -= 1
                a_new = torch.minimum(a_new, a)

            # because of numerical errors it can be that the gradient points in the wrong
            # direction on some pixels, but is very small.
            # We just make sure these points do not move backwards
            # I replaced this line with 
            # a_new = torch.maximum(a_new, a)
            # above. Reduced the time from 15% to 0.8 % 
            #
            # it seems that this masking is time insensitive.
            # 
            #a_new[grad * direction >= 0] = a[grad * direction >= 0]

            # if direction == 1:
            #     assert (a_new >= a).all()
            # elif
            a = a_new

            nit += 1
        print(nit)
        if nit == maxit:
            raise ConvergenceError
        driving_a_prev = driving_a
        mean_a_RK.append(torch.mean(a))
    return mean_a_RK

# %%
start_time = time.time()
mean_a_RK = simulate()
time_cuda = time.time() - start_time

# %% [markdown]
# %load_ext snakeviz
# %% [markdown]
# %snakeviz mean_a_RK = simulate()
# %%
%load_ext line_profiler
%lprun -f simulate -T line_profile.txt mean_a_RK = simulate()

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.plot(a_drivings, a_drivings - mean_a_RK, label="KR")

ax.legend()
plt.show(block=True)
# %% [markdown]
# ## compare numpy vs. pytorch cuda

# %% [markdown]
# ### numpy implementation 

# %%

q_front_rfft = 2 * np.pi * np.fft.rfftfreq(npx_front, L / npx_front)

# pinning field and its interpolation
period = Lx
grid_spacing = 1
indexes = np.arange(L, dtype=int)
#
np.random.seed(seed)
values = random_forces = np.asfortranarray(np.random.normal(size=(L, Lx)) * rms)

# %%
qk = 2 * np.pi / Lk
a_test = np.zeros(npx_front)
a_test[0] = 1
elastic_stiffness_individual = np.fft.irfft(q_front_rfft * np.fft.rfft(a_test), n=npx_front)[0] + qk

a_init = np.zeros(L) + 1e-14
a = a_init.copy()

# initialize the collocation points that correspond to the crack front.
# of course for a = 0 it is trivial.....
kinks = np.arange(npx_propagation)
colloc_point_above = np.searchsorted(kinks, a, side="right")


# %%
#%load_ext snakeviz
# %%
#%%snakeviz
def simulate_numpy():
    #logger = Logger(outevery=100)
    logger = None

    driving_a_prev = -10
    mean_a_RK = []
    a = np.zeros(npx_front)
    for driving_a in a_drivings:
        # whether the crack is expected to move forward or backward
        direction = np.sign(driving_a - driving_a_prev)
        # print(direction)
        nit = 0
        while nit < maxit:
            ###
            # Nullify the force on each pixel, assuming each pixel moves individually
            pinning_field_slope = (values[indexes, colloc_point_above % npx_propagation] - values[indexes, (colloc_point_above - 1) % npx_propagation]) / grid_spacing
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
        if nit == maxit:
            raise ConvergenceError
        driving_a_prev = driving_a
        mean_a_RK.append(np.mean(a))
    return mean_a_RK


# %%
start_time = time.time()
mean_a_RK = simulate_numpy()
time_numpy = time.time() - start_time

# %%
time_numpy

# %%
time_cuda

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.plot(a_drivings, a_drivings - mean_a_RK, label="KR")

ax.legend()
plt.show(block=True)

# %%
