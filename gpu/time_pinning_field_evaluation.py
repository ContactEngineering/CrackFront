L = 4096
Lx = 512
import numpy as np

rms = 0.5

values = random_forces = np.random.normal(size=(L, Lx)) * rms
print(np.isfortran(values))
values_fortran = np.asfortranarray(values)
print(np.isfortran(values_fortran))

collocation_points = np.ones(L, dtype=int) * int(Lx /2 )
indexes = np.arange(L, dtype=int)

# %timeit?

# %%timeit -n 4 -r 4
values[:, collocation_points]


# %%timeit -n 4 -r 4
values_fortran[:, collocation_points]

# %%timeit -n 4 -r 4
values[indexes, collocation_points]

# %%timeit -n 4 -r 4
values_fortran[indexes, collocation_points]



# What about a rougher set of collocation points ? 

collocation_points += np.random.randint(-1, 1, size=L)

collocation_points

# %%timeit -n 4 -r 4
values[:, collocation_points]

# %%timeit -n 4 -r 4
values_fortran[:, collocation_points]

# %%timeit -n 4 -r 4
values[indexes, collocation_points]

# %%timeit -n 4 -r 4
values_fortran[indexes, collocation_points]

collocation_points += np.random.randint(-10, 10, size=L)

# %%timeit -n 4 -r 4
values[:, collocation_points]

# %%timeit -n 4 -r 4
values_fortran[:, collocation_points]

# %%timeit -n 4 -r 4
values[indexes, collocation_points]

# %%timeit -n 4 -r 4
values_fortran[indexes, collocation_points]

#
# Observation: with the fortran array, the evaluation is faster, but only when the roughness of the front is small. 
#
# The difference is not huge and maybe not visible


