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
from scipy.optimize import OptimizeResult

from CrackFront.GenericElasticLine import ElasticLine
from scipy.optimize import OptimizeResult


class linear_interpolated_pinning_field:
    def __init__(self, values):
        L, Lx = values.shape
        self.npx_front = L
        self.npx_propagation = Lx
        self.period = Lx
        self.values = values
        self.grid_spacing = 1
        self.a_below = np.zeros(L, dtype=int)
        self.a_above = np.zeros(L, dtype=int)
        self.kinks = np.arange(self.npx_propagation)
        self.indexes = np.arange(L, dtype=int)

    def __call__(self, a, der="0"):
        np.floor(a, out=self.a_below, casting="unsafe").reshape(-1)
        np.ceil(a, out=self.a_above, casting="unsafe").reshape(-1)

        # Wrapping periodic boundary conditions
        self.a_below[self.a_below >= self.npx_propagation] = self.a_below[self.a_below >= self.npx_propagation] - self.npx_propagation
        self.a_above[self.a_above >= self.npx_propagation] = self.a_above[self.a_above >= self.npx_propagation] - self.npx_propagation

        # print(a_below)

        value_below = self.values[self.indexes, self.a_below].reshape(-1)
        value_above = self.values[self.indexes, self.a_above].reshape(-1)

        slope = (value_above - value_below)

        if der == "0":
            return value_below + slope * (a - self.a_below)
        elif der == "1":
            return slope

    #
    # Alternative to ceil and floor
    # from bisect import bisect_left
    #
    # https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html
    #
    #

class linear_interpolated_pinning_field_equaly_spaced:
    def __init__(self, values, kinks):
        """
        Linearly interpolates the pinning field in crack propagation direction

        Parameters:
        -----------
        values: np.ndarray of shape (npx_front, npx_propagation)
            values of the pinning field at the kinks
        kinks: np.ndarray of shape (npx_propagation)
            equidistantly spaced points representing the grid of the piecewise linear interpolation

        """
        L, Lx = values.shape
        self.npx_front = L
        self.npx_propagation = Lx
        self.grid_spacing = kinks[1] - kinks[0]
        self.period = Lx * self.grid_spacing
        self.values = values
        self.integral_values = np.zeros_like(values)
        increments = self.grid_spacing * self.values[:, :-1] \
                        + self.grid_spacing / 2 * (self.values[:, 1:] - self.values[:, :-1])
        self.integral_values[:, 1:] = np.cumsum(increments, axis=-1)

        self.a_below = np.zeros(L, dtype=int)
        self.a_above = np.zeros(L, dtype=int)
        self.kinks = kinks
        self.indexes = np.arange(L, dtype=int)

    def __call__(self, a, der="0"):

        self.a_above = np.searchsorted(self.kinks, a, side="right")
        self.a_below = self.a_above -1
        # TODO:  Wrapping periodic boundary conditions

        # print(a_below)

        value_below = self.values[self.indexes, self.a_below].reshape(-1)
        value_above = self.values[self.indexes, self.a_above].reshape(-1)

        slope = (value_above - value_below) / self.grid_spacing

        if der == "0":
            return value_below + slope * (a - self.kinks[self.a_below])
        elif der == "1":
            return slope
        elif der == "-1":
            return self.integral_values[self.indexes, self.a_below] \
                   + value_below * (a - self.kinks[self.a_below]) \
                   + slope * (a - self.kinks[self.a_below]) ** 2 / 2


def brute_rosso_krauth(a, driving_a, line, gtol=1e-4, maxit=10000, direction=1, logger=None):
    r"""
    Variation of PRE 65

    One requirement for this algorithm is that the starting position is purely advancing,
    i.e. the gradient is strictly negative when we pull the line in positive `a` direction

    In the original they move each pixel one after another.

    Here we move them all at once, which should make better use of vectorization and parallelisation.

    This might have drawbacks when the pinning field is weak, where pixels fail collectively.


    """
    L = line.npx_front
    a_test = np.zeros(L)
    a_test[0] = 1
    elastic_stiffness_individual = line.elastic_gradient(a_test, 0)[0]
    # elastic reaction force when we move the pixel individually

    grad = line.gradient(a, driving_a)
    if (grad * direction > 0).any():
        print("WARNING: Starting Configuration is not purely advancing or receding")

    nit = 0
    while (np.max(abs(grad)) > gtol) and nit < maxit:
        if logger:
            logger.st(["it", "max. residual"], [nit, np.max(abs(grad))])

        ###
        # Nullify the force on each pixel, assuming each pixel moves individually
        stiffness = line.pinning_field(a, "1") + elastic_stiffness_individual
        increment = - grad / stiffness
        ###

        # negative stiffness generates wrong step length.
        a_new = np.where(stiffness > 0, a + increment, a + (1 + 1e-14) * direction)
        # TODO: I am not sure that is correct, ending up in the middle of the next pixel might be too far.

        # because of numerical errors it can be that the gradient points in the wrong
        # direction on some pixels, but is very small.
        # We just make sure these points do not move backwards
        # The same could be achieved by a_new = np.maximum(a_new, a)
        a_new[grad * direction >= 0] = a[grad * direction >= 0]

        if direction == 1:
            a_ceiled = np.ceil(a)
            # We let the line advance only until the boundary to the next pixel.
            # This is because the step length was based on the pinning curvature
            # which is erroneous as soon as we meet the next pixel
            a_new = np.minimum(a_ceiled, a_new)
            a = a_new

            # When the pixel is at the edge, I am not sure on which pixel the curvature is actually evaluated.
            # A quickfix is to simply push the front a littlbit before.
            # TODO: this is not optimal.
            # - I could actually do that the iteration before, 2 lines above.
            # - Using piecewise quadratic, instead of linear interpolation solves this problem because the curvature is continuous
            # - Evaluating the pinning field in this code instead of in a black box way, like I do for other spacings
            a = np.where(a > np.floor(a), a, a + 1e-14)

        elif direction == -1:
            # a is decreasing
            a_floored = np.floor(a)
            a_new = np.maximum(a_floored, a_new)
            a = a_new
            a = np.where(a < np.ceil(a), a, a - 1e-14)

        grad = line.gradient(a, driving_a)

        nit += 1


    if nit == maxit:
        success = False
    else:
        success = True

    result = OptimizeResult({
        'success': success,
        'x': a,
        'nit': nit,
        })
    return result


def brute_rosso_krauth_other_spacing(a, driving_a, line, gtol=1e-4, maxit=10000, direction=1, logger=None):
    r"""
    WARNING: this has still some bugs when the front crosses the periodic boundary conditions.

    Variation of PRE 65

    One requirement for this algorithm is that the starting position is purely advancing,
    i.e. the gradient is strictly negative when we pull the line in positive `a` direction

    In the original they move each pixel one after another.

    Here we move them all at once, which should make better use of vectorization and parallelisation.

    This might have drawbacks when the pinning field is weak, where pixels fail collectively.

    """
    L = len(a)
    a_test = np.zeros(L)
    a_test[0] = 1
    elastic_stiffness_individual = line.elastic_gradient(a_test, 0)[0]
    # elastic reaction force when we move the pixel individually

    indexes = line.pinning_field.indexes
    kinks = line.pinning_field.kinks
    values = line.pinning_field.values
    npx_propagation = line.pinning_field.npx_propagation

    # This would also work on irregular grids
    # colloc_point_above = np.searchsorted(kinks, a, side="right")
    # colloc_point_above[a < 0] = colloc_point_above[a < 0] - npx_propagation

    colloc_point_above = np.zeros_like(a, dtype=int)
    colloc_point_above = np.ceil(a / line.pinning_field.grid_spacing, casting="unsafe", out=colloc_point_above)
    colloc_point_above[colloc_point_above==a] += 1
    grid_spacing = line.pinning_field.grid_spacing

    pinning_field_slope = (values[indexes, colloc_point_above % npx_propagation] - values[indexes, (colloc_point_above - 1) % npx_propagation]) / grid_spacing
    grad = line.elastic_gradient(a, driving_a) \
           + values[indexes, (colloc_point_above - 1) % npx_propagation] \
           + pinning_field_slope * (a - grid_spacing * (colloc_point_above - 1))
    if (grad * direction > 0).any():
        print("WARNING: Starting Configuration is not purely advancing or receding")

    nit = 0
    while (np.max(abs(grad)) > gtol) and nit < maxit:

        ###
        # Nullify the force on each pixel, assuming each pixel moves individually
        pinning_field_slope = (values[indexes, colloc_point_above % npx_propagation] - values[indexes, (colloc_point_above - 1) % npx_propagation]) / grid_spacing
        grad = line.elastic_gradient(a, driving_a) \
               + values[indexes, (colloc_point_above - 1) % npx_propagation] \
               + pinning_field_slope * (a - grid_spacing * (colloc_point_above - 1))

        if logger:
            logger.st(["it", "min. residual", "max. residual", "min. a", "mean a", "max. a", "min. collo", "max.collo"],
                      [nit, np.min(grad), np.max(grad), np.min(a), np.mean(a), np.max(a), np.min(colloc_point_above), np.max(colloc_point_above)])

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

        a = a_new

        nit += 1

    if nit == maxit:
        success = False
    else:
        success = True

    result = OptimizeResult({
        'success': success,
        'x': a,
        'nit': nit,
        })
    return result


def example_large_disp_force():
    print("example_large_disp_force")

    import time
    from CrackFront.Optimization.fixed_radius_trustregion_newton_cg import trustregion_newton_cg
    L = 2048
    Lx = 2048

    random_forces = np.random.normal(size=(L, Lx)) * .1  # *0.05
    pinning_field = linear_interpolated_pinning_field(random_forces)

    line = ElasticLine(L, L / 4, pinning_field=pinning_field)

    a_init = np.zeros(L) + 1e-14
    a = a_init.copy()
    w = 1

    grad = line.gradient(a, w)

    while (grad > 0).any():
        w += 1
        grad = line.gradient(a, w)

    gtol = 1e-6

    a_forcings = w + np.linspace(0, 1, 50)

    start_time = time.time()

    # %%

    mean_a_RK = []
    a = np.zeros(L)
    start_time = time.time()
    for a_forcing in a_forcings:
        sol = brute_rosso_krauth(a, a_forcing, line, gtol=gtol, maxit=100000)
        assert sol.success
        a = sol.x
        mean_a_RK.append(np.mean(a))
    print("TIME ROSSO Krauth", time.time() - start_time)

    # %%
    mean_a_trust_coarse = []
    a = np.zeros(L)
    start_time = time.time()
    for a_forcing in a_forcings:
        wrapped_gradient = lambda a: line.gradient(a, a_forcing)
        sol = trustregion_newton_cg(
            x0=a, gradient=wrapped_gradient,
            hessian_product=lambda a, p: line.hessian_product(p, a),
            trust_radius=1 / 2,
            maxiter=1000000,
            gtol=gtol,  # he has issues to reach the gtol at small values of a
            )
        a = sol.x
        assert sol.success
        mean_a_trust_coarse.append(np.mean(a))
    print("TIME TR COARSE", time.time() - start_time)

    # %%
    mean_a_trust = []
    a = np.zeros(L)
    start_time = time.time()
    for a_forcing in a_forcings:
        wrapped_gradient = lambda a: line.gradient(a, a_forcing)
        sol = trustregion_newton_cg(
            x0=a, gradient=wrapped_gradient,
            hessian_product=lambda a, p: line.hessian_product(p, a),
            trust_radius=1 / 8,
            gtol=gtol,
            maxiter=10000000,
            )
        a = sol.x
        assert sol.success

        mean_a_trust.append(np.mean(a))
    print("TIME TR", time.time() - start_time)

    # %%
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    ax.plot(a_forcings, a_forcings - mean_a_RK, label="KR")
    ax.plot(a_forcings, a_forcings - mean_a_trust, "+", label="TR, safe")

    ax.plot(a_forcings, a_forcings - mean_a_trust_coarse, "x", label="TR")
    ax.legend()
    plt.pause(.5)


def example_strong_pinning():
    print("example_strong_pinning")

    import time
    from CrackFront.Optimization.fixed_radius_trustregion_newton_cg import trustregion_newton_cg
    L = 2048
    Lx = 2048

    random_forces = np.random.normal(size=(L, Lx)) * 1.  # *0.05
    pinning_field = linear_interpolated_pinning_field(random_forces)

    line = ElasticLine(L, L / 16, pinning_field=pinning_field)

    a_init = np.zeros(L) + 1e-14
    a = a_init.copy()
    w = 1

    grad = line.gradient(a, w)

    while (grad > 0).any():
        w += 1
        grad = line.gradient(a, w)

    gtol = 1e-6

    a_forcings = w + np.linspace(0, 1, 200)

    start_time = time.time()

    # %%

    mean_a_RK = []
    a = np.zeros(L)
    start_time = time.time()
    for a_forcing in a_forcings:
        # print(a_forcing)

        sol = brute_rosso_krauth(a, a_forcing, line, gtol=gtol, maxit=100000)
        assert sol.success

        a = sol.x

        mean_a_RK.append(np.mean(a))
    print("TIME ROSSO Krauth", time.time() - start_time)

    # %%
    mean_a_trust_coarse = []
    a = np.zeros(L)
    start_time = time.time()
    for a_forcing in a_forcings:
        wrapped_gradient = lambda a: line.gradient(a, a_forcing)
        sol = trustregion_newton_cg(
            x0=a, gradient=wrapped_gradient,
            hessian_product=lambda a, p: line.hessian_product(p, a),
            trust_radius=1 / 2,
            maxiter=1000000,
            gtol=gtol, )
        a = sol.x
        assert sol.success
        mean_a_trust_coarse.append(np.mean(a))
    print("TIME TR COARSE", time.time() - start_time)

    # %%
    mean_a_trust = []
    a = np.zeros(L)
    start_time = time.time()
    for a_forcing in a_forcings:
        wrapped_gradient = lambda a: line.gradient(a, a_forcing)
        sol = trustregion_newton_cg(
            x0=a, gradient=wrapped_gradient,
            hessian_product=lambda a, p: line.hessian_product(p, a),
            trust_radius=1 / 8,
            gtol=gtol,
            maxiter=10000000)
        a = sol.x
        assert sol.success

        mean_a_trust.append(np.mean(a))
    print("TIME TR", time.time() - start_time)

    # %%
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    ax.plot(a_forcings, a_forcings - mean_a_RK, label="KR")
    ax.plot(a_forcings, a_forcings - mean_a_trust, "+", label="TR, safe")

    ax.plot(a_forcings, a_forcings - mean_a_trust_coarse, "x", label="TR")
    ax.legend()
    plt.pause(.5)

def time_RK_implementations():
    print("time_RK_implementations")

    import time

    L = 2048
    Lx = 2048

    random_forces = np.random.normal(size=(L, Lx)) * 1.  # *0.05
    pinning_field = linear_interpolated_pinning_field(random_forces)

    line = ElasticLine(L, L / 16, pinning_field=pinning_field)

    a_init = np.zeros(L) + 1e-14
    a = a_init.copy()
    w = 1

    grad = line.gradient(a, w)

    while (grad > 0).any():
        w += 1
        grad = line.gradient(a, w)

    gtol = 1e-6

    a_forcings = w + np.linspace(0, 1, 200)

    start_time = time.time()

    # %%

    for rosso_krauth in [brute_rosso_krauth, brute_rosso_krauth_other_spacing]:
        mean_a_RK = []
        a = np.zeros(L)
        start_time = time.time()
        for a_forcing in a_forcings:
            # print(a_forcing)
            wrapped_gradient = lambda a: line.gradient(a, a_forcing)

            sol = rosso_krauth(a, w, line, maxit=100000, gtol=gtol, )
            assert sol.success
            a = sol.x
            mean_a_RK.append(np.mean(a))
        print("TIME ROSSO Krauth", time.time() - start_time)


def example_brute_rosso_krauth_other_spacing():
    print("example_brute_rosso_krauth_other_spacing")
    L = 4
    Lx = 50

    a_init = np.random.normal(size=L)
    random_forces = np.zeros((L, Lx))
    pinning_field = linear_interpolated_pinning_field(random_forces)

    line = ElasticLine(L, L / 4, pinning_field=pinning_field)

    a = a_init.copy()
    w = -1
    grad = line.gradient(a, w)

    while (grad < 0).any():
        w -= 1
        grad = line.gradient(a, w)

    sol = brute_rosso_krauth_other_spacing(a, w, line, gtol=1e-10, dir=-1)
    assert sol.success

    np.testing.assert_allclose(sol.x, w)


if __name__ == "__main__":
    example_brute_rosso_krauth_other_spacing()
    time_RK_implementations()
    example_strong_pinning()
    example_large_disp_force()
