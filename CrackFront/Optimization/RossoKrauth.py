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
        L, Lx = values.shape
        self.npx_front = L
        self.npx_propagation = Lx
        self.grid_spacing = kinks[1] - kinks[0]
        self.period = Lx * self.grid_spacing
        self.values = values

        self.a_below = np.zeros(L, dtype=int)
        self.a_above = np.zeros(L, dtype=int)
        self.kinks = kinks
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

        slope = (value_above - value_below) / self.grid_spacing

        if der == "0":
            return value_below + slope * (a - self.a_below)
        elif der == "1":
            return slope

def brute_rosso_krauth(a, driving_a, line, gtol=1e-4, maxit=10000, dir=1, logger=None):
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
    if (grad * dir > 0).any():
        raise ValueError("Starting Configuration is not purely advancing or receding")

    nit = 0
    while (np.max(abs(grad)) > gtol) and nit < maxit:
        if logger:
            logger.st(["it", "max. residual"], [nit, np.max(abs(grad))])
        # print(grad)
        # Nullify the force on each pixel, assuming it is greater then
        stiffness = line.pinning_field(a, "1") + elastic_stiffness_individual

        increment = - grad / stiffness
        # negative stiffness generates wrong step length.
        a_new = np.where(stiffness > 0, a + increment, a + (1 + 1e-14) * dir)
        a_new[grad * dir >= 0] = a[grad * dir >= 0]

        if dir == 1:
            a_ceiled = np.ceil(a)
            a_new = np.minimum(a_ceiled, a_new)
            a = a_new
            a = np.where(a > np.floor(a), a, a + 1e-14)

            # I think this line is useless
            #a[np.logical_and((grad < 0), (a == np.floor(a)))] += 1e-13
            # Don't move backward when gradient is negative due to imprecision
        elif dir == -1:
            a_floored = np.floor(a)
            a_new = np.maximum(a_floored, a_new)
            a = a_new
            a = np.where(a < np.ceil(a), a, a - 1e-14)

            #a[np.logical_and((grad < 0), (a == np.floor(a)))] += 1e-13

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


def brute_rosso_krauth_other_spacing(a, driving_a, line, gtol=1e-4, maxit=10000, dir=1, logger=None):
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

    grad = line.gradient(a, driving_a)
    if (grad * dir > 0).any():
        raise ValueError("Starting Configuration is not purely advancing or receding")

    indexes = line.pinning_field.indexes
    kinks = line.pinning_field.kinks
    values = line.pinning_field.values
    colloc_point_above = np.searchsorted(kinks, a, side="right")

    period = line.pinning_field.period

    grid_spacing = line.pinning_field.grid_spacing
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # l, = ax.plot(a)
    # l_above,  = ax.plot(kinks[colloc_point_above])
    nit = 0
    while (np.max(abs(grad)) > gtol) and nit < maxit:
        if logger:
            logger.st(["it", "max. residual"], [nit, np.max(abs(grad))])
        # print(grad)
        # Nullify the force on each pixel, assuming it is greater then
        pinning_field_slope = (values[indexes, colloc_point_above] - values[indexes, colloc_point_above-1]) / grid_spacing
        grad = line.elastic_gradient(a, driving_a) \
            + values[indexes, colloc_point_above-1] \
            + pinning_field_slope * (a - kinks[colloc_point_above-1])

        stiffness = pinning_field_slope + elastic_stiffness_individual

        increment = - grad / stiffness
        # negative stiffness generates wrong step length.
        a_new = np.where(stiffness > 0, a + increment, a + (1 + 1e-14) * dir)
        a_new[grad * dir >= 0] = a[grad * dir >= 0]

        if dir == 1:
            mask_new_pixel = a_new % period >= kinks[colloc_point_above]
            a_new[mask_new_pixel] = kinks[colloc_point_above][mask_new_pixel]
            colloc_point_above[mask_new_pixel] += 1
            a = a_new
        elif dir == -1:
            mask_new_pixel = a_new % period <= kinks[colloc_point_above-1]
            a_new[mask_new_pixel] = kinks[colloc_point_above-1][mask_new_pixel]
            colloc_point_above[mask_new_pixel] -= 1

            a = a_new


        # l_above.set_ydata(kinks[colloc_point_above])
        # l.set_ydata(a)
        # ax.set_ylim(np.min(a-1), np.max(a))
        # plt.pause(.001)

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
        # print(a_forcing)
        wrapped_gradient = lambda a: line.gradient(a, a_forcing)

        a = brute_rosso_krauth(a, wrapped_gradient, pinning_field, line, maxit=100000, tol=gtol, )
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
        wrapped_gradient = lambda a: line.gradient(a, a_forcing)

        a = brute_rosso_krauth(a, wrapped_gradient, pinning_field, line, maxit=100000, tol=gtol, )
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

            a = rosso_krauth(a, w, line, maxit=100000, gtol=gtol, )
            mean_a_RK.append(np.mean(a))
        print("TIME ROSSO Krauth", time.time() - start_time)


if __name__ == "__main__":
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
    #example_strong_pinning()
    #example_large_disp_force()
