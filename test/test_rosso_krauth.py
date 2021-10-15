import numpy as np

from CrackFront.GenericElasticLine import ElasticLine
from CrackFront.Optimization.fixed_radius_trustregion_newton_cg import trustregion_newton_cg
from CrackFront.Optimization.RossoKrauth import (
    linear_interpolated_pinning_field, brute_rosso_krauth,
    brute_rosso_krauth_other_spacing
    )
import pytest

@pytest.mark.parametrize("rosso_krauth", [brute_rosso_krauth, brute_rosso_krauth_other_spacing])
@pytest.mark.parametrize("Lx", [32, 33])
@pytest.mark.parametrize("L", [32, 33])
def test_no_pinning_field(L, Lx, rosso_krauth):
    a_init = np.random.normal(size=L)
    random_forces = np.zeros((L, Lx))
    pinning_field = linear_interpolated_pinning_field(random_forces)

    line = ElasticLine(L, L / 4, pinning_field=pinning_field)

    a = a_init.copy()
    w = 1
    grad = line.gradient(a, w)

    while (grad > 0).any():
        w += 1
        grad = line.gradient(a, w)

    sol = rosso_krauth(a, w, line, gtol=1e-10)
    assert sol.success

    np.testing.assert_allclose(sol.x, w)

@pytest.mark.parametrize("rosso_krauth", [brute_rosso_krauth, brute_rosso_krauth_other_spacing])
@pytest.mark.parametrize("Lx", [32, 33])
@pytest.mark.parametrize("L", [32, 33])
def test_no_pinning_field_moving_left(rosso_krauth, L, Lx):
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

    sol = rosso_krauth(a, w, line, gtol=1e-10, dir=-1)
    assert sol.success

    np.testing.assert_allclose(sol.x, w)

@pytest.mark.parametrize("rosso_krauth", [brute_rosso_krauth, brute_rosso_krauth_other_spacing])
def test_force_displacement_curve(rosso_krauth, plot=False):
    np.random.seed(0)

    L = 256
    Lx = 256

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

    # %%
    mean_a_RK = []
    a = np.zeros(L)
    for a_forcing in a_forcings:
        # print(a_forcing)
        sol = rosso_krauth(a, a_forcing, line, maxit=100000, gtol=gtol, )
        assert sol.success
        a = sol.x
        mean_a_RK.append(np.mean(a))

    # %%
    mean_a_trust_coarse = []
    a = np.zeros(L)
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

    # %%
    mean_a_trust = []
    a = np.zeros(L)
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

    # %%
    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        ax.plot(a_forcings, a_forcings - mean_a_RK, label="KR")
        ax.plot(a_forcings, a_forcings - mean_a_trust, "+", label="TR, safe")

        ax.plot(a_forcings, a_forcings - mean_a_trust_coarse, "x", label="TR")
        ax.legend()
        plt.show(block=True)
    # %%

    np.testing.assert_allclose(mean_a_trust, mean_a_RK)

@pytest.mark.parametrize("rosso_krauth", [brute_rosso_krauth, brute_rosso_krauth_other_spacing])
def test_force_displacement_curve_hysteresis(rosso_krauth, plot=False):
    np.random.seed(0)

    L = 256
    Lx = 256

    random_forces = np.random.normal(size=(L, Lx)) * 1.  # *0.05
    pinning_field = linear_interpolated_pinning_field(random_forces)

    line = ElasticLine(L, L / 16, pinning_field=pinning_field)

    gtol = 1e-10
    w = 0
    # Do an initial configuration
    sol = trustregion_newton_cg(
        x0=np.zeros(L), gradient=lambda a: line.gradient(a, w),
        hessian_product=lambda a, p: line.hessian_product(p, a),
        trust_radius=1 / 8,
        gtol=gtol,
        maxiter=10000000)
    assert sol.success, sol.message
    a_init = sol.x

    a_forcings = np.linspace(0, 20, 300)[1:]
    a_forcings = np.concatenate([a_forcings, a_forcings[:-1][::-1]])

    # %%
    mean_a_RK = []
    a = a_init.copy()
    a_forcing_prev = 0
    for a_forcing in a_forcings:
        # print(a_forcing)
        dir = 1 if a_forcing > a_forcing_prev else -1
        # print(dir)
        sol = rosso_krauth(a, a_forcing, line, maxit=100000, gtol=gtol, dir=dir)
        assert sol.success
        a = sol.x
        mean_a_RK.append(np.mean(a))
        a_forcing_prev = a_forcing

    print("RK done")

    # %%
    mean_a_trust = []
    a = np.zeros(L)
    for a_forcing in a_forcings:
        # print(a_forcing)
        wrapped_gradient = lambda a: line.gradient(a, a_forcing)
        sol = trustregion_newton_cg(
            x0=a, gradient=wrapped_gradient,
            hessian_product=lambda a, p: line.hessian_product(p, a),
            trust_radius=1 / 32,
            gtol=gtol,
            maxiter=100000000)
        a = sol.x
        assert sol.success

        mean_a_trust.append(np.mean(a))

    print("trust  done")
    # %%
    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        ax.plot(a_forcings, a_forcings - mean_a_RK, label="KR")
        ax.plot(a_forcings, a_forcings - mean_a_trust, "+", label="TR, safe")

        ax.legend()
        plt.show(block=True)
    # %%

    np.testing.assert_allclose(mean_a_trust, mean_a_RK)



@pytest.mark.parametrize("rosso_krauth", [brute_rosso_krauth, brute_rosso_krauth_other_spacing])
@pytest.mark.parametrize("Lx", [32, 33])
@pytest.mark.parametrize("L", [32, 33])
def test_no_pinning_field(L, Lx, rosso_krauth):
    a_init = np.random.normal(size=L)
    random_forces = np.zeros((L, Lx))
    pinning_field = linear_interpolated_pinning_field(random_forces)

    line = ElasticLine(L, L / 4, pinning_field=pinning_field)

    a = a_init.copy()
    w = 1
    grad = line.gradient(a, w)

    while (grad > 0).any():
        w += 1
        grad = line.gradient(a, w)

    sol = rosso_krauth(a, w, line, gtol=1e-10)
    assert sol.success

    np.testing.assert_allclose(sol.x, w)

@pytest.mark.parametrize("rosso_krauth", [brute_rosso_krauth, brute_rosso_krauth_other_spacing])
@pytest.mark.parametrize("Lx", [32, 33])
@pytest.mark.parametrize("L", [32, 33])
def test_no_pinning_field_moving_left(rosso_krauth, L, Lx):
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

    sol = rosso_krauth(a, w, line, gtol=1e-10, dir=-1)
    assert sol.success

    np.testing.assert_allclose(sol.x, w)

@pytest.mark.parametrize("rosso_krauth", [brute_rosso_krauth,
                                        pytest.param(brute_rosso_krauth_other_spacing, marks=pytest.mark.skip(reason="does not support the periodic BC"))
                                          ])
def test_force_displacement_curve(rosso_krauth, plot=False):
    np.random.seed(0)

    L = 256
    Lx = 256

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

    # %%
    mean_a_RK = []
    a = np.zeros(L)
    for a_forcing in a_forcings:
        # print(a_forcing)
        sol = rosso_krauth(a, a_forcing, line, maxit=100000, gtol=gtol, )
        assert sol.success
        a = sol.x
        mean_a_RK.append(np.mean(a))

    # %%
    mean_a_trust_coarse = []
    a = np.zeros(L)
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

    # %%
    mean_a_trust = []
    a = np.zeros(L)
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

    # %%
    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        ax.plot(a_forcings, a_forcings - mean_a_RK, label="KR")
        ax.plot(a_forcings, a_forcings - mean_a_trust, "+", label="TR, safe")

        ax.plot(a_forcings, a_forcings - mean_a_trust_coarse, "x", label="TR")
        ax.legend()
        plt.show(block=True)
    # %%

    np.testing.assert_allclose(mean_a_trust, mean_a_RK)

@pytest.mark.parametrize("rosso_krauth", [#brute_rosso_krauth,
    brute_rosso_krauth_other_spacing])
def test_force_displacement_curve_hysteresis_do_not_wrap_periodic(rosso_krauth, plot=False):
    """

    Here the starting position is well inside the random field, so that the solver doesn't need to support
    the periodic boundary conditions
    """
    np.random.seed(0)

    L = 128
    Lx = 128

    random_forces = np.random.normal(size=(L, Lx)) * 1.  # *0.05
    pinning_field = linear_interpolated_pinning_field(random_forces)

    line = ElasticLine(L, L / 16, pinning_field=pinning_field)

    gtol = 1e-10
    w = 20
    # Do an initial configuration
    sol = trustregion_newton_cg(
        x0=np.zeros(L), gradient=lambda a: line.gradient(a, w),
        hessian_product=lambda a, p: line.hessian_product(p, a),
        trust_radius=1 / 8,
        gtol=gtol,
        maxiter=10000000)
    assert sol.success, sol.message
    a_init = sol.x

    a_forcings = w + np.linspace(0, 20, 300)[1:]
    a_forcings = np.concatenate([a_forcings, a_forcings[:-1][::-1]])

    # %%
    mean_a_RK = []
    a = a_init.copy()
    a_forcing_prev = 0
    for a_forcing in a_forcings:
        # print(a_forcing)
        dir = 1 if a_forcing > a_forcing_prev else -1
        # print(dir)
        sol = rosso_krauth(a, a_forcing, line, maxit=10000000, gtol=gtol, dir=dir)
        assert sol.success
        a = sol.x
        mean_a_RK.append(np.mean(a))
        a_forcing_prev = a_forcing

    print("RK done")

    # %%
    mean_a_trust = []
    a = np.zeros(L)
    for a_forcing in a_forcings:
        # print(a_forcing)
        wrapped_gradient = lambda a: line.gradient(a, a_forcing)
        sol = trustregion_newton_cg(
            x0=a, gradient=wrapped_gradient,
            hessian_product=lambda a, p: line.hessian_product(p, a),
            trust_radius=1 / 64,
            gtol=gtol,
            maxiter=100000000)
        a = sol.x
        assert sol.success

        mean_a_trust.append(np.mean(a))

    print("trust  done")
    # %%
    if True:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        ax.plot(a_forcings, a_forcings - mean_a_RK, label="KR")
        ax.plot(a_forcings, a_forcings - mean_a_trust, "+", label="TR, safe")

        ax.legend()
        plt.show(block=True)
    # %%

    np.testing.assert_allclose(mean_a_trust, mean_a_RK)

