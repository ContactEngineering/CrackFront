import numpy as np

from CrackFront.GenericElasticLine import ElasticLine


class linear_interpolated_pinning_field:
    def __init__(self, values):
        L, Lx = values.shape
        self.L = L
        self.Lx = Lx
        self.values = values

        self.a_below = np.zeros(L, dtype=int)
        self.a_above = np.zeros(L, dtype=int)

        self.indexes = np.arange(L, dtype=int)

    def __call__(self, a, der="0"):
        np.floor(a, out=self.a_below, casting="unsafe").reshape(-1)
        np.ceil(a, out=self.a_above, casting="unsafe").reshape(-1)

        self.a_below[self.a_below >= self.Lx] = self.a_below[self.a_below >= self.Lx] - self.Lx
        self.a_above[self.a_above >= self.Lx] = self.a_above[self.a_above >= self.Lx] - self.Lx

        # print(a_below)

        value_below = self.values[self.indexes, self.a_below].reshape(-1)
        value_above = self.values[self.indexes, self.a_above].reshape(-1)

        slope = (value_above - value_below)

        if der == "0":
            return value_below + slope * (a - self.a_below)
        elif der == "1":
            return slope

def brute_rosso_krauth(a, gradient, pinning_field, line, tol=1e-4, maxit=10000, monitor=False):
    """
    Variation of PRE 65

    In the original they move each pixel one after another.

    Here we move them all at once, which should make better use of vectorization and parallelisation.

    This might have drawbacks when the pinning field is weak, where pixels fail collectively.

    """
    L = line.L
    a_test = np.zeros(L)
    a_test[0] = 1
    elastic_stiffness_individual = line.elastic_gradient(a_test, 0)[0]
    # elastic reaction force when we move the pixel individually

    grad = gradient(a)
    nit = 0
    if monitor:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, sharex=True)
        l0, = ax[0].plot(grad, ".")
        l1, = ax[1].plot(np.ones_like(grad), ".")
        # l12, =ax[1].plot(grad)

        l2, = ax[1].plot(grad, ".", c="r")
        l3, = ax[2].plot(a, ".")
        l32, = ax[2].plot(a, ".")

        ax[1].set_ylim(1e-2, 2)
        ax[1].set_yscale("log")
    while (np.max(abs(grad)) > tol) and nit < maxit:
        # print(grad)
        # Nullify the force on each pixel, assuming it is greater then
        stiffness = pinning_field(a.copy(), "1") + elastic_stiffness_individual

        increment = - grad / stiffness
        # negative stiffness generates wrong step length.
        a_new = np.where(stiffness > 0, a + increment, a + 1 + 1e-14)
        # l12.set_ydata(a_new - a)
        a_new[grad >= 0] = a[grad >= 0]

        a_ceiled = np.ceil(a)
        a_new = np.minimum(a_ceiled, a_new)
        if monitor:

            l0.set_ydata(grad)
            l1.set_ydata(a_new - a)
            # ax[1].set_ylim(0, np.max(a_new-a))
            l3.set_ydata(a)
            l32.set_ydata(a_new)
            ax[2].set_ylim(np.min(a) - 2, np.max(a) + 2)
            ax[1].set_ylim(bottom=np.mean(a_new - a) / 10)

            plt.pause(0.01)
        a = a_new
        a = np.where(a > np.floor(a), a, a + 1e-14)
        grad = gradient(a.copy())

        a[np.logical_and((grad < 0), (a == np.floor(a)))] += 1e-13
        # print(grad)
        # Don't move backward when gradient is negative due to imprecision
        # assert (grad < postol).all(), np.max(grad)
        nit += 1

    if nit == maxit:
        print("maxit reached")

    return a



import time

def test_large_disp_force():
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
        w+=1
        grad = line.gradient(a,w)


    gtol = 1e-6

    a_forcings = w + np.linspace(0, 1, 50)

    start_time = time.time()

    #%%

    mean_a_RK=[]
    a = np.zeros(L)
    start_time = time.time()
    for a_forcing in a_forcings:
        #print(a_forcing)
        wrapped_gradient = lambda a: line.gradient(a, a_forcing)

        a = brute_rosso_krauth(a, wrapped_gradient, pinning_field, line, maxit=100000,tol=gtol,)
        mean_a_RK.append(np.mean(a))
    print("TIME ROSSO Krauth", time.time() - start_time)


    # %%
    mean_a_trust_coarse=[]
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
            gnorm_termination=lambda a: np.max(abs(a)),)
        a = sol.x
        assert sol.success
        mean_a_trust_coarse.append(np.mean(a))
    print("TIME TR COARSE", time.time() - start_time)

    # %%
    mean_a_trust=[]
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
            gnorm_termination=lambda a: np.max(abs(a)),)
        a = sol.x
        assert sol.success

        mean_a_trust.append(np.mean(a))
    print("TIME TR", time.time() - start_time)

    # %%
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    ax.plot(a_forcings, a_forcings -mean_a_RK , label="KR")
    ax.plot(a_forcings,  a_forcings - mean_a_trust, "+", label= "TR, safe")

    ax.plot(a_forcings,  a_forcings - mean_a_trust_coarse,  "x",  label= "TR")
    ax.legend()
    plt.pause(.5)


def test_strong_pinning():
    from CrackFront.Optimization.fixed_radius_trustregion_newton_cg import trustregion_newton_cg
    L = 2048
    Lx = 2048

    random_forces = np.random.normal(size=(L, Lx)) * 1. # *0.05
    pinning_field = linear_interpolated_pinning_field(random_forces)

    line = ElasticLine(L, L / 16, pinning_field=pinning_field)


    a_init = np.zeros(L) + 1e-14
    a = a_init.copy()
    w = 1

    grad = line.gradient(a, w)

    while (grad > 0).any():
        w+=1
        grad = line.gradient(a,w)


    gtol = 1e-6

    a_forcings = w + np.linspace(0, 1, 200)

    start_time = time.time()

    #%%

    mean_a_RK=[]
    a = np.zeros(L)
    start_time = time.time()
    for a_forcing in a_forcings:
        #print(a_forcing)
        wrapped_gradient = lambda a: line.gradient(a, a_forcing)

        a = brute_rosso_krauth(a, wrapped_gradient, pinning_field, line, maxit=100000,tol=gtol,)
        mean_a_RK.append(np.mean(a))
    print("TIME ROSSO Krauth", time.time() - start_time)


    # %%
    mean_a_trust_coarse=[]
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
            gnorm_termination=lambda a: np.max(abs(a)),)
        a = sol.x
        assert sol.success
        mean_a_trust_coarse.append(np.mean(a))
    print("TIME TR COARSE", time.time() - start_time)

    # %%
    mean_a_trust=[]
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
            gnorm_termination=lambda a: np.max(abs(a)),)
        a = sol.x
        assert sol.success

        mean_a_trust.append(np.mean(a))
    print("TIME TR", time.time() - start_time)

    # %%
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    ax.plot(a_forcings, a_forcings -mean_a_RK , label="KR")
    ax.plot(a_forcings,  a_forcings - mean_a_trust, "+", label= "TR, safe")

    ax.plot(a_forcings,  a_forcings - mean_a_trust_coarse,  "x",  label= "TR")
    ax.legend()
    plt.pause(.5)


if __name__=="__main__":
    test_strong_pinning()
    test_large_disp_force()