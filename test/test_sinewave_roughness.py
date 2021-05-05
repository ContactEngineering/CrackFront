import numpy as np

from matplotlib import pyplot as plt

from CrackFront.StraightForRoughness import SinewaveCrackFrontLoadEnergyConstK
from CrackFront.Straight import SinewaveCrackFrontLoad
from CrackFront.Optimization import trustregion_newton_cg

# nondimensional units
Es = 1 / np.pi
h = 1.  # amplitude (half peak to valley) of the sinewave
sinewave_lambda = 1.
sx = 1.


def test_CFRconstK_against_CFK_tangential_sinewave():
    """
    In the small roughness limit
    I can fully map the roughness to work of adhesion heterogeneity
    and compare to previous implementations
    """

    # Parameters
    Kc = 0.5  # in units of h pwfc. K / hp wfc = alpha

    sy = 0.25
    load = 0
    n = 32
    _plot = True
    if _plot:
        fig, ax = plt.subplots()

    for kr_amplitude in [0.01 * Kc
                         # 0.1 * Kc,
                         # 0.5 * Kc,
                         ]:
        def kr(a, y, der="0"):
            if der == "0":
                return np.cos(2 * np.pi * y / sy) * kr_amplitude
            else:
                return np.zeros_like(y)

        cf_R = SinewaveCrackFrontLoadEnergyConstK(
            n, sy,
            kr_left=kr,
            kr_right=kr,
            w=Kc ** 2 / (2 * Es)
            )

        a = np.ones(2 * n) * 0.25

        sol = trustregion_newton_cg(
            x0=a,
            gradient=lambda a: cf_R.gradient(a, load),
            hessian_product=lambda a, p: cf_R.hessian_product(p, a, load),
            trust_radius=0.25 * np.min(a),
            maxiter=3000,
            gtol=1e-11)

        a_R = sol.x

        cf_k = SinewaveCrackFrontLoad(
            n, sy,
            kc=lambda a, z: Kc - kr(a, z),
            dkc=lambda a, z: kr(a, z, der="1")
            )

        sol = trustregion_newton_cg(
            x0=a,
            gradient=lambda a: cf_k.gradient(a, load),
            hessian=lambda a: cf_k.hessian(a, load),
            trust_radius=0.25 * np.min(a),
            maxiter=3000,
            gtol=1e-11)

        a_k = sol.x

        if _plot:

            al = a_R[:n]
            ar = a_R[n:]

            ax.plot(.5 + ar, cf_R.y)
            ax.plot(0.5 - al, cf_R.y)

            al = a_k[:n]
            ar = a_k[n:]

            ax.plot(.5 + ar, cf_k.y)
            ax.plot(0.5 - al, cf_k.y)
    plt.show()
    assert np.max(np.abs(a_k - a_R) < 1e-2)


def test_CFRconstK_against_CFK_eggbox():
    Kc = 0.4
    dK = 0.2

    sy = 0.25
    phase_shift = np.pi / 4
    fluctuation_length = sx / 4

    n = 32

    load = 0

    def kr(x, y):
        """
        the origin of the system is at the top of the siewave
        """
        return (2 * dK * np.sin(2 * np.pi * x / fluctuation_length + phase_shift) * np.cos(2 * np.pi * y / sy)) * Kc

    def dkr(x, y):
        return 2 * np.pi * np.cos(2 * np.pi * x / fluctuation_length + phase_shift) * np.cos(
            2 * np.pi * y / sy) * 2 * dK * Kc / fluctuation_length

    def kr_right(a, y, der="0"):
        if der == "0":
            return kr(a, y)
        elif der == "1":
            return dkr(a, y)

    def kr_left(a, y, der="0"):
        if der == "0":
            return kr(- a, y)
        elif der == "1":
            return - dkr(- a, y)

    cf_R = SinewaveCrackFrontLoadEnergyConstK(
        n, sy,
        kr_left=kr_left,
        kr_right=kr_right,
        w=Kc ** 2 / (2 * Es)
        )

    _plot = False
    if _plot:
        fig, ax = plt.subplots()

    a = np.ones(2 * n) * 0.25

    sol = trustregion_newton_cg(
        x0=a,
        gradient=lambda a: cf_R.gradient(a, load),
        hessian_product=lambda a, p: cf_R.hessian_product(p, a, load),
        trust_radius=0.25 * np.min(a),
        maxiter=3000,
        gtol=1e-11)

    a_R = sol.x

    cf_k = SinewaveCrackFrontLoad(
        n, sy,
        kc=lambda x, z: Kc - kr(x, z),
        dkc=lambda x, z: -dkr(x, z)
        )

    sol = trustregion_newton_cg(
        x0=a,
        gradient=lambda a: cf_k.gradient(a, load),
        hessian=lambda a: cf_k.hessian(a, load),
        trust_radius=0.25 * np.min(a),
        maxiter=3000,
        gtol=1e-11)

    a_k = sol.x

    if _plot:

        al = a_R[:n]
        ar = a_R[n:]

        ax.plot(+ ar, cf_R.y)
        ax.plot(- al, cf_R.y)

        al = a_k[:n]
        ar = a_k[n:]

        ax.plot(+ ar, cf_k.y)
        ax.plot(- al, cf_k.y)

        ax.set_xlim(-0.5, 0.5)

        workcmap = plt.get_cmap("coolwarm")
        x, y = np.meshgrid(np.arange(128) * (sx / 128) - 0.5, np.arange(n) * sy / n)
        ax.pcolormesh(x, y, kr(x, y), cmap=workcmap)

        plt.show()

    assert np.max(np.abs(a_k - a_R) < 1e-2)
