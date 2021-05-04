

import pytest
import numpy as np
from Adhesion.ReferenceSolutions import JKR
from matplotlib import pyplot as plt

from CrackFront.StraightForRoughness import SinewaveCrackFrontLoadEnergyConstK
from CrackFront.Straight import SinewaveCrackFrontLoad
from CrackFront.Optimization import trustregion_newton_cg


# nondimensional units
Es = 1 / np.pi
h = 1. # amplitude (half peak to valley) of the sinewave
sinewave_lambda = 1.
sx = 1.

def test_against_whet():
    """
    In the small roughness limit
    I can fully map the roughness to work of adhesion heterogeneity
    and compare to previous implementations
    """

    # Parameters
    Kc = 0.5  # in units of h pwfc. K / hp wfc = alpha
    kr_amplitude = 0.1 * Kc
    sy = 0.5
    load = 0
    n = 32

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
        hessian=lambda a : cf_k.hessian(a, load),
        trust_radius=0.25 * np.min(a),
        maxiter=3000,
        gtol=1e-11)

    a_k = sol.x

    fig, ax = plt.subplots()

    al = a_R[:n]
    ar = a_R[n:]

    ax.plot( .5 + ar, cf_R.y)
    ax.plot( 0.5 - al, cf_R.y)

    al = a_k[:n]
    ar = a_k[n:]

    ax.plot( .5 + ar, cf_k.y)
    ax.plot( 0.5 - al, cf_k.y)


    plt.show()