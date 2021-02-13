
import numpy as np


from CrackFront.CircularFlat import FlatCircularExternalCrackPenetrationLin
from CrackFront.Optimization import trustregion_newton_cg


def test_rays():
    k = 1 / np.sqrt(np.pi)
    
    def kc(radius, angle, **params):
        return np.minimum((k / radius + params["sinewave_amplitude"] * np.cos(angle * params["n_rays"])), 10 * k)

    def dkc(radius, angle, **params):
        return - k / radius ** 2
    
    params = dict(
        sinewave_amplitude=0.05,
        n_rays=32)


    cf = FlatCircularExternalCrackPenetrationLin(512,
                                                 lambda radius, angle: kc(radius, angle, **params),
                                                 lambda radius, angle: dkc(radius, angle, **params))



    trustregion_newton_cg(
        np.ones(512),
        lambda a: cf.gradient(a, penetration=-1.),
        hessian_product=lambda a,p: cf.hessian_product(p, a, penetration=-1.),
        trust_radius=0.5,)