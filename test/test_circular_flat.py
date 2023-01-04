#
# Copyright 2021 Antoine Sanner
#           2021 sanner.antoine@laposte.net
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