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
from Adhesion.ReferenceSolutions.sinewave import JKR
K_J = JKR.stress_intensity_factor_asymmetric

#################### Paramters from JKR nondimensionalisation of the sinewave
Es = 1 / np.pi
h = 1. # amplitude (half peak to valley) of the sinewave
sinewave_lambda = 1.
sx = 1.

#w = # or alpha^2 / (2 E^*)

class SinewaveCrackFrontLoadEnergyConstK:
    r"""

        Crack front equation for the contact of a periodic sinusoidal indenter against a rough surface

        For each the left and right crack front

        .. math::

            U = \frac{1}{L} \int dz U^0(a(z)) + G_c \sum_n \frac{|q_n| L}{2}  \tilde a_n \tilde a_{-n}

            \frac{\delta U}{\delta a} = G^0(a(z))  + G_c \sum_n |q_n| \tilde a_n e^{i q_n z}

            \frac{\delta^2 U}{\delta a}

        Note that we do not implement the energy because U_0 is hard to compute in general.

        :math:`G_0` is the energy release rate below the straight crack:

        .. math::

            G_0 = \frac{1}{2E^*} \left(K_{\rm J}(a) + K_{\rm R}(a, z) \right)^2

        :math:`K_{\rm J}` is the stress intensity factor in the contact of a sinewave against a flat elastic halfspace.py

        Note that it depends on the mean position of the left and right crack.
        The assymmetric solution for the sinewave contact was provided by
        Carbone, G., Mangialardi, L., 2004. Adhesion and friction of an elastic half-space in contact with a slightly wavy rigid surface. Journal of the Mechanics and Physics of Solids 52, 1267–1287. https://doi.org/10/bjmwjw



    """

    def __init__(self, n, sy, kr_right, kr_left, w):
        r"""
        Parameters
        ----------
        n: int
            number of collocation points on each crack front

        kr_right: callable
            When the right crack front is straight and at distance `a` from the tip of the sinusoidal indenter,
            `kr_right(a, z)` is the stress intensity factor at `z`

        kr_left: callable
            When the left crack front is straight and at distance `a` from the tip of the sinusoidal indenter,
            `kr_left(a, z)` is the stress intensity factor at `z`

        w: float
            work of adhesion, i.e. :math:`G_c=K_c^2 / 2 E^*`
        """

        dy = sy / n
        self.y = np.arange(n) * dy
        self.q = 2 * np.pi * np.fft.rfftfreq(n, sy / n)

        self.kr_right = kr_right
        self.kr_left = kr_left

        self.w = w
        self.npx = n

    def elastic_hessp(self, a):
        r"""

        computes the elastic hessian product with one crack shape

        .. math ::
            \sum_n |q_n| \tilde a_n e^{i q_n z}
        """
        return np.fft.irfft(self.q * np.fft.rfft(a), n=self.npx)

    def gradient(self, a, load):
        """
        Returns the gradient, i.e. the energy release rate
        """
        n = int(len(a) / 2)
        al = a[:n]
        ar = a[n:]
        KJl = K_J(a_s=al, a_o=ar, P=load)
        KJr = K_J(a_s=ar, a_o=al, P=load)

        K0l = KJl + self.kr_left(al, self.y)
        K0r = KJr + self.kr_right(ar, self.y)

        G0l = K0l ** 2 / (2 * Es)
        G0r = K0r ** 2 / (2 * Es)

        return np.concatenate(
            [G0l + self.w * self.elastic_hessp(al) - self.w,
             G0r + self.w * self.elastic_hessp(ar) - self.w])

    def hessian_product(self, p, a, load):
        n = int(len(a) / 2)
        al = a[:n]
        ar = a[n:]

        pl = p[:n]
        pr = p[n:]

        KJl = K_J(a_s=al, a_o=ar, P=load)
        KJr = K_J(a_s=ar, a_o=al, P=load)

        dG0l_al = self.kr_left(al, self.y, der="1") + K_J(a_s=al, a_o=ar, P=load, der="1_a_s")
        dG0l_ar = K_J(a_s=al, a_o=ar, P=load, der="1_a_o")

        dG0r_ar = self.kr_right(ar, self.y, der="1") + K_J(a_s=ar, a_o=al, P=load, der="1_a_s")
        dG0r_al = K_J(a_s=ar, a_o=al, P=load, der="1_a_o")

        ldl = dG0l_al * pl + self.w * self.elastic_hessp(pl)
        ldr = dG0l_ar * pr
        rdr = dG0r_ar * pr + self.w * self.elastic_hessp(pr)
        rdl = dG0r_al * pl

        return np.concatenate([ldl + ldr, rdr + rdl])