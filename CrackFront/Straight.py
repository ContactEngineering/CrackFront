#
# Copyright 2020 Antoine Sanner
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

K_P = JKR.stress_intensity_factor_asymmetric


class SinewaveCrackFrontLoad():
    def __init__(self, n, sy, kc, dkc):

        dy = sy / n
        self.y = np.arange(n) * dy

        q = 2 * np.pi * np.fft.rfftfreq(n, sy / n)

        # Defining residual and jacobian
        elastic_jac = np.zeros((n, n))
        v = np.fft.irfft(q / 2, n=n)
        for i in range(n):
            for j in range(n):
                elastic_jac[i, j] = v[i - j]
        # check elastic jacobian
        a_test = np.random.normal(size=n)
        np.testing.assert_allclose(elastic_jac @ a_test,
                                   np.fft.irfft(q / 2 * np.fft.rfft(a_test),
                                                n=n))
        self.elastic_jac = elastic_jac
        self.kc = kc
        self.dkc = dkc

    def gradient(self, a, P):
        """
        This is Kel - Kc
        """
        n = int(len(a) / 2)
        al = a[:n]
        ar = a[n:]
        K0l = K_P(a_s=al, a_o=ar, P=P)
        K0r = K_P(a_s=ar, a_o=al, P=P)

        return np.concatenate(
            [(self.elastic_jac @ al * K_P(a_s=np.mean(al), a_o=np.mean(ar),
                                          P=P) + K0l - self.kc(-al, self.y)),
             self.elastic_jac @ ar * K_P(a_s=np.mean(ar), a_o=np.mean(al),
                                         P=P) + K0r - self.kc(ar, self.y)])

    def hessian(self, a, P):
        """
        this is dKel/da - dKc / da
        """

        n = int(len(a) / 2)
        al = a[:n]
        ar = a[n:]

        ldl = self.elastic_jac * K_P(a_s=np.mean(al), a_o=np.mean(ar), P=P) \
            + np.diag(
            self.elastic_jac @ al
            * K_P(a_s=np.mean(al), a_o=np.mean(ar), P=P, der="1_a_s") / n
            + K_P(a_s=al, a_o=ar, P=P, der="1_a_s")
            + self.dkc(-al, self.y))

        ldr = np.diag(
            self.elastic_jac @ al
            * K_P(a_s=np.mean(al), a_o=np.mean(ar), P=P, der="1_a_o") / n
            + K_P(a_s=al, a_o=ar, P=P, der="1_a_o"))
        rdr = self.elastic_jac * K_P(a_s=np.mean(ar), a_o=np.mean(al), P=P) \
            + np.diag(self.elastic_jac @ ar
                      * K_P(a_s=np.mean(ar), a_o=np.mean(al),
                            P=P, der="1_a_s") / n
                      + K_P(a_s=ar, a_o=al, P=P, der="1_a_s")
                      - self.dkc(ar, self.y))
        rdl = np.diag(self.elastic_jac @ ar
                      * K_P(a_s=np.mean(ar), a_o=np.mean(al),
                            P=P, der="1_a_o") / n
                      + K_P(a_s=ar, a_o=al, P=P, der="1_a_o"))

        return np.block([[ldl, ldr],
                         [rdl, rdr]])
