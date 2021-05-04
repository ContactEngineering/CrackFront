

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
    def __init__(self, n, sy, kr_right, kr_left,w):
        r"""
        .. math::

            U = \frac{1}{L} \int dz U^0(a(z)) + G_c \sum_n \frac{|q_n| L}{2}  \tilde a_n \tilde a_{-n}

            \frac{\delta U}{\delta a} = G^0(a(z))  + G_c \sum_n |q_n| \tilde a_n e^{i q_n z}

            \frac{\delta^2 U}{\delta a}

        Note that we do not implement the energy because U_0 is hard to compute in general


        Parameters
        ----------




        """

        dy = sy / n
        self.y = np.arange(n) * dy
        self.q = 2 * np.pi * np.fft.rfftfreq(n, sy / n)

        self.kr_right = kr_right
        self.kr_left = kr_left

        self.w = w

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
        K0r = KJr + self.kr_right(al, self.y)

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

        K0l = KJl + self.kr_left(al, self.y)
        K0r = KJr + self.kr_right(al, self.y)

        dG0l_al = self.kr_left(al, self.y, der="1") + K_J(a_s=al, a_o= ar, P=load, der="1_a_s")
        dG0l_ar = K_J(a_s=al, a_o= ar, P=load, der="1_a_o")

        dG0r_ar = self.kr_right(ar, self.y, der="1") + K_J(a_s=ar, a_o= al, P=load, der="1_a_s")
        dG0r_al = K_J(a_s=ar, a_o=al, P=load, der="1_a_o")

        ldl = dG0l_al * pl + self.w * self.elastic_hessp(pl)
        ldr = dG0l_ar * pr
        rdr = dG0r_ar * pr + self.w * self.elastic_hessp(pr)
        rdl = dG0r_al * pl

        return np.concatenate([ldl + ldr, rdr + rdl])