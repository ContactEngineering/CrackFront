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

"""

penetration = -1.
Es = 1.


"""
import numpy as np

from CrackFront.Circular import SphereCrackFrontPenetrationBase, NegativeRadiusError


def stress_intensity_factor(contact_radius, Es=1., penetration=-1., der="0"):
    """

    Parameters
    ----------
    contact_radius : array_like of floats
    Es : array_like of floats, optional
        (Default 1.)
    penetration : array_like of floats, optional
        (Default -1.)
    der : {"0", "1_a"}
        order of the derivative
    Returns
    -------
    stress_intensity_factor : array_like of floats
        stress intensity factor

        if Es and penetration are not provided, it is in units of

    """
    if der == "0":
        return - penetration * Es / np.sqrt(np.pi * contact_radius)
    elif der == "1_a":
        return penetration * Es / (2 * np.sqrt(np.pi)) * contact_radius ** (- 3 / 2)
    elif der == "2_a":
        return - penetration * 3 * Es / (4 * np.sqrt(np.pi)) * contact_radius ** (- 5 / 2)
    else:
        raise ValueError(f"Derivative {der} not implemented")


class FlatCircularExternalCrackPenetrationLin(SphereCrackFrontPenetrationBase):
    def gradient(self, radius, penetration):
        if (radius <= 0).any():
            raise NegativeRadiusError
        a0 = np.mean(radius)
        return 1 / a0 * self.elastic_hessp(radius) \
            * stress_intensity_factor(contact_radius=a0,
                                      penetration=penetration,
                                      Es=1.) \
            + stress_intensity_factor(contact_radius=a0,
                                      penetration=penetration,
                                      Es=1.) \
            + stress_intensity_factor(contact_radius=a0,
                                      penetration=penetration,
                                      Es=1.,
                                      der="1_a") \
            * (radius - a0) \
            - self.kc(radius, self.angles)

    def hessian(self, radius, penetration):
        a0 = np.mean(radius)
        K = stress_intensity_factor(
            contact_radius=a0,
            penetration=penetration, Es=1.)

        return self.elastic_jacobian * K / a0 \
            + np.diag((- K / a0 ** 2
                       + stress_intensity_factor(contact_radius=a0,
                                                 penetration=penetration,
                                                 Es=1.,
                                                 der="1_a")
                       / a0
                       ) / self.npx * self.elastic_jacobian @ radius
                      + stress_intensity_factor(contact_radius=a0,
                                                penetration=penetration,
                                                Es=1., der="1_a")
                      + stress_intensity_factor(contact_radius=a0,
                                                penetration=penetration,
                                                Es=1., der="2_a")
                      / self.npx * (radius - a0)
                      - self.dkc(radius, self.angles)
                      )

    def hessian_product(self, p, radius, penetration):
        a0 = np.mean(radius)
        K = stress_intensity_factor(
            contact_radius=a0,
            penetration=penetration, Es=1.)

        return K / a0 * self.elastic_hessp(p) \
            + ((- K / a0 ** 2
                + stress_intensity_factor(contact_radius=a0,
                                          penetration=penetration,
                                          Es=1.,
                                          der="1_a")
                / a0
                ) / self.npx * self.elastic_hessp(radius)
               + stress_intensity_factor(contact_radius=a0,
                                         penetration=penetration,
                                         Es=1., der="1_a")
               + stress_intensity_factor(contact_radius=a0,
                                         penetration=penetration,
                                         Es=1., der="2_a")
               / self.npx * (radius - a0)
               - self.dkc(radius, self.angles)
               ) * p
