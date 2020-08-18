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
from Adhesion.ReferenceSolutions import JKR


def cart2pol(x1, x2):
    r = np.sqrt(x1 ** 2 + x2 ** 2)
    mask = r > 0
    phi = np.zeros_like(r)
    phi[mask] = np.arccos(x1[mask] / r[mask]) * np.sign(x2[mask])
    return r, phi


def pol2cart(radius, angle):
    return radius * np.cos(angle), radius * np.sin(angle)


class NegativeRadiusError(Exception):
    pass


# nondimensional units following Maugis Book:
Es = 3 / 4
w = 1 / np.pi
R = 1.

_jkrkwargs = dict(contact_modulus=Es, radius=R)


class SphereCrackFrontPenetration():

    def __init__(self, npx, kc, dkc):
        """

        the nondimensionalisation assumes that R=1, w=1 / np.pi, Es=3. / 4,

        npx: number of pixels
        lin: bool, default False
            wether to linearize the K0(a, ...) term
        """

        self.npx = npx
        self.angles = np.arange(npx) * 2 * np.pi / npx
        self.nq = np.fft.rfftfreq(npx, 1 / npx)

        self._elastic_jacobian = None

        self.kc = kc
        self.dkc = dkc

    def gradient(self, radius, penetration):
        if (radius <= 0).any():
            raise NegativeRadiusError
        a0 = np.mean(radius)
        return 1 / a0 * self.elastic_hessp(radius) \
            * JKR.stress_intensity_factor(contact_radius=a0,
                                          penetration=penetration,
                                          **_jkrkwargs) \
            + JKR.stress_intensity_factor(contact_radius=radius,
                                          penetration=penetration,
                                          **_jkrkwargs) \
            - self.kc(radius, self.angles)

    def hessian(self, radius, penetration):
        a0 = np.mean(radius)
        K = JKR.stress_intensity_factor(contact_radius=a0,
                                        penetration=penetration,
                                        **_jkrkwargs)

        return self.elastic_jacobian * K / a0 \
            + np.diag(self.elastic_jacobian @ radius / self.npx *
                      (- K / a0 ** 2
                       + JKR.stress_intensity_factor(contact_radius=a0,
                                                     penetration=penetration,
                                                     **_jkrkwargs, der="1_a")
                       / a0
                       )
                      + JKR.stress_intensity_factor(contact_radius=radius,
                                                    penetration=penetration,
                                                    **_jkrkwargs, der="1_a")
                      - self.dkc(radius, self.angles)
                      )

    def hessian_product(self, p, radius, penetration):
        """
        computes efficiently the hessian product
        :math:`H(radius, penetration) p`
        """
        a0 = np.mean(radius)
        K = JKR.stress_intensity_factor(contact_radius=a0,
                                        penetration=penetration, **_jkrkwargs)
        return (
                K / a0 * self.elastic_hessp(p)
                + (self.elastic_hessp(radius) / self.npx *
                   (- K / a0 ** 2
                    + JKR.stress_intensity_factor(contact_radius=a0,
                                                  penetration=penetration,
                                                  **_jkrkwargs, der="1_a")
                    / a0
                    )
                   + JKR.stress_intensity_factor(contact_radius=radius,
                                                 penetration=penetration,
                                                 **_jkrkwargs, der="1_a")
                   - self.dkc(radius, self.angles)
                   ) * p
        )

    @property
    def elastic_jacobian(self):
        if self._elastic_jacobian is None:
            npx = self.npx
            elastic_jac = np.zeros((npx, npx))
            v = np.fft.irfft(self.nq / 2, n=npx)
            for i in range(npx):
                for j in range(npx):
                    elastic_jac[i, j] = v[i - j]
            self._elastic_jacobian = elastic_jac
        return self._elastic_jacobian

    def elastic_hessp(self, a):
        return np.fft.irfft(self.nq / 2 * np.fft.rfft(a), n=self.npx)

    def dump(self, ncFrame, penetration, sol):
        """
        Writes the results of the current solution into the ncFrame

        this assumes the trust-region-newton-cg has been used.

        Parameters
        ----------
        ncFrame:
            frame to the NCStructuredGrid
        penetration:
            current penetration value
        sol:
            output of the minimizer
            `CrackFront.Optimization.trustregion_newton_cg``
        """

        a = sol.x

        ncFrame.penetration = penetration
        ncFrame.radius = a
        ncFrame.mean_radius = mean_radius = np.mean(a)
        ncFrame.contact_area = np.pi * np.mean(a**2)
        # Just for convenience:
        ncFrame.force = JKR.force(contact_radius=mean_radius,
                                  penetration=penetration)

        ncFrame.rms_radius = np.sqrt(np.mean((a - mean_radius)**2))
        ncFrame.min_radius = np.min(a)
        ncFrame.max_radius = np.max(a)

        # infos on convergence
        ncFrame.nit = sol.nit
        ncFrame.n_hits_boundary = sol.n_hits_boundary
        ncFrame.njev = sol.njev
        ncFrame.nhev = sol.nhev


class SphereCrackFrontPenetrationMe(SphereCrackFrontPenetration):
    def gradient(self, radius, penetration):
        if (radius <= 0).any():
            raise NegativeRadiusError
        return ( 1 / radius * self.elastic_hessp(radius) + 1) \
           * JKR.stress_intensity_factor(contact_radius=radius,
                                          penetration=penetration,
                                          **_jkrkwargs) \
            - self.kc(radius, self.angles)

    def hessian(self, radius, penetration):
        raise NotImplementedError

    def hessian_product(self, p, radius, penetration):
        """
        computes efficiently the hessian product
        :math:`H(radius, penetration) p`
        """
        K = JKR.stress_intensity_factor(contact_radius=radius,
                                        penetration=penetration, **_jkrkwargs)
        hesspr = self.elastic_hessp(radius)
        return (
                K * (- hesspr / radius ** 2 * p
                     + self.elastic_hessp(p) / radius)
                + JKR.stress_intensity_factor(contact_radius=radius,
                                              penetration=penetration,
                                              **_jkrkwargs, der="1_a")
                * (1 + hesspr / radius) * p
                - self.dkc(radius, self.angles)
        )

class SphereCrackFrontPenetrationLin(SphereCrackFrontPenetration):
    def gradient(self, radius, penetration):
        if (radius <= 0).any():
            raise NegativeRadiusError
        a0 = np.mean(radius)
        return 1 / a0 * self.elastic_hessp(radius) \
            * JKR.stress_intensity_factor(contact_radius=a0,
                                          penetration=penetration,
                                          **_jkrkwargs) \
            + JKR.stress_intensity_factor(contact_radius=a0,
                                          penetration=penetration,
                                          **_jkrkwargs) \
            + JKR.stress_intensity_factor(contact_radius=a0,
                                          penetration=penetration,
                                          **_jkrkwargs,
                                          der="1_a") \
            * (radius - a0) \
            - self.kc(radius, self.angles)

    def hessian(self, radius, penetration):
        a0 = np.mean(radius)
        K = JKR.stress_intensity_factor(
            contact_radius=a0,
            penetration=penetration, **_jkrkwargs)

        return self.elastic_jacobian * K / a0 \
            + np.diag((- K / a0 ** 2
                       + JKR.stress_intensity_factor(contact_radius=a0,
                                                     penetration=penetration,
                                                     **_jkrkwargs,
                                                     der="1_a")
                       / a0
                       ) / self.npx * self.elastic_jacobian @ radius
                      + JKR.stress_intensity_factor(contact_radius=a0,
                                                    penetration=penetration,
                                                    **_jkrkwargs, der="1_a")
                      + JKR.stress_intensity_factor(contact_radius=a0,
                                                    penetration=penetration,
                                                    **_jkrkwargs, der="2_a")
                      / self.npx * (radius - a0)
                      - self.dkc(radius, self.angles)
                      )

    def hessian_product(self, p, radius, penetration):
        a0 = np.mean(radius)
        K = JKR.stress_intensity_factor(
            contact_radius=a0,
            penetration=penetration, **_jkrkwargs)

        return K / a0 * self.elastic_hessp(p) \
            + ((- K / a0 ** 2
                + JKR.stress_intensity_factor(contact_radius=a0,
                                              penetration=penetration,
                                              **_jkrkwargs,
                                              der="1_a")
                / a0
                ) / self.npx * self.elastic_hessp(radius)
               + JKR.stress_intensity_factor(contact_radius=a0,
                                             penetration=penetration,
                                             **_jkrkwargs, der="1_a")
               + JKR.stress_intensity_factor(contact_radius=a0,
                                             penetration=penetration,
                                             **_jkrkwargs, der="2_a")
               / self.npx * (radius - a0)
               - self.dkc(radius, self.angles)
               ) * p


class Interpolator():
    """
    wraps the cartesion bicubic interpolater provided by a
    `SurfaceTopography.Topography` into polar coordinates
    """
    def __init__(self, field, center=None):
        """
        field:
            SurfaceTopography.topography instance
        """
        self.field = field
        self.interpolator = field.interpolate_bicubic()

        if center is not None:
            self.center = center
        else:
            sx, sy = field.physical_sizes
            self.center = (sx / 2, sy / 2)

    def kc_polar(self, radius, angle):
        """
        the origin of the system is at the sphere tip
        """
        x, y = pol2cart(radius, angle)
        return self.interpolator(x + self.center[0],
                                 y + self.center[1],
                                 derivative=0)

    def dkc_polar(self, radius, angle):
        x, y = pol2cart(radius, angle)
        interp_field, interp_derx, interp_dery = self.interpolator(
            x + self.center[0], y + self.center[1], derivative=1)

        return interp_derx * np.cos(angle) + interp_dery * np.sin(angle)
