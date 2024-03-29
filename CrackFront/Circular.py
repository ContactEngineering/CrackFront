#
# Copyright 2020-2021 Antoine Sanner
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


def cart2pol(x,y):
    z = x + 1j * y
    return abs(z), np.angle(z)


def pol2cart(radius, angle):
    return radius * np.cos(angle), radius * np.sin(angle)


class NegativeRadiusError(Exception):
    pass

class RadiusTooLowError(Exception):
    pass

# nondimensional units following Maugis Book:
Es = 3 / 4
w = 1 / np.pi
R = 1.

_jkrkwargs = dict(contact_modulus=Es, radius=R)


class SphereCrackFrontPenetrationBase():
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
        raise NotImplementedError

    def hessian_product(self, p, radius, penetration):
        raise NotImplementedError

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

    @staticmethod
    def evaluate_normal_force(contact_radius, penetration):
        return SphereCrackFrontPenetrationBase._evaluate_first_order_normal_force(contact_radius, penetration) \
            + SphereCrackFrontPenetrationBase._evaluate_normal_force_correction(
            contact_radius, penetration)

    @staticmethod
    def _evaluate_first_order_normal_force(contact_radius, penetration):
        """
        This is just JKR applied to the mean contact radius
        """
        return JKR.force(contact_radius = np.mean(contact_radius), penetration=penetration)

    def dump(self, ncFrame, penetration, sol, dump_fields=True):
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
        if dump_fields:
            ncFrame.radius = a
        ncFrame.mean_radius = mean_radius = np.mean(a)

        ncFrame.mean_Kc = np.mean(self.kc(a, self.angles))

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


class SphereCrackFrontPenetrationIntermediate(SphereCrackFrontPenetrationBase):
    r"""

    Gao and Rices first order perturbation of the stress intensity
    without linearizing the circular reference but linearizing the nonlocal elasticity term

    .. math ::

        K(a, \theta) = K_\mathrm{J}(a(\theta))
        + \frac{1}{\tilde a_0} K_\mathrm{J}(\tilde a_0)
        \sum \limits_{n \in \mathbb{Z}^{\backslash \{0\}}}  \frac{n}{2} \ \tilde a_n \ e^{i n \theta}

    """
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


class SphereCrackFrontPenetrationFull(SphereCrackFrontPenetrationBase):
    r"""

    Gao and Rices first order perturbation of the stress intensity factor without introducing any
    additional linearization.

    .. math ::

        K(a, \theta) = K_\mathrm{J}(a(\theta))
        \left( 1 + \frac{1}{a(\theta)}
        \sum \limits_{n \in \mathbb{Z}^{\backslash \{0\}}}  \frac{n}{2} \ \tilde a_n \ e^{i n \theta}
        \right)

    """
    def gradient(self, radius, penetration):
        if (radius <= 0).any():
            raise NegativeRadiusError
        return (1 / radius * self.elastic_hessp(radius) + 1) \
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
                - self.dkc(radius, self.angles) * p
        )


class SphereCrackFrontPenetrationLin(SphereCrackFrontPenetrationBase):
    r"""
    Complete linearisation of the stress intensity factor.

    This is the closest to Gao and Rice's expressions

    .. math ::

        K(a, \theta) = K_\mathrm{J}(\tilde a_0)
        + \frac{\partial K_\mathrm{J}}{\partial a}(\tilde a_0) \left(a(\theta) - \tilde a_0\right)
        + \frac{K_\mathrm{J}(\tilde a_0)}{\tilde a_0}
        \sum \limits_{n \in \mathbb{Z}^{\backslash \{0\}}}  \frac{n}{2} \ \tilde a_n \ e^{i n \theta}

    """
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
    # TODO: I have doubts on the correctness of this. See the ERR version.

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

    def field_polar(self, radius, angle):
        """
        the origin of the system is at the sphere tip
        """
        x, y = pol2cart(radius, angle)
        return self.interpolator(x + self.center[0],
                                 y + self.center[1],
                                 derivative=0)

    def dfield_dr_polar(self, radius, angle):
        x, y = pol2cart(radius, angle)
        interp_field, interp_derx, interp_dery = self.interpolator(
            x + self.center[0], y + self.center[1], derivative=1)

        return interp_derx * np.cos(angle) + interp_dery * np.sin(angle)
