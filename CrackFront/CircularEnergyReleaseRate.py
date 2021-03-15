
import numpy as np
from Adhesion.ReferenceSolutions import JKR
from CrackFront.Circular import SphereCrackFrontPenetrationBase, NegativeRadiusError

# nondimensional units following Maugis Book:
Es = 3 / 4
w = 1 / np.pi
R = 1.

_jkrkwargs = dict(contact_modulus=Es, radius=R)

class SphereCrackFrontERRPenetrationLin(SphereCrackFrontPenetrationBase):
    def __init__(self, npx, w, dw):
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

        self.w = w
        self.dw = dw

    def gradient(self, radius, penetration):
        raise NotImplementedError

    def hessian_product(self, p, radius, penetration):
        raise NotImplementedError

    @property
    def elastic_jacobian(self):
        if self._elastic_jacobian is None:
            npx = self.npx
            elastic_jac = np.zeros((npx, npx))
            v = np.fft.irfft(self.nq, n=npx)
            for i in range(npx):
                for j in range(npx):
                    elastic_jac[i, j] = v[i - j]
            self._elastic_jacobian = elastic_jac
        return self._elastic_jacobian

    def elastic_hessp(self, a):
        return np.fft.irfft(self.nq * np.fft.rfft(a), n=self.npx)

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

        ncFrame.mean_w = np.mean(self.w(a, self.angles))

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

    def gradient(self, radius, penetration):
        if (radius <= 0).any():
            raise NegativeRadiusError
        a0 = np.mean(radius)
        sif = JKR.stress_intensity_factor(contact_radius=a0,
                                          penetration=penetration,
                                          **_jkrkwargs)
        return 1 / Es * ((1 + 1 / a0 * self.elastic_hessp(radius)) * sif ** 2 / 2
               + (radius - a0) * sif * JKR.stress_intensity_factor(contact_radius=a0,
                                                                   penetration=penetration,
                                                                   **_jkrkwargs,
                                                                   der="1_a")
                         ) \
               - self.w(radius, self.angles)

    def hessian(self, radius, penetration):
        raise NotImplemented
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
        dK = JKR.stress_intensity_factor(
            contact_radius=a0,
            penetration=penetration, **_jkrkwargs, der="1_a")

        d2K = JKR.stress_intensity_factor(
            contact_radius=a0,
            penetration=penetration, **_jkrkwargs, der="2_a")

        mp = np.mean(p)

        return 1/ Es * ( K ** 2 / 2 / a0 * self.elastic_hessp(p)
               + (
                       K / Es * dK * mp
                       + (dK ** 2 + K * d2K) * mp * (radius - a0)
                       + K * dK * (p - mp)
               ) \
               + self.elastic_hessp(radius) * (K * dK / a0 - K ** 2 / 2 / a0 ** 2) * mp
                         ) \
               - self.dw(radius, self.angles) * p