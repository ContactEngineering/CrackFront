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


# TODO:
#  - split fully linearized and less linearised in two classes
#  - implement hessian product as well.(maybe leave the matrix construction as
#    an option)
class SphereCrackFrontPenetration():
    def __init__(self, npx, kc, dkc, lin=False):
        """

        the nondimensionalisation assumes that R=1, w=1 / np.pi, Es=3. / 4,

        npx: number of pixels
        lin: bool, default False
            wether to linearize the K0(a, ...) term
        """

        # nondimensional units following Maugis Book:
        Es = 3 / 4
        w = 1 / np.pi
        R = 1.

        self.npx = npx
        self.angles = angle = np.arange(npx) * 2 * np.pi / npx
        self.nq = np.fft.rfftfreq(npx, 1 / npx)

        self._elastic_jacobian = None

        if lin:
            def gradient(radius, penetration):
                if (radius <= 0).any():
                    raise NegativeRadiusError
                a0 = np.mean(radius)
                return 1 / a0 * self.elastic_jacobian @ radius \
                    * JKR.stress_intensity_factor(contact_radius=a0, penetration=penetration, radius=R, contact_modulus=Es,) \
                    + JKR.stress_intensity_factor(contact_radius=a0, penetration=penetration, radius=R, contact_modulus=Es,) \
                    + JKR.stress_intensity_factor(contact_radius=a0, penetration=penetration, der="1_a") \
                    * (radius - a0) \
                       - kc(radius, angle)

            def hessian(radius, penetration):
                a0 = np.mean(radius)
                K = JKR.stress_intensity_factor(
                    contact_radius=a0,
                    penetration=penetration)
                return self.elastic_jacobian * K / a0 \
                    + np.diag((- K / a0 **2  + JKR.stress_intensity_factor(contact_radius=a0, penetration=penetration, radius=R, contact_modulus=Es, der="1_a")  / a0) / npx  * self.elastic_jacobian @ radius  \
                    + JKR.stress_intensity_factor(contact_radius=a0, penetration=penetration, radius=R, contact_modulus=Es, der="1_a") \
                    + JKR.stress_intensity_factor(contact_radius=a0, penetration=penetration, radius=R, contact_modulus=Es, der="2_a") / npx * (radius - a0) \
                    - dkc(radius, angle))
        else:
            def gradient(radius, penetration):
                if (radius <=0).any():
                    raise NegativeRadiusError
                a0 = np.mean(radius)
                return 1 / a0 * self.elastic_jacobian @ radius * JKR.stress_intensity_factor(contact_radius=a0, penetration=penetration, radius=R, contact_modulus=Es) \
                + JKR.stress_intensity_factor(contact_radius=radius, penetration=penetration, radius=R, contact_modulus=Es) - kc(radius, angle)

            def hessian(radius, penetration):
                a0 = np.mean(radius)
                K = JKR.stress_intensity_factor(contact_radius=a0, penetration=penetration, radius=R, contact_modulus=Es,)
                return self.elastic_jacobian * K / a0 \
                    + np.diag((- K / a0 **2  + JKR.stress_intensity_factor(contact_radius=a0, penetration=penetration, radius=R, contact_modulus=Es, der="1_a")  / a0) / npx  * self.elastic_jacobian @ radius  \
                    + JKR.stress_intensity_factor(contact_radius=radius, penetration=penetration, radius=R, contact_modulus=Es, der="1_a") \
                    - dkc(radius, angle))

        self.gradient = gradient
        self.hessian = hessian

    @property
    def elastic_jacobian(self):
        if self._elastic_jacobian is None:
            npx = self.npx
            elastic_jac = np.zeros((npx,npx))
            v = np.fft.irfft(self.nq/2, n=npx)
            for i in range(npx):
                for j in range(npx):
                    elastic_jac[i, j] = v[i-j]
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

        # Just for convenience:
        ncFrame.force = JKR.force(contact_radius=mean_radius,
                                  penetration=penetration)

        ncFrame.rms_radius = np.sqrt(np.mean((a - mean_radius)**2))
        ncFrame.min_radius = np.min(a)
        ncFrame.max_radius = np.max(a)

        # infos on convergence
        ncFrame.nit = sol.nit
        ncFrame.n_hits_boundary = sol.n_hits_boundary


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
        self.interpolator = field.interpolate_bicubic()

        if center is not None:
            self.center = center
        else:
            sx, sy = field.physical_sizes
            self.center = (sx / 2, sy / 2)

    def kc(self, radius, angle):
        """
        the origin of the system is at the sphere tip
        """
        x, y = pol2cart(radius, angle)
        return self.interpolator(x + self.center[0],
                                 y + self.center[1],
                                 derivative=0)

    def dkc(self, radius, angle):
        x, y = pol2cart(radius, angle)
        interp_field, interp_derx, interp_dery = self.interpolator(
            x + self.center[0], y + self.center[1], derivative=1)

        return interp_derx * np.cos(angle) + interp_dery * np.sin(angle)
