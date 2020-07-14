
import numpy as np
from Adhesion.ReferenceSolutions import JKR

K = JKR.stress_intensity_factor

class NegativeRadiusError(Exception):
    pass

# TODO:
#  - split fully linearized and less linearised in two classes
#  - implement hessian product as well.(maybe leave )
class SphereCrackFrontPenetration():
    def __init__(self, npx, kc, dkc, R=1, w=1 / np.pi, Es=3. / 4, lin=False):
        """

        npx: number of pixels
        lin: bool, default False
            wether to linearize the K0(a, ...) term
        """
        self.npx = npx
        self.angles = angle = np.arange(npx) * 2 * np.pi / npx
        nq = np.fft.rfftfreq(npx, 1 / npx)

        elastic_jac = np.zeros((npx,npx))
        v = np.fft.irfft(nq/2, n=npx)
        for i in range(npx):
            for j in range(npx):
                elastic_jac[i, j] = v[i-j]
        #check elastic jacobian
        a_test = np.random.normal(size=npx)
        np.testing.assert_allclose(elastic_jac @ a_test, np.fft.irfft(nq / 2 * np.fft.rfft(a_test), n=npx))

        self.elastic_jac = elastic_jac

        if lin:
            def gradient(radius, penetration):
                if (radius <=0).any():
                    raise NegativeRadiusError
                a0 = np.mean(radius)
                return 1 / a0 * elastic_jac @ radius \
                    * JKR.stress_intensity_factor(contact_radius=a0, penetration=penetration, radius=R, contact_modulus=Es,) \
                    + JKR.stress_intensity_factor(contact_radius=a0, penetration=penetration, radius=R, contact_modulus=Es,) \
                    + JKR.stress_intensity_factor(contact_radius=a0, penetration=penetration, radius=R, contact_modulus=Es, der="1_a") \
                    * (radius - a0) \
                       - kc(radius, angle)

            def hessian(radius, penetration):
                a0 = np.mean(radius)
                K = JKR.stress_intensity_factor(
                    contact_radius=a0,
                    penetration=penetration,
                    radius=R,
                    contact_modulus=Es)
                return elastic_jac * K / a0 \
                    + np.diag((- K / a0 **2  + JKR.stress_intensity_factor(contact_radius=a0, penetration=penetration, radius=R, contact_modulus=Es, der="1_a")  / a0) / npx  * elastic_jac @ radius  \
                    + JKR.stress_intensity_factor(contact_radius=a0, penetration=penetration, radius=R, contact_modulus=Es, der="1_a") \
                    + JKR.stress_intensity_factor(contact_radius=a0, penetration=penetration, radius=R, contact_modulus=Es, der="2_a") / npx * (radius - a0) \
                    - dkc(radius, angle))
        else:
            def gradient(radius, penetration):
                if (radius <=0).any():
                    raise NegativeRadiusError
                a0 = np.mean(radius)
                return 1 / a0 * elastic_jac @ radius * JKR.stress_intensity_factor(contact_radius=a0, penetration=penetration, radius=R, contact_modulus=Es) \
                + JKR.stress_intensity_factor(contact_radius=radius, penetration=penetration, radius=R, contact_modulus=Es) - kc(radius, angle)

            def hessian(radius, penetration):
                a0 = np.mean(radius)
                K = JKR.stress_intensity_factor(contact_radius=a0, penetration=penetration, radius=R, contact_modulus=Es,)
                return elastic_jac * K / a0 \
                    + np.diag((- K / a0 **2  + JKR.stress_intensity_factor(contact_radius=a0, penetration=penetration, radius=R, contact_modulus=Es, der="1_a")  / a0) / npx  * elastic_jac @ radius  \
                    + JKR.stress_intensity_factor(contact_radius=radius, penetration=penetration, radius=R, contact_modulus=Es, der="1_a") \
                    - dkc(radius, angle))

        self.gradient = gradient
        self.hessian = hessian

