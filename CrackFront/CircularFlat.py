
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
