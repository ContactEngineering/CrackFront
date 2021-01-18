from CrackFront.Circular import SphereCrackFrontPenetrationBase, \
    NegativeRadiusError

import numpy as np


def stress_intensity_factor(contact_radius, Es=1., penetration=-1., der="0"):
    if der == "0":
        return - penetration * Es / np.sqrt(np.pi * contact_radius)
    elif der == "1_a":
        return penetration / (2 * np.sqrt(np.pi)) * contact_radius ** (-3 / 2)
    elif der == "2_a":
        return - 3 * penetration / (4 * np.sqrt(np.pi)) \
            * contact_radius ** (-5 / 2)


class FlatCrackFrontPenetrationLin(SphereCrackFrontPenetrationBase):
    def gradient(self, radius, penetration=-1.):
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
