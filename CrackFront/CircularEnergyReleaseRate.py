import numpy as np
from Adhesion.ReferenceSolutions import JKR
from CrackFront.Circular import SphereCrackFrontPenetrationBase, NegativeRadiusError, RadiusTooLowError
from scipy.optimize import OptimizeResult

# nondimensional units following Maugis Book:
Es = 3 / 4
w = 1 / np.pi
R = 1.

_jkrkwargs = dict(contact_modulus=Es, radius=R)


class SphereCrackFrontERRPenetrationLin(SphereCrackFrontPenetrationBase):
    r"""

    .. math ::

        G(\theta) = G_\mathrm{J}(\tilde a_0)
        + \frac{\partial G_\mathrm{J}(\tilde a_0)}{\partial a} (a(\theta) - \tilde a_0)
        + \frac{G_\mathrm{J}(\tilde a_0)}{\tilde a_0} \sum_{n \in \mathbf{Z}} |n| e^{i n \theta} \tilde a_n

    """
    def __init__(self, npx, w=None, dw=None, kc=None, dkc=None):
        """

        the nondimensionalisation assumes that R=1, w=1 / np.pi, Es=3. / 4,

        Either the pair of callables w, dw or kc, dkc have to be provided.

        Parameters:
        -----------
        npx: number of pixels
        lin: bool, default False
            wether to linearize the K0(a, ...) term

        """

        self.npx = npx
        self.angles = np.arange(npx) * 2 * np.pi / npx
        self.nq = np.fft.rfftfreq(npx, 1 / npx)

        self._elastic_jacobian = None

        if w is None and dw is None:
            if dkc is not None and kc is not None:
                def w(radius, angle):
                    return kc(radius, angle) ** 2 / (2 * Es)

                def dw(radius, angle):
                    return dkc(radius, angle) * kc(radius, angle) / Es
            else:
                raise ValueError
        elif dkc is not None or kc is not None:
            raise ValueError

        self.w = w
        self.dw = dw

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
        # In the rfft nq is positive only
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

        ncFrame.contact_area = np.pi * np.mean(a ** 2)
        # Just for convenience:
        ncFrame.force_first_order = self._evaluate_first_order_normal_force(contact_radius=a, penetration=penetration)
        ncFrame.force = self.evaluate_normal_force(contact_radius=a, penetration=penetration)

        ncFrame.rms_radius = np.sqrt(np.mean((a - mean_radius) ** 2))
        ncFrame.min_radius = np.min(a)
        ncFrame.max_radius = np.max(a)

        # infos on convergence
        ncFrame.nit = sol.nit
        ncFrame.n_hits_boundary = sol.n_hits_boundary
        ncFrame.njev = sol.njev
        ncFrame.nhev = sol.nhev

    @staticmethod
    def evaluate_normal_force(contact_radius, penetration):
        """
        This is valid up to TODO order in the contact radius flucutations

        I think this is not completely Order ∆a^3 because of the additional linearizations from moving from K to G

        """
        return SphereCrackFrontPenetrationBase._evaluate_first_order_normal_force(contact_radius, penetration) \
            + SphereCrackFrontERRPenetrationLin._evaluate_normal_force_correction(
            contact_radius, penetration)

    @staticmethod
    def _evaluate_normal_force_correction(contact_radius, penetration):
        """

        All parameters are in the JKR units

        Parameters:
        -----------
        contact_radius: np.ndarray of float
            local contact radii along the colocation points of the front
        penetration: float
            rigid body penetration of the spherical indenter
        """
        # see notes of the 210315
        # see notes of the 210401

        a0 = np.mean(contact_radius)
        da = contact_radius - a0
        a2 = np.sum(da ** 2)
        npx = len(contact_radius)

        pixel_size = 2 * np.pi * a0 / npx

        nq = np.fft.rfftfreq(npx, 1 / npx)
        q = nq / a0
        aQa = np.sum(da * np.fft.irfft(q * np.fft.rfft(da)))
        # TODO: This might be wrong:
        #  shouldn't I weight part of this doubly because of the lack of the symmetrics in the rfft ?
        # Ah no it's ok since I do the sum in real space
        return 0.5 * pixel_size * (
                    JKR.nonequilibrium_elastic_energy_release_rate(contact_radius=a0, penetration=penetration,
                                                                   der="2_da") * a2
                    + JKR.nonequilibrium_elastic_energy_release_rate(contact_radius=a0, penetration=penetration,
                                                                     der="1_d") * aQa) \
            / np.pi  # because the ERR is in unit of w, the expression above is in unit of w R.
        # We divide it by pi so that it is in unit of pi w R , i.e. the JKR units

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
        raise NotImplementedError
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

        return 1 / Es * (K ** 2 / 2 / a0 * self.elastic_hessp(p)
                         + (
                                 K * dK * mp
                                 + (dK ** 2 + K * d2K) * mp * (radius - a0)
                                 + K * dK * (p - mp)
                         )
                         + self.elastic_hessp(radius) * (K * dK / a0 - K ** 2 / 2 / a0 ** 2) * mp
                         ) \
            - self.dw(radius, self.angles) * p


class SphereCrackFrontERRPenetrationFull(SphereCrackFrontERRPenetrationLin):
    r"""

    Linear perturbation of the energy release rate but witout any unnecessary
    additional linearization

    .. math ::

        G(\theta) = G_\mathrm{J}(a(\theta)) \left( 1
        + \frac{1}{\tilde a(\theta)} \sum_{n \in \mathbf{Z}} |n| e^{i n \theta} \tilde a_n
        \right)

    """
    def gradient(self, radius, penetration):
        if (radius <= 0).any():
            raise NegativeRadiusError

        sif = JKR.stress_intensity_factor(contact_radius=radius,
                                          penetration=penetration,
                                          **_jkrkwargs)
        return 1 / Es * (1 + 1 / radius * self.elastic_hessp(radius)) * sif ** 2 / 2 \
            - self.w(radius, self.angles)

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

    def hessian_product(self, p, radius, penetration):

        K = JKR.stress_intensity_factor(
            contact_radius=radius,
            penetration=penetration, **_jkrkwargs)
        dK = JKR.stress_intensity_factor(
            contact_radius=radius,
            penetration=penetration, **_jkrkwargs, der="1_a")

        el_hessp = self.elastic_hessp(radius)
        return 1 / Es * (K ** 2 / 2 / radius * self.elastic_hessp(p)
                         + (
                                 K * dK * (1 + el_hessp / radius)
                                 - K ** 2 / 2 * el_hessp / radius ** 2
                         ) * p
                         ) \
            - self.dw(radius, self.angles) * p


class SphereCrackFrontERRPenetrationEnergy(SphereCrackFrontPenetrationBase):
    r"""
    Here we guessed an expression for the energy that has the correct first derivative of the energy release rate.

    The gradient is __not__ the ERR G, but :math:`\frac{\partial U}{\partial a_j} = \Delta \theta a_j G(a_j)`

    the nondimensionalisation assumes that R=1, w=1 / np.pi, Es=3. / 4,

    Either the pair of callables w, dw or kc, dkc have to be provided.

    Parameters:
    -----------
    npx: number of pixels
    lin: bool, default False
        wether to linearize the K0(a, ...) term

    """

    def __init__(self, npx, w=None, dw=None, kc=None, dkc=None):

        # TODO: enable to provide the integral of the work of adhesion field
        self.npx = npx
        self.angles = np.arange(npx) * 2 * np.pi / npx
        self.nq = np.fft.rfftfreq(npx, 1 / npx)

        self._elastic_jacobian = None

        if w is None and dw is None:
            if dkc is not None and kc is not None:
                def w(radius, angle):
                    return kc(radius, angle) ** 2 / (2 * Es)

                def dw(radius, angle):
                    return dkc(radius, angle) * kc(radius, angle) / Es
            else:
                raise ValueError
        elif dkc is not None or kc is not None:
            raise ValueError

        self.w = w
        self.dw = dw

    @staticmethod
    def _n_an_2(contact_radius):
        npx = len(contact_radius)
        nq = np.fft.rfftfreq(npx, 1 / npx)
        fourier_scalar_prod_factors = np.ones(npx // 2 + 1) * 2
        fourier_scalar_prod_factors[0] = 1
        if npx % 2 == 0:
            fourier_scalar_prod_factors[-1] == 1
        a_fourier = np.fft.rfft(contact_radius, norm="forward")
        return np.vdot(a_fourier * nq * fourier_scalar_prod_factors, a_fourier).real

    def elastic_energy(self, contact_radius, penetration):
        # factors for the fourier space scalar product with rfft

        a0 = np.mean(contact_radius)
        return np.mean(JKR.nonequilibrium_elastic_energy(contact_radius=contact_radius, penetration=penetration)) \
            + np.pi * JKR.nonequilibrium_elastic_energy_release_rate(penetration=penetration,
                                                                     contact_radius=a0) \
            * self._n_an_2(contact_radius)

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

    def dump(self, ncFrame, penetration, a, dump_fields=True):
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

        ncFrame.penetration = penetration
        if dump_fields:
            ncFrame.radius = a
        ncFrame.mean_radius = mean_radius = np.mean(a)

        ncFrame.mean_w = np.mean(self.w(a, self.angles))

        ncFrame.contact_area = np.pi * np.mean(a ** 2)
        # Just for convenience:
        ncFrame.force_first_order = self._evaluate_first_order_normal_force(contact_radius=a, penetration=penetration)
        ncFrame.force = self.evaluate_normal_force(contact_radius=a, penetration=penetration)

        ncFrame.rms_radius = np.sqrt(np.mean((a - mean_radius) ** 2))
        ncFrame.min_radius = np.min(a)
        ncFrame.max_radius = np.max(a)

    @staticmethod
    def evaluate_normal_force(contact_radius, penetration):
        """
        """
        return SphereCrackFrontERRPenetrationEnergy._evaluate_normal_force_naive(contact_radius, penetration) \
            + SphereCrackFrontERRPenetrationEnergy._evaluate_normal_force_correction(contact_radius, penetration)

    @staticmethod
    def _evaluate_normal_force_naive(contact_radius, penetration):
        return np.mean(JKR.force(contact_radius=contact_radius, penetration=penetration))

    @staticmethod
    def _evaluate_normal_force_correction(contact_radius, penetration):
        """

        All parameters are in the JKR units

        Parameters:
        -----------
        contact_radius: np.ndarray of float
            local contact radii along the colocation points of the front
        penetration: float
            rigid body penetration of the spherical indenter
        """
        # see notes of the 210408
        a0 = np.mean(contact_radius)
        return np.pi * JKR.nonequilibrium_elastic_energy_release_rate(penetration=penetration,
                                                                      contact_radius=a0,
                                                                      der="1_d") \
            * SphereCrackFrontERRPenetrationEnergy._n_an_2(contact_radius)

    def gradient(self, radius, penetration):
        if (radius <= 0).any():
            raise NegativeRadiusError
        a0 = np.mean(radius)
        eerr_j = JKR.nonequilibrium_elastic_energy_release_rate(
            contact_radius=radius,
            penetration=penetration,
            **_jkrkwargs)
        eerr_0 = JKR.nonequilibrium_elastic_energy_release_rate(
            contact_radius=a0,
            penetration=penetration,
            **_jkrkwargs)
        deerr_da_0 = JKR.nonequilibrium_elastic_energy_release_rate(
            contact_radius=a0,
            penetration=penetration,
            **_jkrkwargs, der="1_a")
        return 2 * np.pi / self.npx * (
                radius * eerr_j
                + eerr_0 * self.elastic_hessp(radius)
                + 0.5 * deerr_da_0 * self._n_an_2(radius)
                - self.w(radius, self.angles) * radius
        )

    def hessian(self, radius, penetration):
        raise NotImplementedError

    def hessian_product(self, p, radius, penetration):
        a0 = np.mean(radius)
        eerr_j = JKR.nonequilibrium_elastic_energy_release_rate(
            contact_radius=radius,
            penetration=penetration,
            **_jkrkwargs)
        eerr_0 = JKR.nonequilibrium_elastic_energy_release_rate(
            contact_radius=a0,
            penetration=penetration,
            **_jkrkwargs)
        deerr_da_0 = JKR.nonequilibrium_elastic_energy_release_rate(
            contact_radius=a0,
            penetration=penetration,
            **_jkrkwargs, der="1_a")
        deerr_da2_0 = JKR.nonequilibrium_elastic_energy_release_rate(
            contact_radius=a0,
            penetration=penetration,
            **_jkrkwargs, der="2_a")
        deerr_da_j = JKR.nonequilibrium_elastic_energy_release_rate(
            contact_radius=radius,
            penetration=penetration,
            **_jkrkwargs, der="1_a")

        elHa = self.elastic_hessp(radius)

        return 2 * np.pi / self.npx * (
            (eerr_j + deerr_da_j * radius) * p
            + 1 / self.npx * deerr_da_0 * (np.sum(elHa * p) + elHa * np.sum(p))
            + eerr_0 * self.elastic_hessp(p)
            + 0.5 / self.npx * deerr_da2_0 * self._n_an_2(radius) * p
            - (self.w(radius, self.angles) + radius * self.dw(radius, self.angles)) * p
        )


class SphereCrackFrontERRPenetrationEnergyConstGc(SphereCrackFrontERRPenetrationEnergy):
    def __init__(self, npx, w=None, dw=None, kc=None, dkc=None, wm=1/np.pi):
        r"""

        Simplification where the prefactor of the deformation energy term is approximated constant

        This is especially convenient appropriate when simulating the effect of roughness, where we approximately know
        that the final position has a constant work of adhesion.

        Parameters:
        -----------
        npx: number of pixels
        wm: average or nominal work of adhesion. This determines the stiffness of the contact line
        """

        # TODO: enable to provide the integral of the work of adhesion field
        self.wm = wm
        super().__init__(npx, w, dw, kc, dkc)

    def elastic_energy(self, contact_radius, penetration):
        # factors for the fourier space scalar product with rfft

        return np.mean(JKR.nonequilibrium_elastic_energy(contact_radius=contact_radius, penetration=penetration)) \
            + np.pi * self.wm * self._n_an_2(contact_radius)

    @staticmethod
    def evaluate_normal_force(contact_radius, penetration):
        """
        """
        return SphereCrackFrontERRPenetrationEnergy._evaluate_normal_force_naive(contact_radius, penetration)

    def gradient(self, radius, penetration):
        if (radius <= 0).any():
            raise NegativeRadiusError
        eerr_j = JKR.nonequilibrium_elastic_energy_release_rate(
            contact_radius=radius,
            penetration=penetration,
            **_jkrkwargs)

        return 2 * np.pi / self.npx * (
                radius * eerr_j
                + self.wm * self.elastic_hessp(radius)
                - self.w(radius, self.angles) * radius)

    def elastic_gradient(self, radius, penetration):
        if (radius <= 0).any():
            raise NegativeRadiusError
        eerr_j = JKR.nonequilibrium_elastic_energy_release_rate(
            contact_radius=radius,
            penetration=penetration,
            **_jkrkwargs)

        return 2 * np.pi / self.npx * (
                radius * eerr_j
                + self.wm * self.elastic_hessp(radius))

    def hessian_product(self, p, radius, penetration):
        eerr_j = JKR.nonequilibrium_elastic_energy_release_rate(
            contact_radius=radius,
            penetration=penetration,
            **_jkrkwargs)
        deerr_da_j = JKR.nonequilibrium_elastic_energy_release_rate(
            contact_radius=radius,
            penetration=penetration,
            **_jkrkwargs, der="1_a")

        return 2 * np.pi / self.npx * (
            (eerr_j + deerr_da_j * radius) * p
            + self.wm * self.elastic_hessp(p)
            - (self.w(radius, self.angles) + radius * self.dw(radius, self.angles)) * p
        )


class SphereCFPenetrationEnergyConstGcPiecewiseLinearField(SphereCrackFrontERRPenetrationEnergyConstGc):
    def __init__(self, piecewise_linear_w, wm=1/np.pi):
        n_pixels = piecewise_linear_w.npx_front
        self.piecewise_linear_w = piecewise_linear_w

        # Note that we still provide the function w and dw, directly,
        # which is useful for postprocessing purposes or using the trust_region_solver.
        super().__init__(npx=n_pixels,
                         w=lambda x, angles: piecewise_linear_w(x, der="0"),
                         dw=lambda x, angles: piecewise_linear_w(x, der="1"), wm=wm)

    def rosso_krauth(self, a, penetration, gtol=1e-4, maxit=10000, dir=1, logger=None):
        """
        This is an adaptation of the Algorithm by Krauth and Rosso PRE 65

        This version of Rosso Krauth uses the fact that the work of adhesion is piecewise linear
        """
        L = len(a)
        a_test = np.zeros(L)
        a_test[0] = 1
        line_stiffness_individual = 2 * np.pi / self.npx * self.wm * self.elastic_hessp(a_test)[0]

        indexes = self.piecewise_linear_w.indexes
        kinks = self.piecewise_linear_w.kinks
        values = self.piecewise_linear_w.values
        grid_spacing = self.piecewise_linear_w.grid_spacing

        # index of the next (higher radius) kink of the piecewise linear work of adhesion
        colloc_point_above = np.searchsorted(kinks, a, side="right")

        pinning_field_slope = (values[indexes, colloc_point_above] - values[indexes, colloc_point_above - 1]) \
            / grid_spacing
        grad = self.elastic_gradient(a, penetration) \
            - values[indexes, colloc_point_above - 1] \
            - pinning_field_slope * (a - kinks[colloc_point_above - 1])
        if (grad * dir > 0).any():
            print("WARNING: Starting Configuration is not purely advancing or receding")

        nit = 0
        while (np.max(abs(grad)) > gtol) and nit < maxit:
            if logger:
                logger.st(["it", "max. residual"], [nit, np.max(abs(grad))])

            # Nullify the force on each pixel
            pinning_field_slope = (values[indexes, colloc_point_above] - values[indexes, colloc_point_above - 1]) \
                / grid_spacing
            grad = self.elastic_gradient(a, penetration) \
                - values[indexes, colloc_point_above - 1] \
                - pinning_field_slope * (a - kinks[colloc_point_above - 1])

            eerr_j = JKR.nonequilibrium_elastic_energy_release_rate(
                contact_radius=a,
                penetration=penetration,
                **_jkrkwargs)
            outer_eastic_stifness = 2 * np.pi / self.npx * eerr_j

            # strictly speaking I should take into account that this is nonlinear
            # But in practice with a fine discretisation the stiffness associated
            # with moving one pixel leads to contact area increments small enough so that this nonlinearity
            # doesn't matter
            stiffness = - pinning_field_slope + outer_eastic_stifness + line_stiffness_individual

            increment = - grad / stiffness

            a_new = a + increment
            mask_negative_stiffness = stiffness <= 0

            if dir == 1:
                # We let the line advance only until the boundary to the next pixel.
                # This is because the step length was based on the pinning curvature
                # which is erroneous as soon as we meet the next pixel
                #
                # Additionally, when the curvature is negative, the increment is negative
                # but the front should actually move forward.
                # In this case as well we advance the front until the edge of the next pixel
                mask_new_pixel = np.logical_or(a_new >= colloc_point_above * grid_spacing, mask_negative_stiffness)
                a_new = np.where(mask_new_pixel, grid_spacing * colloc_point_above, a_new)

                colloc_point_above += mask_new_pixel

                # because of numerical errors it can be that the gradient points in the wrong
                # direction on some pixels, but is very small.
                # We just make sure these points do not move backwards
                a = np.maximum(a_new, a)
            elif dir == -1:
                mask_new_pixel = np.logical_or(a_new <= grid_spacing * (colloc_point_above - 1),
                                                  mask_negative_stiffness)
                a_new = np.where(mask_new_pixel, grid_spacing * (colloc_point_above - 1), a_new)

                colloc_point_above -= mask_new_pixel
                # Why not just -= ?

                a = np.minimum(a_new, a)

            if (colloc_point_above < 1).any():
                raise RadiusTooLowError

            nit += 1

        if nit == maxit:
            success = False
        else:
            success = True

        result = OptimizeResult({
            'success': success,
            'x': a,
            'nit': nit,
            })
        return result
