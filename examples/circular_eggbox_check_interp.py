
"""

Goal is to determine how many interpolation points are needed

"""


import numpy as np
import matplotlib.pyplot as plt
import os

from muFFT.NetCDF import NCStructuredGrid

from CrackFront.Circular import  SphereCrackFrontPenetration, pol2cart, cart2pol
from Adhesion.ReferenceSolutions import JKR
from CrackFront.Optimization import trustregion_newton_cg


from matplotlib.animation import FuncAnimation
from CrackFront.Circular import Interpolator
from SurfaceTopography import Topography

maugis_K = 1.
Es = 3/4
w = 1 / np.pi
R = 1.
mean_Kc = np.sqrt(2 * Es * w)

class RadiusTooLowError(Exception):
    pass


class Eggbox():
    def __init__(self, period, dk, mean_kc=np.sqrt(2 * 3/4 / np.pi)):
        self.period = period
        self.mean_kc = mean_kc
        self.dk = dk

    def kc(self, x, y):
        return (1 + self.dk * np.cos(2 * np.pi * x / self.period)
                 * np.cos(2 * np.pi * y / self.period))  * self.mean_kc

    def kc_polar(self, radius, angle):
        """
        the origin of the system is at the sphere tip
        """

        x, y = pol2cart(radius, angle)
        return self.kc(x, y)

    def dkc_polar(self, radius, angle):
        x, y = pol2cart(radius, angle)
        dx = - 2 * np.pi * np.sin(2 * np.pi * x / self.period) * \
             np.cos(2 * np.pi * y / self.period) \
             * self.dk * self.mean_kc  / self.period

        dy = - 2 * np.pi * np.cos(2 * np.pi * x / self.period) * \
                 np.sin(2 * np.pi * y / self.period) \
                 * self.dk * self.mean_kc  / self.period

        return dx * np.cos(angle) + dy * np.sin(angle)


def simulate_crack_front(
        kc,
        dkc,
        n=512,
        penetrations=np.concatenate((
        np.linspace(0, 1., 200, endpoint=False),
        np.linspace(1., -2., 600)
        )),
        filename="CF.nc",
        pulloff_radius=0.01,
        trust_radius=0.05
        ):
    """

    Parameters:
    -----------
    pulloff_radius: radius at which  the pulloff certainly happend and hence
    the iterations stop

    """

    cf = SphereCrackFrontPenetration(npx=n, kc=kc, dkc=dkc)

    nc_CF = NCStructuredGrid(filename, "w", (n,))

    penetration = 0

    # initial guess
    a = np.ones( n) * pulloff_radius
    #JKR.contact_radius(penetration=penetrations[0])

    j = 0

    try:
        for penetration in penetrations:
            print(f"penetration: {penetration}")
            try:
                def gradient(radius):
                    # TODO: this could be integrated directly in the crack
                    #  front class
                    if np.max(radius) < pulloff_radius:
                        raise RadiusTooLowError
                    return cf.gradient(radius, penetration)
                sol = trustregion_newton_cg(
                    x0=a, gradient=gradient,
                    hessian=lambda a: cf.hessian(a, penetration),
                    trust_radius_from_x=
                        lambda x: np.min((trust_radius, 0.9 * np.min(x))),
                    maxiter=10000,
                    gtol=1e-6  # he has issues to reach the gtol at small values of a
                    )
            except RadiusTooLowError:
                print("lost contact")
                a = np.zeros(n)
                cf.dump(nc_CF[j], penetration, sol)
                break
            print(sol.message)
            assert sol.success
            print("nit: {}".format(sol.nit))
            a = sol.x
            cf.dump(nc_CF[j], penetration, sol)
            j = j + 1
    finally:
        nc_CF.close()



if __name__ == "__main__":

    period = 0.2
    dk = 0.2
    npx = 512

    overwrite = False

    eggbox = Eggbox(
        period=period,
        dk=dk
        )

    penetrations = np.concatenate((
        np.linspace(0, 1., 200, endpoint=False),
        np.linspace(1., -2., 600)
        ))

    pulloff_radius = (np.pi * w * (1 - dk)**2 * R**2 / 6 * maugis_K)**(1/3)

    fn = "eggbox_check_interp_ref.nc"
    if not os.path.isfile(fn) or overwrite:
        simulate_crack_front(
            eggbox.kc_polar,
            eggbox.dkc_polar,
            n=npx,
            penetrations=penetrations,
            filename=fn,
            pulloff_radius=pulloff_radius,
            trust_radius=10 * period
            )
    sx, sy = 4., 4.
    for npx_grid in [64, 128, 256,]:
        fn = f"eggbox_check_interp_n{npx_grid}.nc"
        x, y = np.mgrid[:npx_grid, :npx_grid] / npx_grid * sx

        if not os.path.isfile(fn) or overwrite:
            interp = Interpolator(Topography(eggbox.kc(x - sx / 2, y - sy /2),
                physical_sizes=(sx, sy), periodic=True))
            simulate_crack_front(
                interp.kc,
                interp.dkc,
                n=npx,
                penetrations=penetrations,
                filename=fn,
                pulloff_radius=pulloff_radius,
                trust_radius=10 * period
            )

    import matplotlib.pyplot as plt

    nc_CF = NCStructuredGrid("eggbox_check_interp_ref.nc")
    max_contact_radius = np.max(nc_CF.mean_radius[:])
    a = np.linspace(0, max_contact_radius)

    fns = [
        "eggbox_check_interp_ref.nc",
        "eggbox_check_interp_n64.nc",
        "eggbox_check_interp_n128.nc",
        "eggbox_check_interp_n256.nc",
        #"eggbox_check_interp_n512.nc"
        ]

    fig, ax = plt.subplots()
    for fn in fns:
        nc_CF = NCStructuredGrid(fn)
        ax.plot(nc_CF.penetration, nc_CF.mean_radius, label=fn)
        nc_CF.close()

    ax.plot(JKR.penetration(contact_radius=a,),
            a,
            "--k", label=f"JKR, median w, {w:.2f}")

    ax.set_xlabel(r'Displacement $(\pi^2 w_m^2 R / K^2)^{1/3}$')
    ax.set_ylabel(r'$mean contact radius$' + "\n" +
                  r' ($(\pi w_m R^2 / K)^{1/3}$)')

    fig, ax = plt.subplots()
    for fn in fns:
        nc_CF = NCStructuredGrid(fn)
        ax.plot(nc_CF.penetration, JKR.force(contact_radius=nc_CF.mean_radius[:],
                                             penetration=nc_CF.penetration[:]),
                label=fn)
        nc_CF.close()

    ax.plot(JKR.penetration(contact_radius=a,),
            JKR.force(contact_radius=a,),
            "--k", label=f"JKR, mean w, {w:.2f}")

    ax.legend()


    ax.set_ylabel('Force \n'+r'($\pi w_m R$)')
    ax.set_xlabel(r'Displacement $(\pi^2 w_m^2 R / K^2)^{1/3}$')

    plt.show()

    # FAZIT: 3 points period is not enough but 6 points period is