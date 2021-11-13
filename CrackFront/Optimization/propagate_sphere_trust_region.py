import sys

import numpy as np
from Adhesion.ReferenceSolutions import JKR
from NuMPI.IO.NetCDF import NCStructuredGrid

from CrackFront.Circular import RadiusTooLowError
from CrackFront.CircularEnergyReleaseRate import SphereCrackFrontERRPenetrationEnergyConstGc
from CrackFront.Optimization import trustregion_newton_cg


def penetrations_generator(dpen, max_pen):
    i = 0  # integer penetration value
    pen = dpen * i
    yield pen
    while pen < max_pen:
        i += 1
        pen = dpen * i
        yield pen
    while True:
        i -= 1
        pen = dpen * i
        yield pen

def simulate_crack_front(
        cf,
        penetrations=np.concatenate((
        np.linspace(0, 1., 200, endpoint=False),
        np.linspace(1., -2., 600)
        )),
        filename="CF.nc",
        pulloff_radius=0.01,
        initial_radius=None,
        trust_radius=0.05,
        dump_fields=True,
        gtol=1e-06,
        ):
    """

    Parameters:
    -----------
    pulloff_radius: radius at which  the pulloff certainly happend and hence
    the iterations stop

    """
    n = cf.npx

    nc_CF = NCStructuredGrid(filename, "w", (n,))

    penetration = 0

    # initial guess
    if initial_radius is None:
        a = np.ones(n) * pulloff_radius
    elif not hasattr(initial_radius, "len"):
        a = np.ones(n) * initial_radius
    else:
        a = initial_radius

    j = 0

    def trust_radius_from_x(radius):
        if np.max(radius) < pulloff_radius:
                raise RadiusTooLowError
        return np.min((trust_radius, 0.9 * np.min(radius)))

    try:
        for penetration in penetrations:
            print(f"penetration: {penetration}")
            try:
                sol = trustregion_newton_cg(
                    x0=a, gradient=lambda radius: cf.gradient(radius, penetration),
                    #hessian=lambda a: cf.hessian(a, penetration),
                    hessian_product=lambda a, p: cf.hessian_product(p,
                                                                    radius=a,
                                                                    penetration=penetration),
                    trust_radius_from_x=trust_radius_from_x,
                    maxiter=1000000,
                    gtol=gtol  # he has issues to reach the gtol at small values of a
                    )
            except RadiusTooLowError:
                print("lost contact")
                break
            print(sol.message)
            assert sol.success
            print("nit, njev: {}, {}".format(sol.nit, sol.njev))
            a = sol.x
            cf.dump(nc_CF[j], penetration, a, dump_fields)
            j = j + 1

            nc_CF.sync()
            sys.stdout.flush()


    finally:
        nc_CF.close()
