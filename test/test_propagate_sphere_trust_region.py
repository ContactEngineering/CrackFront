import numpy as np
from Adhesion.ReferenceSolutions import JKR
from NuMPI.IO.NetCDF import NCStructuredGrid

from CrackFront.CircularEnergyReleaseRate import SphereCrackFrontERRPenetrationEnergyConstGc
from CrackFront.Optimization.propagate_sphere_trust_region import penetrations_generator, simulate_crack_front


def test_propagate_sphere_trust_region_vs_JKR():
    # nondimensional units following Maugis Book:
    Es = 3 / 4
    w = 1 / np.pi
    R = 1.
    maugis_K = 1.

    npx = 64

    cf = SphereCrackFrontERRPenetrationEnergyConstGc(npx,
                                                     w=lambda radius, angle: np.ones_like(radius) * w,
                                                     dw=lambda radius, angle: np.zeros_like(radius) * w,)

    pulloff_radius = (np.pi * w * R**2 / 6 * maugis_K)**(1/3)

    simulate_crack_front(
        cf,
        penetrations=penetrations_generator(0.1, 1.),
        filename="trust.nc",
        pulloff_radius=pulloff_radius * 0.5,
        initial_radius=JKR.contact_radius(penetration=0, work_of_adhesion=w) * 0.5,
        trust_radius=0.5,
        dump_fields=False,
        gtol=1e-08
    )

    nc = NCStructuredGrid("trust.nc")
    np.testing.assert_allclose(nc.force, [JKR.force(penetration=p) for p in nc.penetration[:]], rtol=1e-06)
    np.testing.assert_allclose(nc.mean_radius, [JKR.contact_radius(penetration=p) for p in nc.penetration[:]], rtol=1e-06)
