#
# Copyright 2020 Antoine Sanner
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

"""
call
mprof run memory_profiling_random.py
mprof plot
"""
from memory_profiler import profile
import numpy as np
import matplotlib.pyplot as plt
import os

from muFFT.NetCDF import NCStructuredGrid
from SurfaceTopography import Topography

from CrackFront.Circular import (
    SphereCrackFrontPenetrationIntermediate, pol2cart,
    cart2pol, Interpolator
    )


from Adhesion.ReferenceSolutions import JKR
from CrackFront.Optimization import trustregion_newton_cg

from matplotlib.animation import FuncAnimation

FILENAME = "circular_random_CF.nc"

## FIXED by the nondimensionalisation
maugis_K = 1.
Es = 3/4
maugis_K = 1.
w = 1 / np.pi
R = 1.
mean_Kc = np.sqrt(2 * Es * w)

dk = 0.2# $dw # rms stress intensity factor fluctuation
lcor = .1 #$lcor # correlation length: length of the highcut
seed = 1#$seed

smallest_puloff_radius = (np.pi * w * (1- 3 * dk)**2 * R**2 / 6 * maugis_K)**(1/3)

sx, sy = 4., 4.
nx = ny = 4096

# generated a random noise topography, filter it to force correlation length
# and set mean to 0

k_fluct_topo = Topography(
    np.random.uniform(size=(nx, ny)), physical_sizes=(sx, sy), periodic=True
    ).highcut(cutoff_wavelength=lcor).detrend()

k_topo = Topography((k_fluct_topo.scale(dk / k_fluct_topo.rms_height()
                                        ).heights()
                     + 1) * mean_Kc,
                    k_fluct_topo.physical_sizes, periodic=True)
k_topo_interpolator = Interpolator(k_topo)

class RadiusTooLowError(Exception):
    pass

@profile
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
        initial_radius=None,
        trust_radius=0.05
        ):
    """

    Parameters:
    -----------
    pulloff_radius: radius at which  the pulloff certainly happend and hence
    the iterations stop

    """

    cf = SphereCrackFrontPenetrationIntermediate(npx=n, kc=kc, dkc=dkc)

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
        if np.max(radius) < smallest_puloff_radius:
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
                    maxiter=50000,
                    gtol=1e-6  # he has issues to reach the gtol at small values of a
                    )
            except RadiusTooLowError:
                print("lost contact")
                break
            print(sol.message)
            assert sol.success
            print("nit: {}".format(sol.nit))
            a = sol.x
            cf.dump(nc_CF[j], penetration, sol)
            j = j + 1
    finally:
        nc_CF.close()


def penetrations(dpen, max_pen):
    i = 0 # integer penetration value
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
n = 1024
fn = f"circular_random_CF_n{n}.nc"

simulate_crack_front(
    k_topo_interpolator.kc_polar,
    k_topo_interpolator.dkc_polar,
    n=n,
    penetrations=penetrations(dpen=lcor/100, max_pen=1.),
    filename=fn,
    pulloff_radius= (np.pi * w * (1 - 3 * dk)**2 * R**2 / 6 * maugis_K)**(1/3),
    initial_radius=JKR.contact_radius(penetration=0,
                                    work_of_adhesion=w*(1-3 * dk)**2),
    trust_radius=0.25 * lcor
)