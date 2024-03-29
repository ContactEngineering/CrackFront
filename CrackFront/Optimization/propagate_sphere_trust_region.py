#
# Copyright 2021 Antoine Sanner
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
import sys

import numpy as np
from NuMPI.IO.NetCDF import NCStructuredGrid

from CrackFront.Circular import RadiusTooLowError
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
        logger=None,
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
        # TODO: because the trust region is defined using a 2-norm, this condition is not sufficient to exclude negative radii.
        # I think the easiest solution is simply to artificially add a region of high work of adhesion near the center
        # otherwise, all that remains left is to use a constrained CG.
        return np.min((trust_radius, 0.9 * np.min(radius)))

    try:
        for penetration in penetrations:
            print(f"penetration: {penetration}")
            try:
                sol = trustregion_newton_cg(
                    x0=a, gradient=lambda radius: cf.gradient(radius, penetration),
                    hessian_product=lambda a, p: cf.hessian_product(p,
                                                                    radius=a,
                                                                    penetration=penetration),
                    trust_radius_from_x=trust_radius_from_x,
                    maxiter=1000000,
                    gtol=gtol,  # he has issues to reach the gtol at small values of a
                    logger=logger
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
