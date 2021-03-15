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


import numpy as np
import matplotlib.pyplot as plt
import os

from NuMPI.IO.NetCDF import NCStructuredGrid

from CrackFront.Circular import  SphereCrackFrontPenetrationIntermediate, pol2cart, cart2pol
from Adhesion.ReferenceSolutions import JKR
from CrackFront.Optimization import trustregion_newton_cg

from matplotlib.animation import FuncAnimation

FILENAME = "circular_cartesian_eggbox_CF.nc"

## FIXED by the nondimensionalisation
maugis_K = 1.
Es = 3/4
maugis_K = 1.
w = 1 / np.pi
R = 1.
mean_Kc = np.sqrt(2 * Es * w)

## free parameters: rays

# strength of the disorder
dK = 0.4
fluctuation_length = 0.1

smallest_puloff_radius = (np.pi * w * (1-dK)**2 * R**2 / 6 * maugis_K)**(1/3)

class RadiusTooLowError(Exception):
    pass

def kc(radius, angle):
    """
    the origin of the system is at the sphere tip
    """
    if np.max(radius) < smallest_puloff_radius:
        raise RadiusTooLowError
    x, y = pol2cart(radius, angle)
    return  (1 + dK * np.cos(2 * np.pi * x / fluctuation_length)
             * np.cos(2 * np.pi * y / fluctuation_length))  * mean_Kc

def dkc(radius, angle):
    x, y = pol2cart(radius, angle)
    dx = - 2 * np.pi * np.sin(2 * np.pi * x / fluctuation_length) * \
         np.cos(2 * np.pi * y / fluctuation_length) \
         * dK * mean_Kc  / fluctuation_length

    dy = - 2 * np.pi * np.cos(2 * np.pi * x / fluctuation_length) * \
             np.sin(2 * np.pi * y / fluctuation_length) \
             * dK * mean_Kc  / fluctuation_length

    return dx * np.cos(angle) + dy * np.sin(angle)


def simulate_crack_front(n=512, filename=FILENAME):

    cf = SphereCrackFrontPenetrationIntermediate(npx=n, kc=kc, dkc=dkc)

    nc_CF = NCStructuredGrid(filename, "w", (n,))

    penetration = 0

    area = 0

    penetrations = np.concatenate((
        np.linspace(0, 1., 200, endpoint=False),
        np.linspace(1., -2., 600)
        ))

    # initial guess
    a = np.ones( n) * smallest_puloff_radius
    #JKR.contact_radius(penetration=penetrations[0])

    j = 0

    try:
        for penetration in penetrations:
            print(f"penetration: {penetration}")
            try:
                sol = trustregion_newton_cg(
                    x0=a, gradient=lambda a: cf.gradient(a, penetration),
                    hessian=lambda a: cf.hessian(a, penetration),
                    #trust_radius=0.25 * np.min((np.min(a), fluctuation_length)),
                    trust_radius_from_x=
                        lambda x: np.min((0.25 * fluctuation_length, 0.9 * np.min(x))),
                    maxiter=3000,
                    gtol=1e-6  # he has issues to reach the gtol at small values of a
                    )
            except RadiusTooLowError:
                print("lost contact")
                a = np.zeros(n)
                # nc_CF[j].cm_sim_index = i
                nc_CF[j].penetration = penetration
                nc_CF[j].radius = a
                nc_CF[j].mean_radius = np.mean(a)
                break
            print(sol.message)
            assert sol.success
            print("nit: {}".format(sol.nit))
            a = sol.x
            # nc_CF[j].cm_sim_index = i
            nc_CF[j].penetration = penetration
            nc_CF[j].radius = a
            nc_CF[j].mean_radius = np.mean(a)

            # infos on convergence
            nc_CF[j].nit = sol.nit
            nc_CF[j].n_hits_boundary = sol.n_hits_boundary

            j = j + 1
    finally:
        nc_CF.close()


class ContactFrame():
    def __init__(self, kc, physical_size=4, npx=500,):
        """
        Parameters:
        -----------
        kc: function giving the

        # TODO: allow for array. Then physical sizes and npx get superflous

        physical_size:
            width of the region to be plotted, centered in 0

        npx: int
            number of pixels for the meshgrid along each direction

        """

        self.fig, self.ax = plt.subplots()
        self.physical_size = physical_size
        self.npx_plot = npx

        self.ax.set_aspect(1)
        s = sx = sy = physical_size
        x, y = (np.mgrid[:npx, :npx] / npx - 1/2) * s
        rho, phi = cart2pol(x, y)
        workcmap = plt.get_cmap("coolwarm")
        self.ax.imshow(kc(rho, phi).T, cmap=workcmap)

        self.ax.invert_yaxis()

        ticks = np.linspace(-sx/2, (sx/2), 5)

        self.ax.set_xticks(ticks / sx * npx + npx * 0.5)
        self.ax.set_xticklabels([f"{v:.2f}" for v in ticks])

        ticks = np.linspace(-sy/2, sy/2, 5)
        self.ax.set_yticks(ticks / sy * npx + npx * 0.5)
        self.ax.set_yticklabels([f"{v:.2f}" for v in ticks])

        self.ax.set_xlabel(r'$y$ ($(\pi w_m R^2 / E^m)^{1/3}$)')
        self.ax.set_ylabel(r'$y$ ($(\pi w_m R^2 / E^m)^{1/3}$)')

    def cart2pixels(self, x, y):
        """
        converts physical coordinates into pixel coordinates
        """
        return (x + self.physical_size / 2) / self.physical_size * self.npx_plot, \
               (y + self.physical_size / 2) / self.physical_size * self.npx_plot

    def pol2pixels(self, radius, angle): 
        return self.cart2pixels(*pol2cart(radius, angle))
        
    


def plot_CF(filename=FILENAME, index = 10):
    nc_CF = NCStructuredGrid(filename)

    npx_CF = len(nc_CF.radius[0])

    angle = np.arange(npx_CF) / npx_CF * 2 * np.pi

    if index < 0:
        index = len(nc_CF) + index

    sx = sy = s = 2.2 * np.max(nc_CF.mean_radius[...])

    npx_plot = 500

    figure = ContactFrame(kc, s, npx=npx_plot)
    ax = figure.ax
    ax.plot(*figure.pol2pixels(nc_CF.radius[index,...], angle),
            ".-k", ms=1,  label="CF")

    penetration = nc_CF.penetration[index]
    ax.set_title(r"penetration=" + f"{penetration:.2e}")

    #if with_contact_area:
    #    contacting_points = pressures = nc.contacting_points[index]

    nc_CF.close()
    ax.legend()

    figure.fig.savefig(f"contact_pen{penetration:.2e}.pdf")

def animate_CF(filename=FILENAME, index = 10):
    nc_CF = NCStructuredGrid(filename)

    npx_CF = len(nc_CF.radius[0])

    angle = np.arange(npx_CF) / npx_CF * 2 * np.pi

    if index < 0:
        index = len(nc_CF) + index

    sx = sy = s = 2.2 * np.max(nc_CF.mean_radius[...])

    npx_plot = 500

    figure = ContactFrame(kc, s, npx=npx_plot)
    ax = figure.ax
    l, = ax.plot(*figure.pol2pixels(nc_CF.radius[0,...], angle),
            ".-k", ms=1,  label="CF")

    ax.legend()
    penetration = nc_CF.penetration[index]

    def animate(index):
        ax.set_title(r"penetration=" + f"{nc_CF.penetration[index]:.2e}")
        l.set_data(*figure.pol2pixels(nc_CF.radius[index,...], angle))

    FuncAnimation(figure.fig, animate, frames=len(nc_CF), interval=20).save("circular_cartesian_eggbox.mp4")

    #if with_contact_area:
    #    contacting_points = pressures = nc.contacting_points[index]

    nc_CF.close()


    figure.fig.savefig(f"contact_pen{penetration:.2e}.pdf")

if __name__ == "__main__":
    ns = [
    128, 256, #512, 1024,
    #2048
    ]
    overwrite = False
    for n in ns:
        fn = f"circular_cartesian_eggbox_CF_n{n}.nc"
        if not os.path.isfile(fn) or overwrite:
            simulate_crack_front(n, fn)
    
    #plot_CF(index=10)
    animate_CF("circular_cartesian_eggbox_CF_n2048.nc")

    import matplotlib.pyplot as plt

    nc_CF = NCStructuredGrid(FILENAME)
    max_contact_radius = np.max(nc_CF.mean_radius[:])
    a = np.linspace(0, max_contact_radius)
    min_w = (1 - dK)**2 * w
    max_w = (1 + dK)**2 * w

    npx = 2000
    s = sx = sy = 4
    x, y = (np.mgrid[:npx, :npx] / npx - 1/2) * s
    mean_w = np.mean(kc(*cart2pol(x, y)) ** 2 / (2*Es))
    min_w = np.min(kc(*cart2pol(x, y)) ** 2 / (2*Es) )
    max_w = np.max(kc(*cart2pol(x, y)) **2/ (2*Es))

    fig, ax = plt.subplots()
    for n in ns:
        nc_CF = NCStructuredGrid(f"circular_cartesian_eggbox_CF_n{n}.nc")
        ax.plot(nc_CF.penetration, nc_CF.mean_radius, label=f"n={n}")
        nc_CF.close()

    ax.plot(JKR.penetration(contact_radius=a, work_of_adhesion=min_w),
            a,
            "--b", label=f"JKR, min w = {min_w:.2f}")
    ax.plot(JKR.penetration(contact_radius=a,),
            a,
            "--k", label=f"JKR, mean w, {w:.2f}")
    ax.plot(JKR.penetration(contact_radius=a, work_of_adhesion=max_w),
            a,
            "--r", label=f"JKR, max w={max_w:.2f}")

    ax.set_xlabel(r'Displacement $(\pi^2 w_m^2 R / K^2)^{1/3}$')
    ax.set_ylabel(r'$mean contact radius$' +"\n"+
                  r' ($(\pi w_m R^2 / K)^{1/3}$)')

    fig, ax = plt.subplots()
    for n in ns:
        nc_CF = NCStructuredGrid(f"circular_cartesian_eggbox_CF_n{n}.nc")
        ax.plot(nc_CF.penetration, JKR.force(contact_radius=nc_CF.mean_radius[:],
                                             penetration=nc_CF.penetration[:]),
                label=f"n={n}")
        nc_CF.close()

    ax.plot(JKR.penetration(contact_radius=a, work_of_adhesion=min_w),
            JKR.force(contact_radius=a, work_of_adhesion=min_w),
            "--b", label=f"JKR, min w = {min_w:.2f}")
    ax.plot(JKR.penetration(contact_radius=a,),
            JKR.force(contact_radius=a,),
            "--k", label=f"JKR, mean w, {w:.2f}")
    ax.plot(JKR.penetration(contact_radius=a, work_of_adhesion=max_w),
            JKR.force(contact_radius=a, work_of_adhesion=max_w),
            "--r", label=f"JKR, max w={max_w:.2f}")
    ax.legend()


    ax.set_ylabel('Force \n'+r'($\pi w_m R$)')
    ax.set_xlabel(r'Displacement $(\pi^2 w_m^2 R / K^2)^{1/3}$')

    plt.show()

    nc_CF.close()