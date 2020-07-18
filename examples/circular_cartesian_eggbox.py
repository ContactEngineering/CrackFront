

import numpy as np
import matplotlib.pyplot as plt

from muFFT.NetCDF import NCStructuredGrid

from CrackFront.Circular import  SphereCrackFrontPenetration, pol2cart, cart2pol
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
    the origin of the system is at the top of the siewave
    """
    if np.max(radius) < smallest_puloff_radius:
        raise RadiusTooLowError
    x, y = pol2cart(radius, angle)
    return  (1 + 2 * dK * np.cos(2 * np.pi * x / fluctuation_length) * np.cos(2 * np.pi * y / fluctuation_length) )  * mean_Kc

def dkc(radius, angle):
    x, y = pol2cart(radius, angle)
    dx = - 2 * np.pi * np.sin(2 * np.pi * x / fluctuation_length) * \
         np.cos(2 * np.pi * y / fluctuation_length) \
         * 2 * dK * mean_Kc  / fluctuation_length

    dy = - 2 * np.pi * np.cos(2 * np.pi * x / fluctuation_length) * \
             np.sin(2 * np.pi * y / fluctuation_length) \
             * 2 * dK * mean_Kc  / fluctuation_length

    return dx * np.cos(angle) + dy * np.sin(angle)


def simulate_crack_front(n = 512):

    cf = SphereCrackFrontPenetration(npx=n, kc=kc, dkc=dkc)

    nc_CF = NCStructuredGrid(FILENAME, "w", (n,))

    penetration = 0

    area = 0

    penetrations= np.concatenate((
        np.linspace(0, 1., 100, endpoint=False),
        np.linspace(1., -2., 300)
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
    #simulate_crack_front()
    
    #plot_CF(index=10)
    #animate_CF()

    import matplotlib.pyplot as plt
    nc_CF = NCStructuredGrid(FILENAME)
    fig, ax = plt.subplots()
    ax.plot(nc_CF.penetration, nc_CF.mean_radius)


    fig, ax = plt.subplots()
    ax.plot(nc_CF.penetration, JKR.force(contact_radius=nc_CF.mean_radius[:],
                                         penetration=nc_CF.penetration[:]))

    max_contact_radius = np.max(nc_CF.mean_radius[:])
    min_w = (1 - dK)**2 * w
    max_w = (1 + dK)**2 * w

    a = np.linspace(0, max_contact_radius)
    ax.plot(JKR.penetration(contact_radius=a, work_of_adhesion=min_w),
            JKR.force(contact_radius=a, work_of_adhesion=min_w),
            "--b", label="JKR, min w")
    ax.plot(JKR.penetration(contact_radius=a,),
            JKR.force(contact_radius=a,),
            "--k", label="JKR, mean w")
    ax.plot(JKR.penetration(contact_radius=a, work_of_adhesion=max_w),
            JKR.force(contact_radius=a, work_of_adhesion=max_w),
            "--r", label="JKR, max w")

    # TODO: Axes Labels !!!
    plt.show()

    nc_CF.close()