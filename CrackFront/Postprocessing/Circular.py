
import numpy as np
import matplotlib.pyplot as plt
from CrackFront.Circular import cart2pol, pol2cart
from muFFT.NetCDF import NCStructuredGrid

from matplotlib.animation import FuncAnimation

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

    def animate_CF(self, nc_filename, output_filename):
        nc_CF = NCStructuredGrid(nc_filename)

        npx_CF = len(nc_CF.radius[0])

        angle = np.arange(npx_CF) / npx_CF * 2 * np.pi

        ax = self.ax
        l, = ax.plot(*self.pol2pixels(nc_CF.radius[0,...], angle),
                ".-k", ms=1,  label="CF")

        ax.legend()

        def animate(index):
            ax.set_title(r"penetration=" + f"{nc_CF.penetration[index]:.2e}")
            l.set_data(*self.pol2pixels(nc_CF.radius[index,...], angle))

        FuncAnimation(self.fig, animate, frames=len(nc_CF), interval=20).save(output_filename)

        #if with_contact_area:
        #    contacting_points = pressures = nc.contacting_points[index]

        nc_CF.close()
