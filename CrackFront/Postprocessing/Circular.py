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
from CrackFront.Circular import cart2pol, pol2cart
from muFFT.NetCDF import NCStructuredGrid
from SurfaceTopography import Topography
from matplotlib.animation import FuncAnimation
from CrackFront.Tools.Hysteresis import direction_change_index


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
        self.ax.set_aspect(1)
        workcmap = plt.get_cmap("coolwarm")
        if isinstance(kc, Topography):
            self.physical_size = kc.physical_sizes[0]
            self.npx_plot = npx = kc.nb_grid_pts[0]
            s = sx = sy = self.physical_size
            self.ax.imshow(kc.heights().T, cmap=workcmap)
        else:
            self.physical_size = physical_size
            self.npx_plot = npx
            s = sx = sy = physical_size
            x, y = (np.mgrid[:npx, :npx] / npx - 1/2) * s
            rho, phi = cart2pol(x, y)

            self.ax.imshow(kc(rho, phi).T, cmap=workcmap)

        self.ax.invert_yaxis()

        ticks = np.linspace(-sx/2, (sx/2), 5)

        self.ax.set_xticks(ticks / sx * npx + npx * 0.5)
        self.ax.set_xticklabels([f"{v:.2f}" for v in ticks])

        ticks = np.linspace(-sy/2, sy/2, 5)
        self.ax.set_yticks(ticks / sy * npx + npx * 0.5)
        self.ax.set_yticklabels([f"{v:.2f}" for v in ticks])

        self.ax.set_xlabel(r'$y$ ($(\pi w_m R^2 / K)^{1/3}$)')
        self.ax.set_ylabel(r'$y$ ($(\pi w_m R^2 / K)^{1/3}$)')

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


def plot_contact_area_increment(dataset_directories, labels=None, save=False):
    w = 1 / np.pi

    fig_forward, ax_forward = plt.subplots(figsize=(12, 6))
    fig_backward, ax_backward = plt.subplots(figsize=(12, 6))

    max_contact_radius = 0
    for i, directory in enumerate(dataset_directories):
        nc_CF = NCStructuredGrid(directory+"/data/data.nc")
        i_dirchange = direction_change_index(nc_CF.penetration)

        ax_forward.plot(nc_CF.penetration[1:i_dirchange],
                        nc_CF.contact_area[1:i_dirchange]
                        - nc_CF.contact_area[:i_dirchange-1],
                        ".-", lw=0.5,
                        label=directory if labels is None else labels[i])

        ax_backward.plot(nc_CF.penetration[i_dirchange+1:],
                         - (nc_CF.contact_area[i_dirchange+1:]
                         - nc_CF.contact_area[i_dirchange:-1]), ".-",
                         ".-", lw=0.5,
                         label=directory if labels is None else labels[i])
        nc_CF.close()

    ax_forward.set_title("forward")
    ax_forward.set_xlabel(r'Displacement $(\pi^2 w_m^2 R / K^2)^{1/3}$')
    ax_forward.set_ylabel(r'contact area increment' + "\n" +
                          r' ($(\pi w_m R^2 / K)^{2/3}$)')
    ax_forward.set_yscale("log")
    ax_forward.legend(bbox_to_anchor=(0, 1.05), loc="lower left")

    ax_backward.set_title("backward")
    ax_backward.set_xlabel(r'Displacement $(\pi^2 w_m^2 R / K^2)^{1/3}$')
    ax_backward.set_ylabel(r'- contact area increment' + "\n" +
                           r' ($(\pi w_m R^2 / K)^{2/3}$)')
    ax_backward.set_yscale("log")
    ax_backward.legend(bbox_to_anchor=(0, 1.05), loc="lower left")

    plt.show()
    if save:
        fig_backward.save_fig("mean_contact_radius_increment_backward.pdf")
        fig_forward.save_fig("mean_contact_radius_increment_forward.pdf")
    ax_backward.grid()
    ax_forward.grid()
    return fig_forward, fig_backward

