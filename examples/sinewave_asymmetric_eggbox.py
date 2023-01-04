#
# Copyright 2020-2021 Antoine Sanner
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
from CrackFront.Optimization import trustregion_newton_cg
import matplotlib.pyplot as plt

# %% [markdown] contactmechanics simulation

from ContactMechanics import PeriodicFFTElasticHalfSpace

from Adhesion.Interactions import PowerLaw
from ContactMechanics.Tools import Logger
from Adhesion.System import BoundedSmoothContactSystem
from NuMPI.IO.NetCDF import NCStructuredGrid

from CrackFront.Straight import SinewaveCrackFrontLoad

from matplotlib.animation import FuncAnimation

from SurfaceTopography import Topography

def make_topography_from_function(fun, physical_sizes,
                 nb_grid_pts=None,
                 subdomain_locations=None,
                 nb_subdomain_grid_pts=None,
                 origin=(0,0),
                 **kwargs
                 ):
    """
    See also Topography

    Parameters
    ----------
    fun: callable(X, Y)
    origin: tuple or "center"
        (cx, cy)
        h(X,Y) =  fun(X-cx, Y-cy)
        default (0,0)
    physical_sizes
    subdomain_locations
    nb_subdomain_grid_pts
    nb_grid_pts

    Other Parameters
    ----------------

    other kwargs for PyCo.Topography

    Returns
    -------

    SurfaceTopography.topography

    """
    if isinstance(origin,str) and origin=="center":
        origin= tuple([s / 2 for s in physical_sizes])

    if nb_subdomain_grid_pts is None:
        # serial code
        nb_subdomain_grid_pts = nb_grid_pts
        nb_grid_pts=None
    topography = Topography(heights=np.zeros(nb_subdomain_grid_pts),
                     physical_sizes=physical_sizes,
                     subdomain_locations=subdomain_locations,
                     nb_grid_pts=nb_grid_pts,
                     **kwargs,
                     )

    cx, cy = origin
    X, Y = topography.positions()

    topography._heights = fun(X-cx, Y-cy)

    return topography

dK = 0.3
mean_Kc = 0.3 # in units of h pwfc. K / hp wfc = alpha
Es = 1 / np.pi
h=1. # amplitude (half peak to valley) of the sinewave
sinewave_lambda = 1.
sx = 1.
max_rel_area = maxarea = 0.55
fluctuation_length = sx / 8
length_parameter = 0.05
phase_shift=np.pi/4

# discretisation
nx = 512
ny = 128
dx= sx / nx
# discretisation
sy = dx * ny


def kc(x, y):
    """
    the origin of the system is at the top of the siewave
    """
    return  (1 + 2 * dK * np.sin(2 * np.pi * x / fluctuation_length + phase_shift) * np.cos(2 * np.pi * y / sy) )  * mean_Kc

def dkc(x, y):
    return 2 * np.pi * np.cos(2 * np.pi * x / fluctuation_length + phase_shift) * np.cos(2 * np.pi * y / sy) * 2 * dK * mean_Kc  / fluctuation_length

print("setup halfspace")
halfspace = PeriodicFFTElasticHalfSpace((nx, ny), Es, (sx, sy),)
print("create topography")
topography = make_topography_from_function(
    lambda x,y: (-1 - np.cos(2 * np.pi * x / sx)),
    (sx,sy),
    subdomain_locations=halfspace.topography_subdomain_locations,
    nb_subdomain_grid_pts=halfspace.topography_nb_subdomain_grid_pts,
    nb_grid_pts=(nx, ny),
    periodic=True,
)
cx, cy = [s / 2 -  s /n /2 for s, n in zip((sx, sy), (nx, ny))]
x, y = topography.positions()

def simulate_CM():
    interaction = PowerLaw( kc(x -0.5, y)**2 / (Es * 2 ),
    #                         v that way the max stress is still w / rho
                              3 * length_parameter * kc(x - 0.5, y) / mean_Kc,
                              3)

    #########################################
    system = BoundedSmoothContactSystem(halfspace, interaction, topography)

    ncfile = NCStructuredGrid("19_CM.nc", mode="w" ,
                              nb_domain_grid_pts=system.surface.nb_grid_pts,
                              )


    displacement = 0.
    delta_d = 0.01
    disp0=None
    mode = "forward"
    try:
        gtol=1e-4
        i=0
        while True:

            #print("##############################################################")
            print("displacement = {}".format(displacement))
            #print("##############################################################")
            # change disp0 to preserve gap
            #if disp0 is not None:
            #    disp0 += displacement - displacement_prev
            displacement_prev = displacement
            while True:
                sol = system.minimize_proxy(
                    disp0=disp0,
                    lbounds="auto",
                    options=dict(gtol=gtol * abs(interaction.max_tensile) * system.surface.area_per_pt ,
                                 ftol=0, maxcor=3, maxiter=80000),
                    logger=Logger("laststep.log"),
                    offset=displacement,
                    callback=None
                )
                if not sol.success:
                    # ask for confirmation
                    # it can simply be slow convergence
                    print(sol.message)
                    input("continue ? ")
                    u = disp0 = sol.x
                else:
                    break

            u = disp0 = sol.x
            mean_deformation = np.sum(u) / np.prod(halfspace.nb_domain_grid_pts)

            force = - halfspace.evaluate_force(u)
            #
            #
            ncfile[i].contacting_points = contacting_points = np.where(system.gap == 0., 1, 0)
            ncfile[i].pressures = force[halfspace.local_topography_subdomain_slices] / topography.area_per_pt


            contact_area= np.sum(contacting_points) * halfspace.area_per_pt
            ncfile[i].fractional_contact_area =  contact_area / (sx * sy)
            normal_force = force.sum()
            ncfile[i].mean_pressure = normal_force / np.prod(topography.physical_sizes)
            ncfile[i].displacement = displacement
            ncfile[i].mean_deformation = mean_deformation
            ncfile[i].elastic_energy = elastic_energy = system.substrate.energy
            ncfile[i].interaction_energy = interaction_energy = system.interaction.energy
            ncfile[i].energy = energy = system.energy

            rel_area = contact_area / np.prod(topography.physical_sizes)

            #main_logger_headers = ["step", "nit", "nfev","walltime","displacement", "mean deformation", "force","frac. area", "energy"]
            #main_logger.st(main_logger_headers,
            #        [i, sol.nit, sol.nfev,  elapsed_time, displacement, mean_deformation, normal_force, rel_area, energy]
            #        )

            i+=1

            if rel_area >= max_rel_area:
                mode = "backward"
                print("max contact area reached")
            if mode == "forward":
                displacement += delta_d
            elif mode == "backward":
                displacement -= delta_d
                if contact_area == 0:
                    print("left contact, stop here")
                    break
    finally:
        ncfile.close()


# %% [markdown]
#

# %%
def simulate_CF(pausetime=0.00001):

    n = 64
    cf = SinewaveCrackFrontLoad(n=n, sy=sy, kc=kc, dkc=dkc)

    #TODO: implement hessian product
    P = 0.

    import scipy.optimize
    #print(scipy.optimize.check_grad(lambda a: gradient(a, P),
    #                                lambda a: hessian(a, P), x0=a))

    nc_CM = NCStructuredGrid("19_CM.nc")
    nc_CF = NCStructuredGrid("19_CF.nc", "w", (n, ) )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    try:
        j = 0
        a = np.ones(2 * n) * nc_CM[0].fractional_contact_area / 2 # take care not to take a too small initial guess
        for i in np.concatenate(make_monotonic_load_indexes(nc_CM.displacement, nc_CM.mean_pressure)):
            print("CM frac contact area: {}".format(nc_CM[i].fractional_contact_area))
            a_left = np.max( ( 1/2 - x ) * nc_CM[i].contacting_points, axis=0)
            a_right = np.max( ( x - 1/2 ) * nc_CM[i].contacting_points, axis=0)
            ax.plot(nc_CM[i].mean_pressure, a_left[0], "<", c="b")
            ax.plot(nc_CM[i].mean_pressure, a_right[0], ">", c="r" )
            plt.pause(pausetime)

            P = nc_CM[i].mean_pressure
            sol = trustregion_newton_cg(x0=a, gradient=lambda a : cf.gradient(a, P),
                                    hessian=lambda a : cf.hessian(a, P),
                                    trust_radius=0.25 * np.min((np.min(a), fluctuation_length)),
                                    maxiter=3000,
                                    gtol=1e-6 # he has issues to reach the gtol at small values of a
                                    )

            assert sol.success
            print("nit: {}".format(sol.nit))
            a = sol.x
            n = int(len(a) / 2)
            a_left = a[:n]
            a_right = a[n:]
            nc_CF[j].cm_sim_index = i
            nc_CF[j].a_left = a_left
            nc_CF[j].a_right = a_right
            nc_CF[j].mean_a_left = np.mean(a_left)
            nc_CF[j].mean_a_right = np.mean(a_right)
            nc_CF[j].mean_pressure = P
            j = j+1
            ax.plot(nc_CM[i].mean_pressure, a_left[0], "+", c="b" )
            ax.plot(nc_CM[i].mean_pressure, a_right[0], "+", c="r" )
            plt.pause(pausetime)

        print("DONE")
    finally:
        nc_CF.close()
        nc_CM.close()

def direction_change_index(displacements):
    """
    returns the index of the maximal displacements
    """
    for i in range(0,len(displacements)):
        if displacements[i] > displacements[i+1]:
            return i



def make_monotonic_load_indexes(displacements, load):
    r"""
    displacements: assumed to be first monotically increasing and then monotically decreasing

    returns a list of indexes where the load is increasing in the increasing sequence
    of displacements, and the load is decreasing in the decreasing sequence of displacements
    """
    i_dirchange = direction_change_index(displacements)
    increasing_load_indexes = [0]
    for i in range(1, i_dirchange+1):
        if load[increasing_load_indexes[-1]] <= load[i]:
            increasing_load_indexes.append(i)

    decreasing_load_indexes = [i_dirchange-1]
    for i in range(i_dirchange, len(displacements)):
        if load[decreasing_load_indexes[-1]] >= load[i]:
            decreasing_load_indexes.append(i)

    return increasing_load_indexes, decreasing_load_indexes

def demo_make_monotonic_load_indexes():
    nc_CM = NCStructuredGrid("19_CM.nc")
    fig, ax = plt.subplots()

    ax.plot(nc_CM.displacement, nc_CM.mean_pressure)
    increasing_load_indexes, decreasing_load_indexes = make_monotonic_load_indexes(nc_CM.displacement, nc_CM.mean_pressure)
    ax.plot(nc_CM.displacement[increasing_load_indexes], nc_CM.mean_pressure[increasing_load_indexes], "o-")
    ax.plot(nc_CM.displacement[decreasing_load_indexes], nc_CM.mean_pressure[decreasing_load_indexes], "+-")



def make_animation_CM():
    from PythonBib.Plotting.utilitaires_plot import MidpointNormalize
    max_stress = mean_Kc**2 / (2 * Es) / length_parameter
    pnorm = MidpointNormalize(vmin = - max_stress , vmax = max_stress)
    workcmap = plt.get_cmap("coolwarm")
    topographycmap = plt.get_cmap("coolwarm")
    from matplotlib.colors import LinearSegmentedColormap
    # plot contact

    contactcmap = LinearSegmentedColormap.from_list('contactcmap', ((1,1,1,0.),(1,1,1,0.3)), N=256)
    #plot presssures
    pressurecmap = LinearSegmentedColormap.from_list('testCmap', (
        (0.15294117647058825, 0.39215686274509803, 0.09803921568627451,1.),
        (1,1,1,0.6),
        (0.5568627450980392, 0.00392156862745098, 0.3215686274509804,1)), N=256)

    fig = plt.figure(figsize = (9, 3))
    nc = NCStructuredGrid("19_CM.nc")
    x, y = topography.positions()
    zoom = False
    def animate(index):
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        pressures = nc.pressures[index, ...]

        workcmap = plt.get_cmap("coolwarm")
        ax.imshow(kc(x-0.5, y).T,cmap=workcmap)

        plt.colorbar(ax.imshow(pressures.T, norm=pnorm, cmap=pressurecmap))

        ax.set_title(r"$P^*$=" + f"{nc.mean_pressure[index]:.2e}")

        #if with_contact_area:
        #    contacting_points = pressures = nc.contacting_points[index]

        ax.invert_yaxis()

        ticks = np.linspace(-sx/2, (sx/2), 5)

        ax.set_xticks(ticks / sx * nx + nx * 0.5)
        ax.set_xticklabels([f"{v:.2f}" for v in ticks])

        ticks = np.linspace(-sy/2, sy/2, 5)
        ax.set_yticks(ticks / sy * ny + ny * 0.5)
        ax.set_yticklabels([f"{v:.2f}" for v in ticks])
    FuncAnimation(fig, animate, frames=len(nc), interval=50).save("19_CM.mp4", dpi=300)


def make_animation_both():
    from PythonBib.Plotting.utilitaires_plot import MidpointNormalize
    max_stress = mean_Kc**2 / (2 * Es) / length_parameter
    pnorm = MidpointNormalize(vmin = - max_stress , vmax = max_stress)
    workcmap = plt.get_cmap("coolwarm")
    topographycmap = plt.get_cmap("coolwarm")
    from matplotlib.colors import LinearSegmentedColormap
    # plot contact

    contactcmap = LinearSegmentedColormap.from_list('contactcmap', ((1,1,1,0.),(1,1,1,0.3)), N=256)
    #plot presssures
    pressurecmap = LinearSegmentedColormap.from_list('testCmap', (
        (0.15294117647058825, 0.39215686274509803, 0.09803921568627451,1.),
        (1,1,1,0.6),
        (0.5568627450980392, 0.00392156862745098, 0.3215686274509804,1)), N=256)

    fig = plt.figure(figsize = (9, 3))
    nc_CM = NCStructuredGrid("19_CM.nc")
    nc_CF = NCStructuredGrid("19_CF.nc")

    x, y = topography.positions()
    zoom = False
    def animate(j):
        index = nc_CF[j].cm_sim_index
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        pressures = nc_CM.pressures[index, ...]

        workcmap = plt.get_cmap("coolwarm")
        ax.imshow(kc(x-0.5, y).T,cmap=workcmap)

        plt.colorbar(ax.imshow(pressures.T, norm=pnorm, cmap=pressurecmap))

        ax.set_title(r"$P^*$=" + f"{nc_CM.mean_pressure[index]:.2e}")

        #if with_contact_area:
        #    contacting_points = pressures = nc.contacting_points[index]

        ax.invert_yaxis()

        ticks = np.linspace(-sx/2, (sx/2), 5)

        ax.set_xticks(ticks / sx * nx + nx * 0.5)
        ax.set_xticklabels([f"{v:.2f}" for v in ticks])

        ticks = np.linspace(-sy/2, sy/2, 5)
        ax.set_yticks(ticks / sy * ny + ny * 0.5)
        ax.set_yticklabels([f"{v:.2f}" for v in ticks])

        y_cf = np.arange(len(nc_CF[j].a_left)) * sy / len(nc_CF[j].a_left)
        ax.plot((0.5 - nc_CF[j].a_left) / sx * nx, y_cf  / sy * ny, "-k")
        ax.plot((0.5 + nc_CF[j].a_right) / sx * nx, y_cf / sy * ny, "-k")

    FuncAnimation(fig, animate, frames=len(nc_CF), interval=50).save("19_both.mp4")

        #plt.pause(0.1)
        #fig.savefig("")

#simulate_CM()
#simulate_CF()
#show_animation()
#make_animation_CM()
make_animation_both()

nc_CM = NCStructuredGrid("19_CM.nc")
nc_CF = NCStructuredGrid("19_CF.nc")

a_left = np.max( (1/2 - x) * nc_CM.contacting_points, axis=1)
a_right = np.max( ( x - 1/2 ) * nc_CM.contacting_points, axis=1)

fig, ax = plt.subplots()
ax.plot(nc_CM.displacement, np.max(a_left, axis=1), label="CM, a left")
ax.plot(nc_CM.displacement, np.max(a_right, axis=1), label="CM, a right")

ax.set_xlabel("displacement ($h$)")
ax.set_ylabel(r"a ($\lambda$)")
ax.legend()

fig.savefig("19_disp_a.pdf")

indexes = np.concatenate(make_monotonic_load_indexes(nc_CM.displacement, nc_CM.mean_pressure))

fig, ax = plt.subplots()
ax.plot(nc_CM.mean_pressure[indexes], np.max(a_left, axis=1)[indexes], label="CM, a left")
ax.plot(nc_CM.mean_pressure[indexes], np.max(a_right, axis=1)[indexes], label="CM, a right")

ax.plot(nc_CF.mean_pressure, np.max(nc_CF.a_left, axis=1), label="CF, a left")
ax.plot(nc_CF.mean_pressure, np.max(nc_CF.a_right, axis=1), label="CF, a right")

ax.set_xlabel(r"mean pressure ($\pi E^* h / \lambda$)")
ax.set_ylabel(r"a ($\lambda$)")
ax.legend()

fig.savefig("19_p_a.pdf")

nc_CF.close()
nc_CM.close()

fig, (axgeo, axkc) = plt.subplots(2, 1, sharex=True)
axkc.pcolormesh(x, y,  kc(x, y))
axgeo.plot(x, topography.heights()[:, 0])
axgeo.set_xlabel("x ($\lambda$)")
axgeo.set_ylabel(r"indenter geometry ($h$)")
axkc.set_xlabel("x ($\lambda$)")
axkc.set_ylabel(r"$K_c$ ($\pi E^* h / \sqrt{\lambda}$)")

fig.savefig("19_geometry_and_kc.pdf")