

import numpy as np
from Adhesion.System import BoundedSmoothContactSystem
from Adhesion.Interactions import PowerLaw
from ContactMechanics import PeriodicFFTElasticHalfSpace
from ContactMechanics.Tools.Logger import screen, Logger
from SurfaceTopography import make_sphere, Topography

from PythonBib.TopographyGeneration import make_topography_from_function

from muFFT.NetCDF import NCStructuredGrid
from NuMPI import MPI
import time
import os, shutil 
from numpy import pi
from NuMPI.Tools import Reduction

from Adhesion.ReferenceSolutions.sinewave import JKR
from CrackFront.Straight import SinewaveCrackFrontLoad
from CrackFront.Optimization import trustregion_newton_cg

comm = MPI.COMM_WORLD
print(f"comsize:{comm.size}")
pnp = Reduction(comm)
outputdir="."
#################### Paramters from JKR nondimensionalisation                  

Es = 1 / np.pi                                                      
h=1. # amplitude (half peak to valley) of the sinewave                                                                         
sinewave_lambda = 1.                                         
sx = 1.                                                                              
#################### Parameter Definition
# JKR
alpha = 0.3#$alpha
work_of_adhesion = w = alpha**2  * np.pi / 2
#finite interaction range
length_parameter = 0.1#$rho

# discretisation
nx = 256#$npx # number of pixels in the x direction
ny = 256#$npy

dx= sx / nx
dy = dx
sy = dy * ny

# fluctuating work of adhesion
dw = 0.5# $dw # maximum work of adhesion fluctuation
lcor = 0.3#$lcor # correlation length: length of the highcut
seed = 1#$seed
fluctuation_length = sy

## simulation parameters
max_rel_area = maxarea = 0.5# $maxarea
starting_displacement = -.25
delta_d = 0.005


np.random.seed(seed)
# generated a random noise topopgraphy, filter it to force correlation lengtrh and set mean to 0
w_fluct_topo = Topography(
    np.random.uniform(size=(nx, ny)), physical_sizes=(sx, sy), periodic=True
    ).highcut(cutoff_wavelength=lcor).detrend()

w_topo = Topography((w_fluct_topo.scale(dw / np.max(np.abs(w_fluct_topo.heights()))
                    ).heights() + 1) * work_of_adhesion,
                    w_fluct_topo.physical_sizes)

k_topo = Topography(np.sqrt(2 * Es * w_topo.heights()),
                    physical_sizes=w_topo.physical_sizes, periodic=True)
k_topo_interp = k_topo.interpolate_bicubic()

#########################################

interaction = PowerLaw(w_topo.heights(),
                       #                         v that way the max stress is still w / rho
                       3 * length_parameter *
                       np.sqrt(w_topo.heights() / work_of_adhesion),
                       3
                       , communicator=comm)



print("setup halfspace")
halfspace = PeriodicFFTElasticHalfSpace((nx, ny), Es, (sx, sy),
                                       communicator=comm,
                                       fft="mpi")
print("create topography")


topography = make_topography_from_function(
    lambda x,y: (-1 - np.cos(2 * np.pi * x / sx)),
    (sx,sy),
    subdomain_locations=halfspace.topography_subdomain_locations,
    nb_subdomain_grid_pts=halfspace.topography_nb_subdomain_grid_pts,
    nb_grid_pts=(nx, ny),
    periodic=True,
    communicator=comm
)
cx, cy = [s / 2 - s /n /2 for s, n in zip((sx, sy), (nx, ny))]
x, y = topography.positions()

def simulate_CM():
    system = BoundedSmoothContactSystem(halfspace, interaction, topography)

    gtol=1e-4
    if __name__ == '__main__':

        monitor=None
        disp0=None
        print("create nc file")
        ncfile = NCStructuredGrid(outputdir+"/CM.nc", mode="w" ,
                                  nb_domain_grid_pts=system.surface.nb_grid_pts,
                                  decomposition='subdomain',
                                  subdomain_locations=system.surface.subdomain_locations,
                                  nb_subdomain_grid_pts=system.surface.nb_subdomain_grid_pts,
                                  communicator=comm)

        starttime = time.time()
        try:
            mode="forward"

            counter = 1
            i = 0
            j = 0
            displacement = starting_displacement
            mean_deformation = 0
            min_max_blind_cg = 8
            max_blind_cg = 0
            min_maxiter = 4000
            max_iter = min_maxiter
            main_logger = Logger("main.log")
            absstarttime = time.time()
            while True:

                #print("##############################################################")
                print("displacement = {}".format(displacement))
                #print("##############################################################")
                # change disp0 to preserve gap
                if disp0 is not None:
                    disp0 += displacement - displacement_prev
                displacement_prev = displacement

                starttime= time.time()
                sol = system.minimize_proxy(
                    disp0=disp0,
                    lbounds="auto",
                    options=dict(gtol=gtol * abs(interaction.max_tensile) * system.surface.area_per_pt, ftol=0, maxcor=3),
                    logger=Logger("laststep.log"),
                    offset=displacement,
                    callback=None
                )
                elapsed_time=time.time() - starttime
                assert sol.success, sol.message

                u = disp0 = sol.x
                mean_deformation_prev = mean_deformation
                mean_deformation = pnp.sum(u) / np.prod(halfspace.nb_domain_grid_pts)

                force = - halfspace.evaluate_force(u)
                #
                #
                ncfile[i].contacting_points = contacting_points = np.where(system.gap == 0., 1., 0.)
                ncfile[i].contact_area =contact_area= pnp.sum(contacting_points) * halfspace.area_per_pt
                normal_force = force.sum()
                ncfile[i].mean_pressure = normal_force / np.prod(topography.physical_sizes)
                ncfile[i].displacement = displacement
                ncfile[i].mean_deformation = mean_deformation
                ncfile[i].elastic_energy = elastic_energy = system.substrate.energy
                ncfile[i].interaction_energy = interaction_energy = system.interaction.energy
                ncfile[i].energy = energy = system.energy

                rel_area = contact_area / np.prod(topography.physical_sizes)

                main_logger_headers = ["step", "nit", "nfev","walltime","displacement", "mean deformation", "force","frac. area", "energy"]
                main_logger.st(main_logger_headers,
                        [i, sol.nit, sol.nfev,  elapsed_time, displacement, mean_deformation, normal_force, rel_area, energy]
                        )

                ncfile[i].pressures = force[halfspace.local_topography_subdomain_slices] / topography.area_per_pt

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

                i+=1

        finally:
            ncfile.close()
        endtime = time.time()
        print("elapsed time: {}".
              format(endtime-absstarttime))



def simulate_CF_old():
    def kc(x, y):
        return k_topo_interp(x+0.5 * sx, y + 0.5 * sy, derivative=0)

    def dkc(x, y):
        interp_field, interp_derx, interp_dery = k_topo_interp(x+0.5 * sx, y+ 0.5 * sy, derivative=1)
        return interp_derx

    n = 128
    cf = SinewaveCrackFrontLoad(n=n, sy=sy, kc=kc, dkc=dkc)

    nc_CF = NCStructuredGrid("CF.nc", "w", (n,))

    penetration = 0

    area = 0

    mean_pressures = np.concatenate((
        np.linspace(0, 0.3, 200, endpoint=False),
        np.linspace(0.3, -0.15, 200)
        ))

    # initial guess
    a = np.ones(2 * n) * JKR.contact_radius(mean_pressures[0], alpha)

    j = 0

    try:
        for mean_pressure in mean_pressures:
            print(f"mean_pressure: {mean_pressure}")
            sol = trustregion_newton_cg(
                x0=a, gradient=lambda a: cf.gradient(a, mean_pressure),
                hessian=lambda a: cf.hessian(a, mean_pressure),
                trust_radius=0.25 * np.min((np.min(a), fluctuation_length)),
                maxiter=3000,
                gtol=1e-6  # he has issues to reach the gtol at small values of a
                )
            print(sol.message)
            assert sol.success
            print("nit: {}".format(sol.nit))
            a = sol.x
            n = int(len(a) / 2)
            a_left = a[:n]
            a_right = a[n:]
            # nc_CF[j].cm_sim_index = i
            nc_CF[j].a_left = a_left
            nc_CF[j].a_right = a_right
            nc_CF[j].mean_a_left = np.mean(a_left)
            nc_CF[j].mean_a_right = np.mean(a_right)
            nc_CF[j].mean_pressure = mean_pressure
            j = j + 1
    finally:
        nc_CF.close()


def direction_change_index(displacements):
    """
    returns the index of the maximal displacements
    """
    for i in range(0,len(displacements)):
        if displacements[i] > displacements[i+1]:
            return i

def kc(x, y):
    return k_topo_interp(x+0.5 * sx, y, derivative=0)

def dkc(x, y):
    interp_field, interp_derx, interp_dery = k_topo_interp(x+0.5 * sx, y, derivative=1)
    return interp_derx

def simulate_CF_following_CM(pausetime=0.00001):

    n = 128
    cf = SinewaveCrackFrontLoad(n=n, sy=sy, kc=kc, dkc=dkc)

    #TODO: implement hessian product
    P = 0.

    import scipy.optimize
    #print(scipy.optimize.check_grad(lambda a: gradient(a, P),
    #                                lambda a: hessian(a, P), x0=a))

    nc_CM = NCStructuredGrid("CM.nc")
    nc_CF = NCStructuredGrid("CF.nc", "w", (n, ) )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    try:
        j = 0
        a = None # take care not to take a too small initial guess
        for i in np.concatenate(make_monotonic_load_indexes(nc_CM.displacement, nc_CM.mean_pressure))[1:]:
            print("CM frac contact area: {}".format(nc_CM[i].contact_area / (sx * sy)))
            if nc_CM[i].contact_area == 0:
                continue
            if nc_CM[i].mean_pressure <= 0:
                if nc_CM[i].mean_pressure >=  nc_CM[i-1].mean_pressure:
                    continue



            a_left = np.max( ( 1/2 - x ) * nc_CM[i].contacting_points, axis=0)
            a_right = np.max( ( x - 1/2 ) * nc_CM[i].contacting_points, axis=0)
            ax.plot(nc_CM[i].mean_pressure, a_left[0], "<", c="b")
            ax.plot(nc_CM[i].mean_pressure, a_right[0], ">", c="r" )
            plt.pause(pausetime)

            P = nc_CM[i].mean_pressure

            if a is None:
                a = np.ones(2 * n) * JKR.contact_radius(P, alpha)
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
    nc_CM = NCStructuredGrid("CM.nc")
    fig, ax = plt.subplots()

    ax.plot(nc_CM.displacement, nc_CM.mean_pressure)
    increasing_load_indexes, decreasing_load_indexes = make_monotonic_load_indexes(nc_CM.displacement, nc_CM.mean_pressure)
    ax.plot(nc_CM.displacement[increasing_load_indexes], nc_CM.mean_pressure[increasing_load_indexes], "o-")
    ax.plot(nc_CM.displacement[decreasing_load_indexes], nc_CM.mean_pressure[decreasing_load_indexes], "+-")


def make_animation_both():
    import matplotlib.pyplot as plt
    from PythonBib.Plotting.utilitaires_plot import MidpointNormalize
    from matplotlib.animation import FuncAnimation
    max_stress = w / length_parameter
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
    nc_CM = NCStructuredGrid("CM.nc")
    nc_CF = NCStructuredGrid("CF.nc")

    x, y = topography.positions()
    zoom = False
    def animate(j):
        index = nc_CF[j].cm_sim_index
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        pressures = nc_CM.pressures[index, ...]

        workcmap = plt.get_cmap("coolwarm")
        ax.imshow(w_topo.heights().T, cmap=workcmap)

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

    FuncAnimation(fig, animate, frames=len(nc_CF), interval=50).save("both.mp4")


if __name__ == "__main__":
    #simulate_CM()
    #simulate_CF()
    #simulate_CF_following_CM()
    make_animation_both()
    import matplotlib.pyplot as plt
    nc_CM = NCStructuredGrid("CM.nc")
    nc_CF = NCStructuredGrid("CF.nc")

    cx, cy = [s / 2 -  s /n /2 for s, n in zip((sx, sy), (nx, ny))]
    x, y = topography.positions()

    a_left = np.max( (1/2 - x) * nc_CM.contacting_points, axis=1)
    a_right = np.max( ( x - 1/2 ) * nc_CM.contacting_points, axis=1)

    #indexes = np.concatenate(make_monotonic_load_indexes(nc_CM.displacement, nc_CM.mean_pressure))

    indexes = slice(None)

    fig, ax = plt.subplots()
    ax.plot(nc_CM.mean_pressure[indexes], np.max(a_left, axis=1)[indexes], label="CM, a left")
    ax.plot(nc_CM.mean_pressure[indexes], np.max(a_right, axis=1)[indexes], label="CM, a right")

    ax.plot(nc_CF.mean_pressure, np.max(nc_CF.a_left, axis=1), label="CF, a left")
    ax.plot(nc_CF.mean_pressure, np.max(nc_CF.a_right, axis=1), label="CF, a right")
    ax.legend()
    plt.show()