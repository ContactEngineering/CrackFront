# # Vizualizing the relationship between SIF and roughness 

# +
from CrackFront.Roughness import straight_crack_sif_from_roughness
from SurfaceTopography.Special import make_topography_from_function
import numpy as np
import matplotlib.pyplot as plt

plt.set_cmap("coolwarm")
# -

# \begin{align}\label{eq:SIF_from_h}
# K^\mathrm{R}(a, z) 
# &= - \frac{1}{4\pi^2} \int \limits_{-\infty}^\infty \int \limits_{-\infty}^\infty dq_x dq_z
# e^{i (q_z z +q_x a)} \frac{E^*}{\sqrt{2}} \sqrt{(|q_z| + i q_x)}  h(q_x, q_z)
# \\
# &= - \frac{1}{4\pi^2} \int \limits_{-\infty}^\infty \int \limits_{-\infty}^\infty dq_x dq_z
# e^{i (q_z z +q_x a)} \frac{E^*}{\sqrt{2}} \sqrt{|q|} e^{i \phi / 2}  h(q_x, q_z)
# \end{align}
# qith $\phi = \arctan(q_x/|q_z| )$
#
# Example for simple waviness $h(x, z ) = \cos(\vec q \cdot \vec{x})$: 
#
# \begin{equation}
# K(x,z) = - \sqrt{|q|} \frac{E^\prime }{\sqrt{2}} \cos(\vec q \cdot \vec{x} + \phi / 2)
# \end{equation}
#
# This means that when the $\vec q$ is tangential to the crack front they are in phase.
#
# When $\vec q$ is perpendicular, there is a phase shift of $\pi / 4$.
#
# For intermediate directions, the phase shift is smaller and identical along $q_z$ and $q_x$
#

# ## Waviness tangential to the front: work of adhesion and waviness perfectly in phase.

# +

# x: towards noncontact area, normal to the crack front
# y: tangential to the crack front
nx = ny = 64


angle = np.pi / 2  # wrt the normal of the crack front towards the noncontact area
wavelength = 0.2

qx = np.cos(angle) * 2 * np.pi / wavelength 
qy = np.sin(angle) * 2 * np.pi / wavelength 


sy =  2 * 2 *np.pi / qy
sx = sy

topography = make_topography_from_function(lambda x, y: np.cos(qx * x + qy * y), physical_sizes=(sx, sy), nb_grid_pts=(nx, ny), periodic=True )
sif = straight_crack_sif_from_roughness(topography)

fig, (ax1, ax2) = plt.subplots(1,2)
plt.colorbar(ax1.imshow(topography.heights(), extent=(0, sx, 0, sy)), label="heights", ax=ax1)
ax1.grid(False)

plt.colorbar(ax2.imshow(-sif, extent=(0, sx, 0, sy)), label="- stress intensity factor", ax=ax2)
ax2.grid(False)

ax1.set_xlabel("x")
ax2.set_xlabel("x")
ax1.set_ylabel("y")
ax2.set_ylabel("y")

fig.tight_layout()

fig, (axx, axy) = plt.subplots(1,2)

x,y,z = topography.positions_and_heights()
axx.plot(x[:,0], z[:, 0], label="heights")
axy.plot(y[0,:], z[0,:], label="heights")

z = -sif / np.max(sif)
axx.plot(x[:,0], z[:, 0], label="-sif  / max(sif)")
axy.plot(y[0,:], z[0,:], label="-sif  / max(sif)")

axx.set_xlabel("x")
axy.set_xlabel("y")
axx.legend()

# -

# ## Waviness along x: phaseshift: max. work of adhesion slightly ahead of the roughness peak.

# +

# x: towards noncontact area, normal to the crack front
# y: tangential to the crack front
nx = ny = 64


angle = 0 # wrt the normal of the crack front towards the noncontact area
wavelength = 0.2

qx = np.cos(angle) * 2 * np.pi / wavelength 
qy = np.sin(angle) * 2 * np.pi / wavelength 

sx = 2 * 2 *np.pi / qx
sy = sx

topography = make_topography_from_function(lambda x, y: np.cos(qx * x + qy * y), physical_sizes=(sx, sy), nb_grid_pts=(nx, ny), periodic=True )
sif = straight_crack_sif_from_roughness(topography)

fig, (ax1, ax2) = plt.subplots(1,2)
plt.colorbar(ax1.imshow(topography.heights(), extent=(0, sx, 0, sy)), label="heights", ax=ax1)
ax1.grid(False)

plt.colorbar(ax2.imshow(-sif, extent=(0, sx, 0, sy)), label="- stress intensity factor", ax=ax2)
ax2.grid(False)

ax1.set_xlabel("x")
ax2.set_xlabel("x")
ax1.set_ylabel("y")
ax2.set_ylabel("y")

fig.tight_layout()

fig, (axx, axy) = plt.subplots(1,2)

x,y,z = topography.positions_and_heights()
axx.plot(x[:,0], z[:, 0], label="heights")
axy.plot(y[0,:], z[0,:], label="heights")

z = -sif / np.max(sif)
axx.plot(x[:,0], z[:, 0], label="-sif  / max(sif)")
axy.plot(y[0,:], z[0,:], label="-sif  / max(sif)")

axx.set_xlabel("x")
axy.set_xlabel("y")
axx.legend()

# -

# ## Intermediate wavevectors: same phaseshift in both directions, but the phaseshift is smaller

# +
# x: towards noncontact area, normal to the crack front
# y: tangential to the crack front
nx = ny = 64


angle = 0.7 *   np.pi / 2  # wrt the normal of the crack front towards the noncontact area
wavelength = 0.2

qx = np.cos(angle) * 2 * np.pi / wavelength 
qy = np.sin(angle) * 2 * np.pi / wavelength 

sx = 2 * 2 *np.pi / qx
sy = 2 * 2 *np.pi / qy

topography = make_topography_from_function(lambda x, y: np.cos(qx * x + qy * y), physical_sizes=(sx, sy), nb_grid_pts=(nx, ny), periodic=True )
sif = straight_crack_sif_from_roughness(topography)

fig, (ax1, ax2) = plt.subplots(1,2)
plt.colorbar(ax1.imshow(topography.heights(), extent=(0, sx, 0, sy)), label="heights", ax=ax1)
ax1.grid(False)

plt.colorbar(ax2.imshow(-sif, extent=(0, sx, 0, sy)), label="- stress intensity factor", ax=ax2)
ax2.grid(False)

ax1.set_xlabel("x")
ax2.set_xlabel("x")
ax1.set_ylabel("y")
ax2.set_ylabel("y")

fig.tight_layout()

fig, (axx, axy) = plt.subplots(1,2)

x,y,z = topography.positions_and_heights()
axx.plot(x[:,0], z[:, 0], label="heights")
axy.plot(y[0,:], z[0,:], label="heights")

z = -sif / np.max(sif)
axx.plot(x[:,0], z[:, 0], label="-sif  / max(sif)")
axy.plot(y[0,:], z[0,:], label="-sif  / max(sif)")

axx.set_xlabel("x")
axy.set_xlabel("y")
axx.legend()

# -






