#
# Copyright 2021, 2023 Antoine Sanner
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


def straight_crack_sif_from_roughness(roughness, Es=1):
    """
    Parameters:
    -----------
    roughness: Topography
    Es: float, optional
        Johnson's contact modulus default 1

    """
    nx, ny = roughness.nb_grid_pts
    sx, sy = roughness.physical_sizes
    dx, dy = roughness.pixel_size

    # qx, qy = compute_wavevectors((nx, ny), (sx, sy), 2)
    qx = 2 * np.pi * np.fft.fftfreq(nx, dx).reshape(-1, 1)
    qy = 2 * np.pi * np.fft.rfftfreq(ny, dy).reshape(1, -1)

    q = np.sqrt(qx ** 2 + qy ** 2)

    kernel = Es / np.sqrt(2) * q / np.sqrt(abs(qy) - 1j * qx)
    kernel[0, 0] = 0
    K_wavy = - np.fft.irfft2(kernel * np.fft.rfft2(roughness.heights()),
                             s=(nx, ny))

    return K_wavy


def circular_crack_sif_from_roughness(roughness, radius, angle, Es=1):
    r"""

    Computes the integral

    ..math ::
        K^R(a, \theta) = - \frac{1}{4\pi^2} \int \limits_{-\infty}^\infty \int \limits_{-\infty}^\infty dq_x dq_z
e^{i \vec q \cdot \vec x} \frac{E^*}{\sqrt{2}} \sqrt{|q_z \cos\theta - q_x \sin \theta| + i (q_x \cos \theta + q_z \sin \theta)}  h(\vec q)

    by brute force summation.
    :math:`\vec{x} = (\cos\theta a, \sin\theta a)`

    Parameters:
    -----------
    radius: float or array of floats
        contact raius where to evaluate the stress intensity factor
    angle: float or array of floats
        angle wher to evaluate the stresss intensity factor
    roughness: Topography
        positive means peaks and negative means valleys.
        We place the center of the polar coordinate system in the center of the topography
    Es: float, optional
        Johnson's contact modulus default 1

    """

    nx, ny = roughness.nb_grid_pts
    sx, sy = roughness.physical_sizes
    dx, dy = roughness.pixel_size

    # qx, qy = compute_wavevectors((nx, ny), (sx, sy), 2)
    qx = 2 * np.pi * np.fft.fftfreq(nx, dx).reshape(-1, 1, 1, 1)
    qy = 2 * np.pi * np.fft.fftfreq(ny, dy).reshape(1, -1, 1, 1)

    _angle = angle.reshape(1, 1, -1, 1)
    # direction tangential to the crack front
    q_front = qy * np.cos(_angle) - qx * np.sin(_angle)
    # direction in crack propagation direction
    q_propagation = qx * np.cos(_angle) + qy * np.sin(_angle)

    kernel = Es / np.sqrt(2) * np.sqrt(abs(q_front) + 1j * q_propagation)
    kernel[0, 0] = 0
    # place the center of the coordinate system in the center of the topography
    heights = np.roll(roughness.heights(), [n // 2 for n in roughness.nb_grid_pts], axis=(0, 1))

    SIF = - 1 / (nx * ny) * np.sum(np.fft.fft2(heights)[:, :, np.newaxis, np.newaxis] * kernel * np.exp(
        1j * (q_propagation * radius.reshape(1, 1, 1, -1))), axis=(0, 1))

    return SIF


def circular_crack_sif_from_roughness_memory_friendly(roughness, radius, angle, Es=1, verbose=0):

    nx, ny = roughness.nb_grid_pts
    sx, sy = roughness.physical_sizes
    dx, dy = roughness.pixel_size

    # qx, qy = compute_wavevectors((nx, ny), (sx, sy), 2)
    qx = 2 * np.pi * np.fft.fftfreq(nx, dx).reshape(-1, 1)
    qy = 2 * np.pi * np.fft.fftfreq(ny, dy).reshape(1, -1)

    heights = np.roll(roughness.heights(), [n // 2 for n in roughness.nb_grid_pts], axis=(0, 1))
    heights_fourier = np.fft.fft2(heights)[:, :]

    # place the center of the coordinate system in the center of the topography

    _radius = radius.flat
    SIF = np.zeros((len(angle), len(_radius)))
    for idx_angle in range(len(angle)):
        if verbose:
            print("{:d} / {:d}\r".format(idx_angle, len(angle)))
            sys.stdout.flush()
        _angle = angle[idx_angle]
        # direction tangential to the crack front
        q_front = qy * np.cos(_angle) - qx * np.sin(_angle)
        # direction in crack propagation direction
        q_propagation = qx * np.cos(_angle) + qy * np.sin(_angle)
        kernel = Es / np.sqrt(2) * np.sqrt(abs(q_front) + 1j * q_propagation)
        kernel[0, 0] = 0
        for idx_radius in range(len(_radius)):
            SIF[idx_angle, idx_radius] = - 1 / (nx * ny) * np.sum((heights_fourier * kernel
                                                                   * np.exp(1j * (q_propagation * _radius[idx_radius]))
                                                                   ).real, axis=(0, 1))

    return SIF

def circular_crack_sif_from_roughness_via_bicubic(roughness, radius, angle, nb_grid_pts_fourier_interp, Es=1, verbose=0):

    # TODO: There is no bicubic interpolation here !
    #
    # Did I mean to implement this : dtool_item_by_name $FRCT/de0f0c45-51f9-49a3-96ad-b8b8d3c218b5 comparison_lc0.2.html ?
    # And this is just a copy paste from above ? (computational complexity seems to be N4)
    nx, ny = roughness.nb_grid_pts
    sx, sy = roughness.physical_sizes
    dx, dy = roughness.pixel_size

    # qx, qy = compute_wavevectors((nx, ny), (sx, sy), 2)
    qx = 2 * np.pi * np.fft.fftfreq(nx, dx).reshape(-1, 1)
    qy = 2 * np.pi * np.fft.fftfreq(ny, dy).reshape(1, -1)

    heights = np.roll(roughness.heights(), [n // 2 for n in roughness.nb_grid_pts], axis=(0, 1))
    heights_fourier = np.fft.fft2(heights)[:, :]

    # place the center of the coordinate system in the center of the topography

    _radius = radius.flat
    SIF = np.zeros((len(angle), len(_radius)))
    for idx_angle in range(len(angle)):
        if verbose:
            print("{:d} / {:d}\r".format(idx_angle, len(angle)))
            sys.stdout.flush()
        _angle = angle[idx_angle]
        # direction tangential to the crack front
        q_front = qy * np.cos(_angle) - qx * np.sin(_angle)
        # direction in crack propagation direction
        q_propagation = qx * np.cos(_angle) + qy * np.sin(_angle)
        kernel = Es / np.sqrt(2) * np.sqrt(abs(q_front) + 1j * q_propagation)
        kernel[0, 0] = 0
        for idx_radius in range(len(_radius)): # TODO: Can this be made without a for loop ?
            SIF[idx_angle, idx_radius] = - 1 / (nx * ny) * np.sum((heights_fourier * kernel
                                                                   * np.exp(1j * (q_propagation * _radius[idx_radius]))
                                                                   ).real, axis=(0, 1))

    return SIF
