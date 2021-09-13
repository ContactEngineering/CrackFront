import numpy as np
from CrackFront.Circular import pol2cart


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
        positive means peaks and negative means valleys
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
    SIF = 1 / (nx * ny) * np.sum(np.fft.fft2(roughness.heights())[:, :, np.newaxis, np.newaxis] * kernel * np.exp(
        1j * (q_propagation * radius.reshape(1, 1, 1, -1))), axis=(0, 1))

    return SIF
