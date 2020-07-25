"""
A comment on the Nyquist frequency:

We will preferably use powers of 2 in the number of grid points, because
FFTs are much faster in this case.

For even number of grid points, there is only one (complex) entry in the
Fourier spectrum at the Nyquist frequency.

The discrete representation of the sinewave at the Nyquist frequency (spanning
two grid points) can represent the sampling of multiple sinewaves with
amplitudes and phases. One has to make an assumption on the phase.

It is common to choose the  phase that corresponds to the smallest amplitude
and hence the smallest deformation energy.

It is also the assumption we make when doing a fourier-interpolation.

The discretistion points of the crack front are also collocation points of the
fracture toughness landscape. There is no reason to choose a fourier
interpolation (implied by the spectral method for elasticity) that has the
peaks slightly offset wrt. to the collocation points.


"""