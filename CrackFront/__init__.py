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

from .DiscoverVersion import __version__  # noqa: F401
