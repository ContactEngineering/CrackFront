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

import matplotlib.pyplot as plt
import argparse

from NuMPI.IO.NetCDF import NCStructuredGrid

parser = argparse.ArgumentParser()
parser.add_argument('ncfile', metavar='nc', type=str,
                    help='nc file')
args = parser.parse_args()

nc = NCStructuredGrid(args.ncfile)
fig, ax = plt.subplots()

ax.plot(nc.nit, ".", label="nb Newton iterations")
ax.plot(nc.n_hits_boundary, "x", label="nb trustregion hit", )

ax.set_xlabel("simulation step")
ax.set_ylabel("number")

ax.set_yscale("log")

ax.legend()
plt.show()