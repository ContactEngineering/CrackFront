#!/usr/bin/env ipython

import matplotlib.pyplot as plt
import argparse

from muFFT.NetCDF import NCStructuredGrid

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