#
# Copyright 2021 Antoine Sanner
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
from NuMPI.IO.NetCDF import NCStructuredGrid

from CrackFront.Optimization.propagate_elastic_line import propagate_rosso_krauth
from CrackFront.Optimization.propagate_elastic_line_pytorch import \
    propagate_rosso_krauth as propagate_rosso_krauth_torch


def test_propagate_rosso_krauth_numpy_vs_pytorch_accuracy():
    params = dict(
        line_length=256,  # starting from 8000 pix numpy starts to slower then cuda
        propagation_length=256,
        rms=.1,
        structural_length=64,
        n_steps=10,
        # randomness:
        seed=0,
        # numerics:
        gtol=1e-10,
        maxit=10000000,
        )
    params.update(initial_a=- np.ones(params["line_length"]) * params["structural_length"] * params["rms"] ** 2 * 2)
    np.random.seed(params["seed"])
    pinning_forces = np.random.normal(size=(params["line_length"], params["propagation_length"])) * params["rms"]

    propagate_rosso_krauth(
        **params,
        pinning_forces=pinning_forces,
        dump_fields=True,
        simulation_type="reciprocating parabolic potential",
        # logger=Logger("simulation.log", outevery=100),
        filename="numpy.nc"
        )

    propagate_rosso_krauth_torch(
        **params,
        pinning_forces=pinning_forces,
        dump_fields=True,
        simulation_type="reciprocating parabolic potential",
        # logger=Logger("simulation.log", outevery=100),
        filename="torch.nc"
        )

    nc_torch = NCStructuredGrid("torch.nc")
    nc_numpy = NCStructuredGrid("numpy.nc")

    # fig, ax = plt.subplots()
    #
    # ax.plot(nc_torch.driving_position, nc_torch.driving_force, "+")
    # ax.plot(nc_numpy.driving_position, nc_numpy.driving_force, "x")
    #
    # fig, ax = plt.subplots()
    # for i in range(len(nc_torch)):
    #     ax.plot(nc_torch.position[i] - nc_numpy.position[i], "-")

    np.testing.assert_allclose(nc_torch.position_mean, nc_numpy.position_mean)
    np.testing.assert_allclose(nc_torch.position, nc_numpy.position, atol=1e-5)
    # 1e-5 is already very small compared to the heterogeneity spacing
    np.testing.assert_allclose(nc_torch.position_rms, nc_numpy.position_rms, )
    np.testing.assert_allclose(nc_torch.driving_force, nc_numpy.driving_force, )
    np.testing.assert_allclose(nc_torch.elastic_potential, nc_numpy.elastic_potential, )
    np.testing.assert_allclose(nc_torch.nit, nc_numpy.nit,)

import threading
import signal
import os
import time
def test_restart():

    params = dict(
        line_length=256,
        propagation_length=256,
        rms=.1,
        structural_length=64,
        n_steps=100,
        # randomness:
        seed=0,
        # numerics:
        gtol=1e-10,
        maxit=10000000,
        )
    params.update(initial_a=- np.ones(params["line_length"]) * params["structural_length"] * params["rms"] ** 2 * 2)
    np.random.seed(params["seed"])
    pinning_forces = np.random.normal(size=(params["line_length"], params["propagation_length"])) * params["rms"]

    propagate_rosso_krauth_torch(
        **params,
        pinning_forces=pinning_forces,
        dump_fields=False,
        simulation_type="reciprocating parabolic potential",
        # logger=Logger("simulation.log", outevery=100),
        filename="uninterupted.nc"
        )

    pid = os.getpid()
    # https://stackoverflow.com/questions/26158373/how-to-really-test-signal-handling-in-python
    def trigger_signal():
        # You could do something more robust, e.g. wait until port is listening
        time.sleep(1)
        os.kill(pid, signal.SIGUSR1)

    thread = threading.Thread(target=trigger_signal)
    thread.daemon = True
    thread.start()

    # simulate until interrupted by signal
    assert not propagate_rosso_krauth_torch(
        **params,
        pinning_forces=pinning_forces,
        dump_fields=False,
        simulation_type="reciprocating parabolic potential",
        # logger=Logger("simulation.log", outevery=100),
        filename="interupted.nc",
        handle_signals=True,
        restart=False,
        ), "Simulation finished successfully before being interupted"

    # restart and finish simulation
    params.update(initial_a=np.load("restart_position.npy"))
    assert propagate_rosso_krauth_torch(
        **params,
        pinning_forces=pinning_forces,
        dump_fields=False,
        simulation_type="reciprocating parabolic potential",
        # logger=Logger("simulation.log", outevery=100),
        filename="interupted.nc",
        handle_signals=False,
        restart=True,
        )

    # Now we assert the result was unaffected by the restart
    nc_uninterupted = NCStructuredGrid("uninterupted.nc")
    nc_interupted = NCStructuredGrid("interupted.nc")

    np.testing.assert_allclose(nc_interupted.position_mean, nc_uninterupted.position_mean)
    #np.testing.assert_allclose(nc_interupted.position, nc_uninterupted.position, atol=1e-5)
    # 1e-5 is already very small compared to the heterogeneity spacing
    np.testing.assert_allclose(nc_interupted.position_rms, nc_uninterupted.position_rms, )
    np.testing.assert_allclose(nc_interupted.driving_force, nc_uninterupted.driving_force, )
    np.testing.assert_allclose(nc_interupted.elastic_potential, nc_uninterupted.elastic_potential, )
    np.testing.assert_allclose(nc_interupted.nit, nc_uninterupted.nit,)
