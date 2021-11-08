import numpy as np
from NuMPI.IO.NetCDF import NCStructuredGrid

from CrackFront.Optimization.propagate_elastic_line import propagate_rosso_krauth
from CrackFront.Optimization.propagate_elastic_line_pytorch import propagate_rosso_krauth as propagate_rosso_krauth_torch

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
        #logger=Logger("simulation.log", outevery=100),
        filename="numpy.nc"
        )

    propagate_rosso_krauth_torch(
        **params,
        pinning_forces=pinning_forces,
        dump_fields=True,
        simulation_type="reciprocating parabolic potential",
        #logger=Logger("simulation.log", outevery=100),
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
