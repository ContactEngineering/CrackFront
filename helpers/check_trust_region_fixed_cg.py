import numpy as np
import scipy

from CrackFront.GenericElasticLine import ElasticLine
from CrackFront.Optimization import trustregion_newton_cg

penetration = 0

w = 10

w_amplitude = 0.4

n_rays = 8
L = npx = 32

# l_waviness = 0.1
q_waviness = 2 * np.pi / 0.1

z = np.arange(L)

def pinning_field(a, der="0"):
    if der == "0":
        return w
    elif der == "1":
        return np.zeros_like(a)

cf = ElasticLine(npx, Lk=npx / 3, pinning_field=pinning_field)


driving_position = 0
a_init = np.random.normal(size=npx)
jac_mag_init = scipy.linalg.norm(cf.gradient(a_init, driving_position))
def cg_tolerance(jac_mag):
    #print(jac_mag)
    return 0.5 * min(1., jac_mag / jac_mag_init) * jac_mag

sol = trustregion_newton_cg(
        a_init,
        lambda a: cf.gradient(a, driving_position),
        hessian_product=lambda a, p: cf.hessian_product(p, a),
        trust_radius=100,
        gtol=1e-11,
        cg_tolerance=cg_tolerance
        )

# for small trust radii we have a lot of n_hits_boundary
# for large radii we have 0
# nit and njev are always almost the same, that means that nit corresponds to the total number of CG iterations

print(sol)
