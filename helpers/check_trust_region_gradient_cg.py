import numpy as np
import scipy

from CrackFront.GenericElasticLine import ElasticLine
from CrackFront.Optimization import fixed_radius_trustregion_newton_cg, gradient_based_trustregion_newton_cg

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
        trust_radius=0.5,
        reduce_threshold=0.8,
        trust_threshold=0.9,
        max_trust_radius=100,
        gtol=1e-11,
        verbose=True
        )

print(sol)
