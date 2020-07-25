"""
The matrix product here is a convolution.

We get a huge speedup by using the fft.

"""


import numpy as np
import timeit

commonsetup="""
import numpy as np
n = 8192
sy = 1.
q = 2 * np.pi * np.fft.rfftfreq(n, sy / n)
a_test = np.random.normal(size = n)
"""

timemat = timeit.timeit("elastic_jac @ a_test", setup=commonsetup+
"""
elastic_jac = np.zeros((n,n))
v = np.fft.irfft(q/2, n=n)
for i in range(n):
    for j in range(n):
        elastic_jac[i, j] = v[i-j]
""", number=100)
#elastic_jac @ a_test

timefft = timeit.timeit("np.fft.irfft(q / 2 * np.fft.rfft(a_test), n=n)",
setup=commonsetup +
"""
elastic_jac = np.zeros((n,n))
v = np.fft.irfft(q/2, n=n)
for i in range(n):
    for j in range(n):
        elastic_jac[i, j] = v[i-j]
""", number=100)

print(f"timemat={timemat}, timefft={timefft}")