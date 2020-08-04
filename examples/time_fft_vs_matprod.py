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