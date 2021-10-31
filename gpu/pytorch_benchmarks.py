import torch
import numpy as np
import time
import matplotlib.pyplot as plt

torch.cuda.is_available()

L = 8

values_numpy = np.random.normal(size=L)
values_torch = torch.from_numpy(values_numpy)
values_cuda = values_torch.cuda()

values_torch

values_numpy

values_cuda

4096 * 64

times_numpy = []
times_torch = []
times_cuda = []
repeats=10
Ls = [256, 1024, 4096, 16384, 65536, 262144]
for L in Ls:
    values_numpy = np.random.normal(size=L)
    values_torch = torch.from_numpy(values_numpy)
    values_cuda = values_torch.cuda()
    
    start_time = time.time()
    for i in range(repeats):
        r = np.fft.rfft(values_numpy)
    times_numpy.append(time.time() - start_time)
    
    
    start_time = time.time()
    for i in range(repeats):
        r = torch.fft.rfft(values_torch)
    times_torch.append(time.time() - start_time)
    
    start_time = time.time()
    for i in range(repeats):
        r = torch.fft.rfft(values_cuda)
    times_cuda.append(time.time() - start_time)
    

# +
fig, ax = plt.subplots()

ax.plot(Ls, times_numpy, "o", label="numpy")
ax.plot(Ls, times_torch, "+", label="torch, CPU")
ax.plot(Ls, times_cuda, "x", label="torch, cuda")

ax.legend()

ax.set_xscale("log")
ax.set_yscale("log")
# -

# The speedup of the FFT is indeed excellent

# # it is better to do maxes instead of masks

times_numpy = []
times_cuda2 = []
times_cuda = []
repeats=10
Ls = [256, 1024, 4096, 16384, 65536, 262144]
for L in Ls:
    values_numpy = np.random.normal(size=L)
    values_numpy2 = np.random.normal(size=L)
    
    values_cuda = torch.from_numpy(values_numpy).cuda()
    values_cuda2 = torch.from_numpy(values_numpy).cuda()
    
    values2_cuda = torch.from_numpy(values_numpy).cuda()
    values2_cuda2 = torch.from_numpy(values_numpy).cuda()
    
    start_time = time.time()
    for i in range(repeats):
        mask = values_numpy > values_numpy2
        values_numpy[mask] = values_numpy2[mask]
    times_numpy.append(time.time() - start_time)
    
    
    
    start_time = time.time()
    for i in range(repeats):
        mask = values_cuda > values_cuda2
        values_cuda[mask] = values_cuda2[mask]
    times_cuda.append(time.time() - start_time)
    
    
    
    start_time = time.time()
    for i in range(repeats):
        values_cuda = torch.maximum(values_cuda, values_cuda2)
    times_cuda2.append(time.time() - start_time)
    

# +
fig, ax = plt.subplots()

ax.plot(Ls, times_numpy, "o", label="numpy, mask")
ax.plot(Ls, times_cuda, "x", label="cuda, mask")
ax.plot(Ls, times_cuda2, "+", label="cuda, max")

ax.legend()

ax.set_xscale("log")
ax.set_yscale("log")
# -


