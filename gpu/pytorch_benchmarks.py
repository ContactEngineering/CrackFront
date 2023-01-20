import torch
import numpy as np
import time
import matplotlib.pyplot as plt

torch.cuda.is_available()

L = 8

values_numpy = np.random.normal(size=L)
values_torch = torch.from_numpy(values_numpy)
values_cuda = values_torch.cuda()
values_cuda32 = values_torch.to(dtype=torch.float32).cuda()

values_torch

values_numpy

values_cuda

r = torch.fft.rfft(values_cuda)
r.dtype

values_cuda32.dtype

r = torch.fft.rfft(values_cuda32)
r.dtype


Ls = [256, 1024, 4096, 
      16384, 
      65536, 
      262144,
      1048576,
      4194304
     ]

# +
times_numpy = []
times_torch = []
times_cuda = []
times_cuda32 = []

repeats=1

for L in Ls:
    values_numpy = np.random.normal(size=L)
    values_torch = torch.from_numpy(values_numpy)
    values_cuda = values_torch.cuda()
    values_cuda32 = values_torch.to(dtype=torch.float32).cuda() 
    
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
    
    start_time = time.time()
    for i in range(repeats):
        r = torch.fft.rfft(values_cuda32)
    times_cuda32.append(time.time() - start_time)
    
    


# +
fig, ax = plt.subplots()

ax.plot(Ls, times_numpy, "o", label="numpy")
ax.plot(Ls, times_torch, "+", label="torch, CPU")
ax.plot(Ls, times_cuda, "x", label="torch, cuda, float64")
ax.plot(Ls, times_cuda32, "s", label="torch, cuda, float32")
# I also test what happens 

ax.legend()

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("vector length")
ax.set_ylabel(f"time for {repeats} repeats")

# -

# The speedup of the FFT is indeed excellent

np.divide(times_cuda,times_cuda32)

# Only at a very high number of grid points there is advantage at using float32

# ### Just for fun: how does the 2D FFT speed up ? 

Ls_2d = [64, 256, 1024, 4096, 8192]

# +
times_2d_numpy = []
times_2d_torch = []
times_2d_cuda = []
times_2d_cuda32 = []

for L in Ls_2d:
    values_numpy = np.random.normal(size=(L,L))
    values_torch = torch.from_numpy(values_numpy)
    values_cuda = values_torch.cuda()
    values_cuda32 = values_torch.to(dtype=torch.float32).cuda() 
    
    start_time = time.time()
    for i in range(repeats):
        r = np.fft.rfft(values_numpy)
    times_2d_numpy.append(time.time() - start_time)
    
    
    start_time = time.time()
    for i in range(repeats):
        r = torch.fft.rfft(values_torch)
        
    times_2d_torch.append(time.time() - start_time)
    
    start_time = time.time()
    for i in range(repeats):
        r = torch.fft.rfft(values_cuda)
    times_2d_cuda.append(time.time() - start_time)
    
    start_time = time.time()
    for i in range(repeats):
        r = torch.fft.rfft(values_cuda32)
    times_2d_cuda32.append(time.time() - start_time)
    


# +
fig, (ax_1d, ax_2d) = plt.subplots(2,1, sharex=True)

ax_1d.plot(Ls, times_numpy, "o", label="numpy")
ax_1d.plot(Ls, times_torch, "+", label="torch, CPU")
ax_1d.plot(Ls, times_cuda, "x", label="torch, cuda, float64")
ax_1d.plot(Ls, times_cuda32, "s", label="torch, cuda, float32")
# I also test what happens 

ax_1d.legend()

ax_1d.set_xscale("log")
ax_1d.set_yscale("log")

ax_1d.set_xlabel("linear length")
ax_1d.set_ylabel(f"time for {repeats} repeats")

ax_2d.plot(Ls_2d, times_2d_numpy, "o", label="numpy")
ax_2d.plot(Ls_2d, times_2d_torch, "+", label="torch, CPU")
ax_2d.plot(Ls_2d, times_2d_cuda, "x", label="torch, cuda, float64")
ax_2d.plot(Ls_2d, times_2d_cuda32, "s", label="torch, cuda, float32")
# I also test what happens 

ax_2d.legend()

ax_2d.set_xscale("log")
ax_2d.set_yscale("log")

ax_2d.set_xlabel("linear length")
ax_2d.set_ylabel(f"time for {repeats} repeats")


# +
fig, ax = plt.subplots()

ax.plot(Ls, times_numpy, "o", c="b", label="numpy")
ax.plot(Ls, times_torch, "+", c="b",label="torch, CPU")
ax.plot(Ls, times_cuda, "x", c="b",label="torch, cuda, float64")
ax.plot(Ls, times_cuda32, "s", c="b",label="torch, cuda, float32")


ax.plot(np.array(Ls_2d)**2, times_2d_numpy, "o", c="r",label="2d, numpy")
ax.plot(np.array(Ls_2d)**2, times_2d_torch, "+", c="r",label="2d, torch, CPU")
ax.plot(np.array(Ls_2d)**2, times_2d_cuda, "x", c="r",label="2d, torch, cuda, float64")
ax.plot(np.array(Ls_2d)**2, times_2d_cuda32, "s", c="r",label="2d, torch, cuda, float32")

# I also test what happens 

ax.legend()

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("total length")
ax.set_ylabel(f"time for {repeats} repeats")

# -

# - Per DOF, 2D FFTs are only slightly faster, both in numpy and pytorch
# - using 32bit precision we get 4 orders of magnitude faster. With 64bit let's say 3 orders of magnitude
# - Interestingly,  pytorch is slower then numpy on cpu in 1d, it faster then numpy in 2d 

# # float32 vs. float64 on some other operations

# +
times_numpy = []
times_torch = []
times_cuda = []
times_cuda32 = []

repeats=1

for L in Ls:
    values_numpy = np.random.normal(size=L)
    values_torch = torch.from_numpy(values_numpy)
    values_cuda = values_torch.cuda()
    values_cuda32 = values_torch.to(dtype=torch.float32).cuda() 
    
    values_numpy_2 = np.random.normal(size=L)
    values_torch_2 = torch.from_numpy(values_numpy)
    values_cuda_2 = values_torch.cuda()
    values_cuda32_2 = values_torch.to(dtype=torch.float32).cuda() 
    
    start_time = time.time()
    for i in range(repeats):
        r = values_numpy + values_numpy_2
    times_numpy.append(time.time() - start_time)
    
    
    start_time = time.time()
    for i in range(repeats):
        r = values_torch + values_torch_2
        
    times_torch.append(time.time() - start_time)
    
    start_time = time.time()
    for i in range(repeats):
        r = values_cuda + values_cuda_2
    times_cuda.append(time.time() - start_time)
    
    start_time = time.time()
    for i in range(repeats):
        r = values_cuda32 + values_cuda32_2
    times_cuda32.append(time.time() - start_time)
    
    


# +
fig, ax = plt.subplots()

ax.plot(Ls, times_numpy, "o", label="numpy")
ax.plot(Ls, times_torch, "+", label="torch, CPU")
ax.plot(Ls, times_cuda, "x", label="torch, cuda, float64")
ax.plot(Ls, times_cuda32, "s", label="torch, cuda, float32")
# I also test what happens 

ax.legend()

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("vector length")
ax.set_ylabel(f"time for {repeats} repeats")

ax.set_title("addition")


# +
times_numpy = []
times_torch = []
times_cuda = []
times_cuda32 = []

repeats=1

for L in Ls:
    values_numpy = np.random.normal(size=L)
    values_torch = torch.from_numpy(values_numpy)
    values_cuda = values_torch.cuda()
    values_cuda32 = values_torch.to(dtype=torch.float32).cuda() 
    
    values_numpy_2 = np.random.normal(size=L)
    values_torch_2 = torch.from_numpy(values_numpy)
    values_cuda_2 = values_torch.cuda()
    values_cuda32_2 = values_torch.to(dtype=torch.float32).cuda() 
    
    start_time = time.time()
    for i in range(repeats):
        r = values_numpy * values_numpy_2
    times_numpy.append(time.time() - start_time)
    
    
    start_time = time.time()
    for i in range(repeats):
        r = values_torch * values_torch_2
        
    times_torch.append(time.time() - start_time)
    
    start_time = time.time()
    for i in range(repeats):
        r = values_cuda * values_cuda_2
    times_cuda.append(time.time() - start_time)
    
    start_time = time.time()
    for i in range(repeats):
        r = values_cuda32 * values_cuda32_2
    times_cuda32.append(time.time() - start_time)
    
    


# +
fig, ax = plt.subplots()

ax.plot(Ls, times_numpy, "o", label="numpy")
ax.plot(Ls, times_torch, "+", label="torch, CPU")
ax.plot(Ls, times_cuda, "x", label="torch, cuda, float64")
ax.plot(Ls, times_cuda32, "s", label="torch, cuda, float32")
# I also test what happens 

ax.legend()

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("vector length")
ax.set_ylabel(f"time for {repeats} repeats")

ax.set_title("multiplication")

# -

# Similar conclusion than for the FFT: unless at very large systems I do not have a real benifit of using the float32.
#
# For these large systems I might actually need the float64 precision.

# # it is better to do maxes or wheres instead of masks

# +
times_numpy = []
times_cuda3 = []

times_cuda2 = []
times_cuda = []
repeats=10
for L in Ls:
    values_numpy = np.random.normal(size=L)
    values_numpy2 = np.random.normal(size=L)
    
    values_cuda = torch.from_numpy(values_numpy).cuda()
    values_cuda2 = torch.from_numpy(values_numpy).cuda()
    
    values2_cuda = torch.from_numpy(values_numpy).cuda()
    values2_cuda2 = torch.from_numpy(values_numpy).cuda()
    
    values3_cuda = torch.from_numpy(values_numpy).cuda()
    values3_cuda2 = torch.from_numpy(values_numpy).cuda()
    
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
        values3_cuda = torch.where(values3_cuda > values3_cuda2, values3_cuda, values3_cuda2)
    times_cuda3.append(time.time() - start_time)
    
    start_time = time.time()
    for i in range(repeats):
        values2_cuda = torch.maximum(values2_cuda, values2_cuda2)
    times_cuda2.append(time.time() - start_time)


# +
fig, ax = plt.subplots()

ax.plot(Ls, times_numpy, "o", label="numpy, mask")
ax.plot(Ls, times_cuda, "x", label="cuda, mask")
ax.plot(Ls, times_cuda2, "+", label="cuda, max")
ax.plot(Ls, times_cuda3, "<", label="cuda, where")


ax.legend()

ax.set_xscale("log")
ax.set_yscale("log")


ax.set_xlabel("vector length")
ax.set_ylabel(f"time for {repeats} repeats")
# -
3276832768


# +
times_numpy = []
times_cuda3 = []

times_cuda2 = []
times_cuda = []
repeats=10

for L in Ls:
    values_numpy = np.random.normal(size=L)
    values_numpy2 = np.random.normal(size=L)
    
    values_cuda = torch.from_numpy(values_numpy).cuda()
    values_cuda2 = torch.from_numpy(values_numpy).cuda()
    
    values2_cuda = torch.from_numpy(values_numpy).cuda()
    values2_cuda2 = torch.from_numpy(values_numpy).cuda()
    
    values3_cuda = torch.from_numpy(values_numpy).cuda()
    values3_cuda2 = torch.from_numpy(values_numpy).cuda()
    
    start_time = time.time()
    for i in range(repeats):
        mask = values_numpy > values_numpy2
        values_numpy[mask] = values_numpy2[mask]
    times_numpy.append(time.time() - start_time)
    
    
    
    start_time = time.time()
    for i in range(repeats):
        mask = values_cuda > values_cuda2
        values_cuda[mask] += 1
    times_cuda.append(time.time() - start_time)
    
    start_time = time.time()
    for i in range(repeats):
        values3_cuda += values_cuda > values_cuda2
    times_cuda3.append(time.time() - start_time)
    
    #start_time = time.time()
    #for i in range(repeats):
    #    values2_cuda = torch.maximum(values2_cuda, values2_cuda2)
    #times_cuda2.append(time.time() - start_time)


# +
fig, ax = plt.subplots()

ax.plot(Ls, times_numpy, "o", label="numpy, mask")
ax.plot(Ls, times_cuda, "x", label="cuda, mask")
#ax.plot(Ls, times_cuda2, "+", label="cuda, max")
ax.plot(Ls, times_cuda3, "<", label="cuda, += mask")


ax.legend()

ax.set_xscale("log")
ax.set_yscale("log")


ax.set_xlabel("vector length")
ax.set_ylabel(f"time for {repeats} repeats")
# +
times_numpy = []
times_cuda3 = []

times_cuda2 = []
times_cuda = []
repeats=10

for L in Ls:
    values_numpy = np.random.normal(size=L)
    values_numpy2 = np.random.normal(size=L)
    
    values_cuda = torch.from_numpy(values_numpy).cuda()
    values_cuda2 = torch.from_numpy(values_numpy).cuda()
    
    values2_cuda = torch.from_numpy(values_numpy).cuda()
    values2_cuda2 = torch.from_numpy(values_numpy).cuda()
    
    values3_cuda = torch.from_numpy(values_numpy).cuda()
    values3_cuda2 = torch.from_numpy(values_numpy).cuda()
    
    start_time = time.time()
    for i in range(repeats):
        mask = values_numpy > values_numpy2
        values_numpy[mask] = values_numpy2[mask]
    times_numpy.append(time.time() - start_time)
    
    
    
    start_time = time.time()
    for i in range(repeats):
        mask = values_cuda > values_cuda2
        values_cuda[mask] += 1
    times_cuda.append(time.time() - start_time)
    
    start_time = time.time()
    for i in range(repeats):
        values3_cuda += values_cuda > values_cuda2
    times_cuda3.append(time.time() - start_time)
    
    start_time = time.time()
    for i in range(repeats):
        values2_cuda.add_(values_cuda > values_cuda2)
    times_cuda2.append(time.time() - start_time)


# +
fig, ax = plt.subplots()

ax.plot(Ls, times_numpy, "o", label="numpy, mask")
ax.plot(Ls, times_cuda, "x", label="cuda, mask")
ax.plot(Ls, times_cuda2, "+", label="cuda, add_")
ax.plot(Ls, times_cuda3, "<", label="cuda, += mask")


ax.legend()

ax.set_xscale("log")
ax.set_yscale("log")


ax.set_xlabel("vector length")
ax.set_ylabel(f"time for {repeats} repeats")
# -
# ### rfftfreq is faster starting from 16000 pixels only.

# +
times_numpy = []
times_torch = []
times_cuda = []
times_numpy_and_transfer = []
repeats=10

for L in Ls:
    values_numpy = np.random.normal(size=L)
    values_torch = torch.from_numpy(values_numpy)
    values_cuda = values_torch.cuda()
    
    start_time = time.time()
    for i in range(repeats):
        r = np.fft.rfftfreq(L, 1/L)
    times_numpy.append(time.time() - start_time)
    
    
    start_time = time.time()
    for i in range(repeats):
        r = torch.fft.rfftfreq(L, 1/L, device=torch.device("cpu"))
    times_torch.append(time.time() - start_time)
    
    start_time = time.time()
    for i in range(repeats):
        r = torch.fft.rfftfreq(L, 1/L, device=torch.device("cuda"))
    times_cuda.append(time.time() - start_time)
    start_time = time.time()
    for i in range(repeats):
        r = np.fft.rfftfreq(L, 1/L,)
        r = torch.from_numpy(r).to(device=torch.device("cuda"))
    times_numpy_and_transfer.append(time.time() - start_time)

# +
fig, ax = plt.subplots()


ax.plot(Ls, times_numpy, "o", label="numpy")
ax.plot(Ls, times_numpy_and_transfer, "s", label="numpy and transfer")

ax.plot(Ls, times_torch, "+", label="torch, CPU")
ax.plot(Ls, times_cuda, "x", label="torch, cuda")

ax.legend()

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("vector length")
ax.set_ylabel(f"time for {repeats} repeats")

# -

Ls








