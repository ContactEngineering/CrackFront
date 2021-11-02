---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Next steps:

## make the code CPU-GPU agnostic
Or only numpy - torch agnostic ?  
    - use += instead of add_
    - Initialisations are difficult in an agnostic way, unless there is a way to override the default for the device. 
    - It seeems that not (https://pytorch.org/docs/stable/notes/cuda.html?highlight=device). 
    - https://pytorch.org/docs/stable/generated/torch.cuda.set_device.html?highlight=torch%20cuda%20set_device#torch.cuda.set_device is probably a way. TODO: test.
    - alternative: provide some eventually empty kwarg for all array initialisations.
## Test that makes sure we get the same result in the cuda code and in the original numpy code
## scaling test showing how cuda compute time increases with system size
## Storage order for pinning field ? 
## use 2nd order polynomials ? -> more polynom coefficients are for free because stored in the fast memory direction
## measure memory requirements
## if needed do caching of the field.



# Some useful resources 

<!-- #region -->

Pytorch examples 

https://github.com/ritchieng/eigenvectors-from-eigenvalues

https://www.youtube.com/playlist?list=PLlMkM4tgfjnJ3I-dbhO9JTw7gNty6o_2m

https://github.com/dionhaefner/pyhpc-benchmarks/blob/master/benchmarks/isoneutral_mixing/isoneutral_pytorch.py

torch has really a lot of numpy similar functions implemented

https://pytorch.org/docs/stable/generated/torch.cartesian_prod.html#torch.cartesian_prod
https://pytorch.org/docs/stable/generated/torch.minimum.html#torch.minimum

Some GPU functionalities

https://pytorch.org/docs/stable/cuda.html

maybe also usefull to monitor memory usage

https://pytorch.org/docs/stable/notes/cuda.html?highlight=device



#### Optional: compute smallest eigenvalue. 
(but maybe I don't need it, anyway)

There is also the eigenvalue computation. But is there also the sparse compute of only the first eigenvalue ?

#### Course on GPU

http://courses.cms.caltech.edu/cs179/

#### Implementation 
##### GPU the FFT 

##### What about all the logic for the heterogeneity ? 

##### quadratic or linear ? 

##### Load only part of the heterogeneity

##### Workflow:

- make it work on laptop
- execute on AMD node
- execute on gpu node

## On NEMO 

ml devel/cuda/11.3 

Follwing the `Getting started` instructions:


python3 -m pip install --user torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
<!-- #endregion -->

```python

```
