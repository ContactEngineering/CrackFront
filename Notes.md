
# Progress: 

- how to generate the regularise bimodal distribution ? 
    - true bimodal and then smoothen with bicubic ? 
    - apply a smoothed step function on the gaussian random field instead of
      the true step function ?
      
- figure out the discretisation needs of the bicubic interpolation

- smarter trust region radius: 
   - for bad ininitial guesses of the crack front position, the crack has to
     to move in steps as high as the heterogeneity, despite the gradioent is
     dominated by the elastic term. I think that in this situation, the 
     trustregion radius can be safely increased adaptively, so I should
     probably implement that.
     
## Evaluation of the random field

for a sparse collection nonzero wavevectors. It might be more efficient to do 
directly brute force fourier interpolation. 

### GPU

Pytorch examples 

https://github.com/ritchieng/eigenvectors-from-eigenvalues

https://www.youtube.com/playlist?list=PLlMkM4tgfjnJ3I-dbhO9JTw7gNty6o_2m

https://github.com/dionhaefner/pyhpc-benchmarks/blob/master/benchmarks/isoneutral_mixing/isoneutral_pytorch.py

torch has really a lot of numpy similar functions implemented

https://pytorch.org/docs/stable/generated/torch.cartesian_prod.html#torch.cartesian_prod
https://pytorch.org/docs/stable/generated/torch.minimum.html#torch.minimum

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