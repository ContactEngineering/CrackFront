
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
 