
"""
Check if bicubic interpolation makes some ringing if the work of adhesion
distribution is like a step function.

If not, using bivubic interpolation can make the crackfront model useable on
biphase work of adhesion distribution.

However, this would probably mean that the trust region radius will be as small
as the distance between kollokation points of the bicubic interpolation.

"""
