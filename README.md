CrackFront 
===========

This code implements a linearized Crack-Front model for the mechanics of adhesive contacts in Python.
The model includes: 
- linear noncircular perturbation of the energy release rate with respect to the circular JKR solution.
- solving for the equilibrium position of the crack-front on a heterogeneous work of adhesion field. Following resolution algorithms are implemented:
  - a modified trust-region Newton-CG minimization algorithm 
  - the crack-propagation algorithm by Rosso and Krauth, that generates a monotonically increasing (or decreasing) sequence of crack positions 
- computation of the work of adhesion heterogeneity equivalent to surface roughness.


The crack-perturbation model for the adhesion of spheres with work of adhesion heterogeneity is described and validated against the boundary-element method in 
[Sanner, Pastewka, JMPS (2023)](https://www.sciencedirect.com/science/article/pii/S0022509622000059).

The crack-perturbation model for the adhesion of rough spheres is described and validated against the boundary-element method in
[Sanner, Kumar, Jacobs, Dhinojwala,Pastewka, ArXiv (2023)]( 	
https://doi.org/10.48550/arXiv.2307.14233).


The crack perturbation method is based on the first-order perturbation of the stress intensity factor derived by Gao and Rice using weight-function theory. 
- [Gao, Rice, J.Appl.Mech., 54 (1987)](https://asmedigitalcollection.asme.org/appliedmechanics/article/54/3/627/423328/Nearly-Circular-Connections-of-Elastic-Half-Spaces) 
- [J. R. Rice, Weight function theory for three-dimensional elastic crack analysis, Fracture
Mechanics: Perspectives and Directions (Twentieth Symposium), R. Wei, R. Gangloff, eds.
(American Society for Testing and Materials, Philadelphia, USA, 1989), pp. 29–57.](https://www.astm.org/stp18819s.html)

The crack-propagation algorithm by Rosso and Krauth is described in 
- [A. Rosso, W. Krauth, Roughness at the depinning threshold for a long-range elastic string,422
Phys. Rev. E 65, 025101 (2002)](https://doi.org/10.1103/PhysRevE.65.025101)

GPU accelerated
---------------

This code can make use of GPUs accelaration using pytorch if the hardware is available.  

Installation
------------

First install the dependencies listed below, then quick install with: `python3 -m pip install git+https://github.com/ContactEngineering/CrackFront.git`


Dependencies
------------

The package requires :
- **numpy** - https://www.numpy.org/
- **NuMPI** - https://github.com/imtek-simulation/numpi
- **pytorch** - https://gitlab.com/muspectre/muspectre
- **Adhesion** - https://github.com/ContactEngineering/Adhesion and the dependencies of that package


Funding
-------

Development of this project is funded 
by the [Deutsche Forschungsgemeinschaft](https://www.dfg.de/en) within [EXC 2193](https://gepris.dfg.de/gepris/projekt/390951807)
and by the [European Research Council](https://erc.europa.eu) within [Starting Grant 757343](https://cordis.europa.eu/project/id/757343).