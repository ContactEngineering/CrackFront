Installation
============

You need Python 3 and FFTW3_ to run ContactMechanics. All Python dependencies can be installed
automatically by invoking

Direct installation with pip
----------------------------

ContactMechanics can be installed by invoking

.. code-block:: bash

    python3 -m pip  install [--user] git+https://github.com/ComputationalMechanics/CrackFront.git

The command will install other dependencies including SurfaceTopogaraphy_, Contactmechanics_, Adhesion_, muFFT_, NuMPI_-

Note that these packages depend on OpenBlas_ , Lapack_ and FFTW3_. Make sure to follow the installation instructions.

Installation from source directory
----------------------------------

If you cloned the repository. You can install the dependencies with

.. code-block:: bash

    python3 -m pip install -r requirements.txt

in the source directory. ContactMechanics can be installed by invoking

.. code-block:: bash

   python3 -m pip install [--user] .

or

.. code-block:: bash

   python3 setup.py install [--user]

in the source directoy. The command line parameter `--user` is optional and leads to a local installation in the current user's `$HOME/.local` directory.



.. _Adhesion:https://github.com/ComputationalMechanics/Adhesion.git
.. _SurfaceTopogaraphy: https://github.com/ComputationalMechanics/SurfaceTopography.git
.. _Contactmechanics: https://github.com/ComputationalMechanics/ContactMechanics.git
.. _FFTW3: http://www.fftw.org/
.. _muFFT: https://gitlab.com/muspectre/muspectre.git
.. _nuMPI: https://github.com/IMTEK-Simulation/NuMPI.git
.. _runtests: https://github.com/bccp/runtests
.. _Homebrew: https://brew.sh/
.. _OpenBLAS: https://www.openblas.net/
.. _LAPACK: http://www.netlib.org/lapack/