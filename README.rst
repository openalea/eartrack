.. image:: https://readthedocs.org/projects/eartrack/badge/?version=latest
    :target: http://eartrack.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://travis-ci.org/openalea/eartrack.svg?branch=master
    :target: https://travis-ci.org/openalea/eartrack
    :alt: Travis build status (osx and linux)

.. image:: https://ci.appveyor.com/api/projects/status/bpbmurhqv10pcy0j/branch/master?svg=true
    :target: https://ci.appveyor.com/project/artzet-s/eartrack-xo7du
    :alt: Appveyor build status (Windows x86 and x64)
    
.. image:: https://anaconda.org/openalea/openalea.eartrack/badges/version.svg   
    :target: https://anaconda.org/openalea/openalea.eartrack

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1002155.svg
   :target: https://doi.org/10.5281/zenodo.1002155

.. image:: https://anaconda.org/openalea/openalea.eartrack/badges/license.svg
    :target: https://anaconda.org/openalea/openalea.eartrack
    
========
EarTrack
========

An imaging library to detect and track future position of ear on maize plants.

**EarTrack** is released under a `Cecill-C <http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html>`_ license.


.. contents::

=============
Documentation
=============

The documentation is available at `<https://eartrack.readthedocs.io>`_

Tutorials are available as Jupyter notebook :

* `Getting started with eartrack <http://nbviewer.jupyter.org/github/openalea/eartrack/blob/master/example/getting_started_with_eartrack.ipynb>`_

* `Eartrack step by step <http://nbviewer.jupyter.org/github/openalea/eartrack/blob/master/example/eartrack_step_by_step.ipynb>`_

===========================
Installation with Miniconda
===========================

Miniconda installation
----------------------

Follow official website instruction to install miniconda : http://conda.pydata.org/miniconda.html

To create conda `environment <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_  and activate it :

.. code:: shell 
    
    conda create -n eartrack python=2.7
    source activate eartrack # just "activate eartrack" on windows

User installation
-----------------

Available for linux, Windows and Mac OSX on 64 bits.

.. code:: shell

    conda install -c conda-forge -c openalea openalea.eartrack


Source code installation
------------------------

Please follow documentation installation page `<https://eartrack.readthedocs
.io/en/latest/source/install/index.html>`_.

Help and Support
----------------

Please open an **Issue** if you need support or that you run into any error (Installation, Runtime, etc.). 
We'll try to resolve it as soon as possible.

Citation
--------

Brichet N, Fournier C, Turc O, Strauss O, Artzet S, Pradal C, Welcker C, Tardieu F, Cabrera-Bosquet L. 2017.__
A robot-assisted imaging pipeline for tracking the growths of maize ear and silks in a high-throughput phenotyping platform.__
Plant Methods 13:96 `doi:10.1186/s13007-017-0246-7 <https://doi.org/10.1186/s13007-017-0246-7>`_

Authors
-------

.. include:: AUTHORS.rst
* Nicolas Brichet <nicolas.brichet@inra.fr>
* Christian Fournier <christian.fournier@inra.fr>
* Simon Artzet <simon.artzet@gmail.com>
* Christophe Pradal <christophe.pradal@inria.fr>
