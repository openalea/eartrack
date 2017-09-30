===========================
Installation with Miniconda
===========================

Miniconda installation
----------------------

Follow official website instruction to install miniconda :

http://conda.pydata.org/miniconda.html

On Linux / Ubuntu / MacOS
-------------------------

Create virtual environment and activate it
..........................................

.. code:: shell

    conda create --name eartrack python
    source activate eartrack

Dependencies install
....................

.. code:: shell

    conda install -c conda-forge numpy matplotlib opencv scikit-image
    conda install -c openalea openalea.deploy openalea.core

(Optional) Package managing tools :

.. code:: shell

    conda install -c conda-forge notebook nose sphinx sphinx_rtd_theme pandoc


Eartrack install
................

.. code:: shell

    conda install -c openalea openalea.eartrack

On Windows
----------

Create virtual environment and activate it
..........................................

.. code:: shell

    conda create --name eartrack python
    activate eartrack

Dependencies install
....................

.. code:: shell

    conda install -c conda-forge numpy matplotlib scikit-image opencv pywin32
    conda install -c openalea openalea.deploy openalea.core

(Optional) Package managing tools :

.. code:: shell

    conda install -c conda-forge notebook nose sphinx sphinx_rtd_theme pandoc


Eartrack install
................

.. code:: shell

    conda install -c openalea openalea.eartrack
