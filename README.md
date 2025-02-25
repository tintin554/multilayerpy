# MultilayerPy

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6411188.svg)](https://doi.org/10.5281/zenodo.6411188)

 A package used to build a kinetic multi-layer model for aerosol particles and films. Released under the GNU GPL v3.0 license.

### Contents
- [Overview](#overview)
- [Installation](#installation)
- [Contributing](#contributing)
- [Reference documentation](#reference-documentation)
- [Tutorials](#tutorials) 
- [Acknowledgement and citation](#acknowledgement-and-citation)

## Overview
MultilayerPy is a Python-based framework for constructing, running and optimising kinetic multi-layer models of aerosol particles and films.
In this repository you will find the core MultilayerPy package along with reference documentation (html files) and a directory of tutorial Jupyter notebooks to get you started. 

Here is an [introductory video explaining the package and its installation](https://www.youtube.com/watch?v=3BXoENXfueE). 

Below is the live webinar that walks you through using the package.
[![Here is the live webinar that walks you through using the package.](https://img.youtube.com/vi/6m_v1M60PwQ/hqdefault.jpg)](https://youtu.be/6m_v1M60PwQ)

There is also a [crash course](https://www.youtube.com/watch?v=ErxTOz0NLhw) video which takes the user through the crash course notebook. 

The framework is summarised in the figure below:

![image](summary_fig.png)

MultilayerPy takes advantage of the object-oriented programming (OOP) paradigm and represents the common building blocks (reaction scheme, diffusion regime, model components) as separate classes.

These classes can be used to build the model, the code for which is automatically generated by MultilayerPy. 

The model can be paired with experimental data in order to carry out inverse modelling: where model parameters are adjusted such that the best fit to the data is obtained. This can be done using a local or global optimisation algorithm.

Additionally, Markov Chain Monte Carlo (MCMC) sampling of the model-data system can be carried out using MultilayerPy, incorporating the well-established emcee Python package. 

A more detailed description of how MultilayerPy works is outlined in the reference manual and in the upcoming manuscript (close to submission). 

## Installation
There are currently three ways of installing MultilayerPy:

#### 1) `pip` install
Running `pip install multilayerpy` in your terminal window will download and build MultilayerPy in your python environment, enabling you to import the package without being in the source code directory. This is the most straightforward method. It does not include any of the tutorial notebooks and data, which need to be downloaded separately from the GitHub repository or Zeondo archive. 

Ensure that you have the correct packages installed as listed below in the dependencies. The standard Anaconda python distribution should work fine with this method of installation. 

#### 2) Download the .zip file
You can download this repository as a .zip file from the releases tab in this repository or by clicking on the upper-left "code" button and selecting "download ZIP" for the latest version (the latest version may still be under development). 

#### 3) Clone the Git Hub repository
To keep an up-to-date version on your system, clone the repository using git and the command line:

`$ git clone https://github.com/tintin554/kinetic-multilayer-model-builder.git`

Then a simple `$ git pull` would update your local copy with the latest "nightly" version of the package. 
Be warned that these may not be the final stable versions of the next release of MultilayerPy. Check this by typing `print(multilayerpy.__version__)` into your python terminal.
Download the package from the releases tab for the latest stable version. 

If you want to make changes to the source code and/or submit pull requests, fork this repository and include your own features. This is highly encouraged! (see [Contributing](#contributing))

Unit tests should be run after installing or updating the package using methods 2 & 3. Run the testing.py script from the terminal: 

`$ python -m unittest testing.py -v`

There should be no test failures. Alternatively, the `testing.py` code could be run from an interactive Spyder session. 

### Anaconda Python distribution (Spyder and Jupyter Notebook)

The tutorials for this package are written in Jupyter notebooks and the code was developed using Spyder. Both of these programs are available in the [Anaconda Python distribution](https://www.anaconda.com/products/distribution). It is recommended that the user uses this Python distribution to run MultilayerPy.

### Dependencies
This package requires the standard Anaconda python distribution (developed on Python version `3.8.8`) and the following packages (and versions) are used for development:
- SciPy (1.7.3) 
- NumPy (1.21.5)
- Matplotlib (3.3.4)
- emcee (3.1.1)

Scipy, Numpy and Matplotlib come with the standard Anaconda python distribution. The most straightforward way to install emcee on your system is to type `pip install emcee` into your python terminal. Note that SciPy version >1.7.1 is required for the full functionality of the `optimize` module.
MultilayerPy works on emcee version 3.1.1 onwards (this is monitored). 

## Reference documentation
Open the reference documentation html file in your internet browser. There is a search function which enables the user to search the package for documenation on a specific aspect of the source code. 

## Tutorials
You will find the tutorial Jupyter notebooks in the GitHub repository and the linked Zenodo archive. 

If you have downloaded the package from the Zenodo archive or as a .zip from the GihHub repository, the notebooks should be run in the parent directory containing `multilayerpy`. 

If you have installed the package using `pip` the notebook can be run from anywhere. Note that some notebooks require data available in the GihHub repository and Zenodo archive. 

## Contributing
In order for this project to progress, input from the community is vital. Please use the issues section of this repository to flag any issues. Forking this repository and submitting pull requests for discussion is highly encouraged as this is the best way to develop and improve this package. 

## Acknowledgement and citation
We would greatly appreciate citation of the description paper ([description paper here](https://doi.org/10.5194/gmd-15-7139-2022)) and source code (see beginning of this README) if MultilayerPy was used in your study. 

The citation is:

`A. Milsom, A. Lees, A. M. Squires and C. Pfrang, "MultilayerPy (v1.0): a Python-based framework for building, running and optimising kinetic multi-layer models of aerosols and films", Geosci. Model Dev., 15, 7139–7151, 2022`
