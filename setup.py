
import pathlib
from setuptools import setup


# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="multilayerpy",
    version="1.0.1",
    description="Build, run and optimise kinetic multi-layer models of aerosols and films",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/tintin554/multilayerpy",
    author="Adam Milsom",
    author_email="a.milsom.2@bham.ac.uk",
    license="GPL3.0",
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["multilayerpy"],
    install_requires=["scipy>=1.7.1",
                      "numpy",
                      "matplotlib",
                      "emcee"],
    
)

