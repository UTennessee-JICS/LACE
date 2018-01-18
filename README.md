# README #

This README would normally document whatever steps are necessary to get your application up and running.

### What is this repository for? ###

* Quick summary
Linear Algebra for Computational Engineering
* Version
0.0
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

### How do I get set up? ###

* Summary of set up
Make from project root for now
* Configuration
* Dependencies
	* Requires Google Test, Google Mock, and Intel's Math Kernel Library (MKL) to be installed and in paths
		1. Google Test and Google Mock can be built from the Google Test public github repository
			* $ cd ${HOME} or anywhere you find convenient
			* $ git clone https://github.com/google/googletest.git 
			* follow instructions from https://github.com/google/googletest/blob/master/googletest/README.md#using-cmake
			* eg.
				* $ export GTEST_DIR=${HOME}/googletest/googletest
				* $ mkdir mybuild
				* $ cd mybuild
				* $ cmake -Dgtest_build_samples=ON ${GTEST_DIR}
				* $ make
				* $ cd ${HOME}/googletest/googlemock
				* $ export GMOCK_DIR=${HOME}/googletest/googlemock
				* $ mkdir mybuild
				* $ cd mybuild
				* $ cmake -Dgtest_build_samples=ON ${GMOCK_DIR}
				* $ make
			* Adding definitions for GTEST_DIR and GMOCK_DIR directories to your shell profile file (.bashrc) will simplify future LACE development. LACES's makefile expects these locations to be defined.  
		2. MKL
			* Register and download MKL from https://software.intel.com/en-us/mkl
			* following installation of MKL enable LACE to access it by sourcing it:
				* source <path-to-installation>/intel/compilers_and_libraries/linux/mkl/bin/mklvars.sh intel64
				* Adding this sourcing to your shell profile file (.bashrc) will simplify future LACE development. LACES's makefile expects locations defined by mklvars.sh, like MKL_ROOT, to be defined.  

* Database configuration
* How to run tests
* Deployment instructions

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Repo owner or admin
* Other community or team contact