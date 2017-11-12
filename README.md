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
Requires Google Test, Google Mock, and MKL to be installed an in paths
$ cd ${HOME}
$ git clone https://github.com/google/googletest.git 
follow instructions from https://github.com/google/googletest/blob/master/googletest/README.md#using-cmake
eg.
$ export GTEST_DIR=${HOME}/googletest/googletest
$ mkdir mybuild
$ cd mybuild
$ cmake -Dgtest_build_samples=ON ${GTEST_DIR}
$ make
$ cd ${HOME}/googletest/googlemock
$ mkdir mybuild
$ cd mybuild
$ cmake -Dgtest_build_samples=ON ${GMOCK_DIR}
$ make
Adding definitions for GTEST_DIR and GMOCK dir to your shell profile file (.bashrc) will simplify future LACE development.  
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