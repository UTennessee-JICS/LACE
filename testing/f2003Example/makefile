

all: f2003C++malloc

f2003C++malloc: c_sub.c fort_c.f
	g++ -std=gnu++11 -Wall -g -c c_sub.c
	gfortran -g -Wall -ffree-form -lstdc++ -o f2003C++malloc fort_c.f c_sub.o
