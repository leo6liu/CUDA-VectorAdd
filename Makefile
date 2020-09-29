# file: Makefile
# author: Leo Battalora
#

all: add

add: Makefile main.cu functs.cu kernels.cu header.cuh
	nvcc -o add main.cu functs.cu kernels.cu \
		--compiler-options -fopenmp

clean:
	rm -f add

#
# end of file
