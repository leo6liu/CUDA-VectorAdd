// file: header.cuh
// author: Leo Battalora
//

// make sure definitions are only made once
//
#ifndef HEADER_CUH
#define HEADER_CUH

// include files
//
#include <argp.h>
#include <stdio.h>
#include <omp.h>

// define constants
//
#define VECTOR_LENGTH (1 << 28) // 2^28 * sizeof(float) = 1 GiB
 
// function definitions
//
void vec_rand_init(float *vector, int len);
void vec_print(FILE *fs, float *vector, int len);
float vec_error(float *vec_1, float *vec_2, int len);
void vec_add_cpu(float *a, float *b, float *c, int len);
void vec_add_gpu(float *a, float *b, float *c, int len, int n_gpus,
		 int threads, int blocks);

// kernel definitions
//
__global__ void VecAdd(float *a, float *b, int len);

#endif
//
// end of file
