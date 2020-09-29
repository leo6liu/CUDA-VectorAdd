//   _____
//  /     \             file: main.cu
//  vvvvvvv  /|__/|   author: Leo Battalora
//     I   /O,O   |
//     I /_____   |      /|/|
//    J|/^ ^ ^ \  |    /00  |    _//|
//     |^ ^ ^ ^ |W|   |/^^\ |   /oo |
//      \m___m__|_|    \m_m_|   \mm_|
//
// An example program which detects number of CUDA compute-capable devices,
// generates two float vectors of length 2^28 with values[0,1], adds them using
// the available CUDA compute-capable devices, and compares the result with a
// CPU solution by calculating the total error. 
//

// include files
//
#include "header.cuh"

// ----------------------------------------------------------------------------
// function: main
//
int
main (int argc, char **argv)
{
  // get number of CUDA compute-capable devices
  //
  int n_gpus;
  cudaGetDeviceCount(&n_gpus);
  printf("INFO: %d CUDA compute-capable devices detected.\n", n_gpus);
  
  // exit program if no CUDA compute-capable devices found
  //
  if (n_gpus == 0) {
    printf("STATUS: Exiting progam... (no CUDA-enabled devices found)\n");
    return (0);
  }

  // print length and type of vectors
  //
  printf("INFO: This program adds two float vectors (A + B = C) of length %d.\n", 
	 VECTOR_LENGTH);

  // declare and allocate host vectors for a + b = c
  //
  float *a = (float *)malloc(VECTOR_LENGTH * sizeof(float));
  float *b = (float *)malloc(VECTOR_LENGTH * sizeof(float));
  float *c_cpu = (float *)malloc(VECTOR_LENGTH * sizeof(float));
  float *c_gpu = (float *)malloc(VECTOR_LENGTH * sizeof(float));

  // use current time as seed for random generator
  //
  srand(time(NULL));

  // initialize values for a and b
  //
  printf("STATUS: Initializing values for vector A...\n");
  vec_rand_init(a, VECTOR_LENGTH);
  printf("STATUS: Initializing values for vector B...\n");
  vec_rand_init(b, VECTOR_LENGTH);

  // print first five values of a and b
  //
  printf("DEBUG: Vector A:\n");
  vec_print(stdout, a, 3);
  printf("   ...\n");
  printf("DEBUG: Vector B:\n");
  vec_print(stdout, b, 3);
  printf("   ...\n");
  
  // calculate entrywise sum on GPU(s)
  //
  printf("STATUS: Calculating entrywise sum on GPU...\n");
  vec_add_gpu(a, b, c_gpu, VECTOR_LENGTH, n_gpus);

  // calcuate entrywise sum on CPU
  //
  printf("STATUS: Calculating entrywise sum on CPU...\n");
  vec_add_cpu(a, b, c_cpu, VECTOR_LENGTH);
  
  // print first five values of c_gpu
  //
  printf("DEBUG: Vector C (GPU):\n");
  vec_print(stdout, c_gpu, 3);
  printf("   ...\n");

  // print first five values of c_cpu
  //
  printf("DEBUG: Vector C (CPU):\n");
  vec_print(stdout, c_cpu, 3);
  printf("   ...\n");
  
  // calculate error between CPU and GPU results
  //
  printf("STATUS: Calculating total error between CPU and GPU results...\n");
  float error = vec_error(c_cpu, c_gpu, VECTOR_LENGTH);
  printf("INFO: Total error: %f\n", error);

  // deallocate host vectors
  //
  free(a);
  free(b);
  free(c_cpu);
  free(c_gpu);
  
  // exit normally
  //
  return (0);
}
