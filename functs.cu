// file: functs.cu
// author: Leo Battalora
//

// include files
//
#include "header.cuh"

// ----------------------------------------------------------------------------
// function: vec_rand_init
//
void
vec_rand_init (float *vector, int len)
{
  for (int i = 0; i < len; i++) {
    vector[i] = (float)rand() / (float)RAND_MAX;
  }
}

// ----------------------------------------------------------------------------
// function: vec_print
//
void
vec_print (FILE *fs, float *vector, int len)
{
  for (int i = 0; i < len; i++) {
    fprintf(fs, "%3d: %f\n", i, vector[i]);
  }
}

// ----------------------------------------------------------------------------
// function: vec_error
//
float
vec_error (float *vec_1, float *vec_2, int len)
{
  float error = 0.0;
  for (int i = 0; i < len; i++) {
    error += fabsf(vec_1[i] - vec_2[i]) / vec_1[i];
  }
  return (error);
}

// ----------------------------------------------------------------------------
// function: vec_add_cpu
//
void
vec_add_cpu (float *a, float *b, float *c, int len)
{
  // create timer
  //
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // start timer
  //
  cudaEventRecord(start, 0);

  // calculate entrywise sum
  //
  for (int i = 0; i < len; i++) {
    c[i] = a[i] + b[i];
  }

  // stop timer and record compute time
  //
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop); // wait for stop event to complete
  float compute_time;
  cudaEventElapsedTime(&compute_time, start, stop);

  // destroy timer
  //
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  
  // print CPU compute time to stdout
  //
  printf("INFO: CPU compute time: %.3f ms\n", compute_time);
}

// ----------------------------------------------------------------------------
// function: vec_add_gpu
//
void
vec_add_gpu (float *h_a, float *h_b, float *h_c, int len, int n_gpus)
{
  // declare variables and cuda events for timer
  //
  cudaEvent_t start, stop;
  float in_time, compute_time, out_time;

  // split into n_gpus number of threads
  //
  int cpu_thread_id;
  omp_set_num_threads(n_gpus);
#pragma omp parallel private(cpu_thread_id)
  {
    // get cpu_thread_id and set CUDA device accordingly
    //
    cpu_thread_id = omp_get_thread_num();
    cudaSetDevice(cpu_thread_id);
    
    // get device properties
    //
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cpu_thread_id);
    printf("INFO: Device %d name: %s\n", cpu_thread_id, prop.name);

    // calculate length of vector for each gpu
    //
    int default_gpu_len = (int)ceil((double)len / (double)n_gpus);
    int gpu_len = default_gpu_len;
    if ((gpu_len * (cpu_thread_id + 1)) > len) {
      gpu_len = gpu_len - ((gpu_len * (cpu_thread_id + 1) - len)) + 1;
    }
    
    // check if two float vectors of length gpu_len will fit on the GPU
    //
    if (prop.totalGlobalMem < gpu_len * sizeof(float) * 2) {
      printf("ERROR: Insufficient GPU memory for calculation\n");
      printf("STATUS: Expect undefined behavior...\n");
    }

    // calculate number of threads per block and number of blocks
    //
    int threads = prop.maxThreadsPerBlock;
    int blocks;
    if (ceil((double)gpu_len / (double)threads) > prop.maxGridSize[0]) {
      blocks = prop.maxGridSize[0];
    } else {
      blocks = (int)ceil((float)gpu_len / threads);
    }
    
#pragma omp barrier // sync threads before starting in_time timer
    
    // create and start timer for in_time
    //
    if (cpu_thread_id == 0) {
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start, 0);
    }

    // declare device vectors
    //
    float *d_a; // output will overwrite d_a
    float *d_b;
    
    // allocate vectors in device memory
    //
    cudaMalloc((void **)&d_a, gpu_len * sizeof(float));
    cudaMalloc((void **)&d_b, gpu_len * sizeof(float));
    
    // copy host input vectors to device
    //
    cudaMemcpy(d_a, h_a + (default_gpu_len * cpu_thread_id), 
	       gpu_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b + (default_gpu_len * cpu_thread_id), 
	       gpu_len * sizeof(float), cudaMemcpyHostToDevice);
    
#pragma omp barrier // sync threads before stopping in_time timer

    // stop timer and record in_time
    //
    if (cpu_thread_id == 0) {
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&in_time, start, stop);
    }

    // start timer for compute_time
    //
    if (cpu_thread_id == 0) {
      cudaEventRecord(start, 0);
    }

    // execute VecAdd kernel
    //
    VecAdd<<<blocks, threads>>>(d_a, d_b, gpu_len);
  
#pragma omp barrier // sync threads before stopping compute_time timer

    // stop timer and record compute_time
    //
    if (cpu_thread_id == 0) {
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&compute_time, start, stop);
    }

    // start timer for out_time
    //
    if (cpu_thread_id == 0) {
      cudaEventRecord(start, 0);
    }

    // copy device output vector to host
    //
    cudaMemcpy(h_c + (default_gpu_len * cpu_thread_id), d_a, 
	       gpu_len * sizeof(float), cudaMemcpyDeviceToHost);
    
    // free device vectors
    //
    cudaFree(d_a);
    cudaFree(d_b);

#pragma omp barrier // sync threads before stopping out_time timer

    // stop timer, record out_time, and destroy timer
    //
    if (cpu_thread_id == 0) {
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&out_time, start, stop);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
    }  
  }
  
  // print in, compute, and out times
  //
  printf("INFO: GPU in time: %.3f ms\n", in_time);
  printf("INFO: GPU compute time: %.3f ms\n", compute_time);
  printf("INFO: GPU out time: %.3f ms\n", out_time);
  printf("INFO: Total GPU time: %.3f ms\n", in_time + compute_time + 
	 out_time);
}

//
// end of file
