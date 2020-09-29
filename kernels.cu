// file: kernels.cu
// author: Leo Battalora
//

// ----------------------------------------------------------------------------
// kernel: VecAdd
//
__global__ void
VecAdd (float *a, float *b, int len)
{
  // calculate vector index
  //
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  // keep striding over grids until all indices for this thread have been 
  // calculated 
  while (i < len) {
    // perform addition and save result in a[i]
    //
    a[i] = a[i] + b[i];

    i += gridDim.x * blockDim.x; // grid stride
  }
}

//
// end of file
