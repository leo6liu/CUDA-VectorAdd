# CUDA VectorAdd

An example program which detects number of CUDA compute-capable devices,
generates two float vectors of length 2^28 with values[0,1], adds them using
the available CUDA compute-capable devices, and compares the result with a CPU
solution by calculating the total error.