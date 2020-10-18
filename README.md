# CUDA VectorAdd

An example program which detects number of CUDA compute-capable devices,
generates two float vectors of length 2^28 with values[0,1], adds them using
the available CUDA compute-capable devices, and compares the result with a CPU
solution by calculating the total error.

	 $ ./add --help
	 CUDA VectorAdd -- a program which scales across multiple GPUs to perform vector
	 addition

	   -b, --blocks-per-gpu=LEN   Specify the number of blocks per GPU (defaults to
	                              either the the number of elements each GPU needs
				      to process divided by the threads per block or the
				      GPU's maxGridSize[0])
	   -N, --vector-length=LEN    Specify the vector lengths (defaults to 2^28)
	   -t, --threads-per-block=LEN   Specify the number of threads per GPU block
                                         (defaults to maxThreadsPerBlock of GPU)
	   -v, --verbose              Explains what is being done
	   -?, --help                 Give this help list
	       --usage                Give a short usage message
	   -V, --version              Print program version
	 Mandatory or optional arguments to long options are also mandatory or optional
	 for any corresponding short options.
	 
	 Report bugs to <leo6@temple.edu>.
