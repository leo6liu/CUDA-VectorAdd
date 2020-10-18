# CUDA VectorAdd

An example program which detects number of CUDA compute-capable devices,
generates two float vectors of length 2^28 with values[0,1], adds them using
the available CUDA compute-capable devices, and compares the result with a CPU
solution by calculating the total error.

	$ ./add --help
	Usage: add [OPTION...]
	CUDA VectorAdd -- a program which scales across multiple GPUs to perform vector
	addition

	  -t, --threads-per-block=DIM   Specify the number of threads per GPU block
	                                (defaults to 512)
	  -v, --verbose              Explains what is being done
	  -?, --help                 Give this help list
	      --usage                Give a short usage message
	  -V, --version              Print program version

	Mandatory or optional arguments to long options are also mandatory or optional
	for any corresponding short options.

	Report bugs to <leo6@temple.edu>.
