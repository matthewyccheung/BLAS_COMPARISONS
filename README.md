# Abstract

Five different algorithms for matrix multiplication was tested and compared with Floating Point Operations Per Second (FLOPS) for square matrices. These algorithms are Triple Loop, Dot Product (BLAS-1), Saxpy (BLAS-1), Matrix-Vector (BLAS-2) and Outer-Product (BLAS-2). The size of the matrix was varied and was plotted versus MFLOPS. In this case, 10 values from 10 to 500 were selected. From the collected data, the MFLOPS ranking was matrix-vector, outer product, dot product, saxpy and triple loop. The rankings were expected because BLAS-2 were theorized to be faster than BLAS-1 and both were theorzied to be faster than the triple loop due to the number of FLOPS/MemoryReferences. The curves plotted were consistent with the expected shape - as the size of the matrix gets larger and larger, the MFLOPS eventually plateaus. This will be more evident as the size of the matrix gets larger and larger.

# Repository Contents

The repository contains a Jupyter Notebook Report of the background information and numerical experiments. For more information about BLAS kernel implementations, see http://www.netlib.org/blas/
