OMP= -Xcompiler -fopenmp

Release: main.o simjoin.o  inverted_index.o  cuda_distances.o utils.o
	nvcc -arch=sm_20 -O3 -lgomp   main.o simjoin.o inverted_index.o  cuda_distances.o  utils.o -o sim

main.o: main.cu cuda_distances.cuh  simjoin.cuh  inverted_index.cuh utils.cuh structs.cuh
	nvcc -arch=sm_20 -O3 $(OMP)  -c main.cu
	
simjoin.o: simjoin.cu cuda_distances.cuh  simjoin.cuh  inverted_index.cuh utils.cuh structs.cuh 
	nvcc -arch=sm_20 -O3  $(OMP) -c simjoin.cu

inverted_index.o: inverted_index.cu inverted_index.cuh utils.cuh 
	nvcc -arch=sm_20 -O3  $(OMP) -c inverted_index.cu

cuda_distances.o: cuda_distances.cu cuda_distances.cuh 
	nvcc -arch=sm_20 -O3  $(OMP) -c cuda_distances.cu

utils.o: utils.cu utils.cuh 
	nvcc -arch=sm_20 -O3 $(OMP)  -c utils.cu

clean:
	rm *.o sim
