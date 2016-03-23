/*********************************************************************
11
12	 Copyright (C) 2015 by Wisllay Vitrio
13
14	 This program is free software; you can redistribute it and/or modify
15	 it under the terms of the GNU General Public License as published by
16	 the Free Software Foundation; either version 2 of the License, or
17	 (at your option) any later version.
18
19	 This program is distributed in the hope that it will be useful,
20	 but WITHOUT ANY WARRANTY; without even the implied warranty of
21	 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
22	 GNU General Public License for more details.
23
24	 You should have received a copy of the GNU General Public License
25	 along with this program; if not, write to the Free Software
26	 Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
27
28	 ********************************************************************/

/* *
 * knn.cu
 */

#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <queue>
#include <vector>
#include <set>
#include <functional>


#include "knn.cuh"
#include "structs.cuh"
#include "utils.cuh"
#include "inverted_index.cuh"
#include "cuda_distances.cuh"

/*
* We pass the distance function  as a pointer  (*distance)
*/

__host__ Similarity* KNN(InvertedIndex inverted_index, vector<Entry> &query, float threshold, Similarity* distances,
	void(*distance)(InvertedIndex, Entry*, int*, Similarity*, int D), struct DeviceVariables *dev_vars, int docid) {
	
	int stream_id = omp_get_num_threads();

	dim3 grid, threads;
	get_grid_config(grid, threads);

	int *d_count = dev_vars->d_count, *d_index = dev_vars->d_index;
	Entry *d_query = dev_vars->d_query;
	Similarity *d_dist = dev_vars->d_dist;
	float *d_qnorm = &dev_vars->d_qnorms[0], *d_qnorml1 = &dev_vars->d_qnorms[1];

	gpuAssert(cudaMemcpyAsync(d_query, &query[0], query.size() * sizeof(Entry), cudaMemcpyHostToDevice));
	gpuAssert(cudaMemset(dev_vars->d_qnorms, 0, 2 * sizeof(float)));

	get_term_count_and_tf_idf << <grid, threads >> >(inverted_index, d_query, d_count, d_qnorm, d_qnorml1, query.size());

	thrust::device_ptr<int> thrust_d_count(d_count);
	thrust::device_ptr<int> thrust_d_index(d_index);
	thrust::inclusive_scan(thrust_d_count, thrust_d_count + query.size(), thrust_d_index);

	distance(inverted_index, d_query, d_index, d_dist, query.size());
	
	gpuAssert(cudaMemcpyAsync(distances, d_dist, inverted_index.num_docs * sizeof(Similarity), cudaMemcpyDeviceToHost));
	
	return distances;
}

__global__ void get_term_count_and_tf_idf(InvertedIndex inverted_index, Entry *query, int *count, float *d_qnorm, float *d_qnorml1, int N) {
	int block_size = N / gridDim.x + (N % gridDim.x == 0 ? 0 : 1);		//Partition size
	int offset = block_size * (blockIdx.x); 				//Beginning of the block
	int lim = min(offset + block_size, N); 					//End of the block
	int size = lim - offset; 						//Block size

	query += offset;
	count += offset;

	for (int i = threadIdx.x; i < size; i += blockDim.x) {
		Entry entry = query[i];

		int idf = inverted_index.d_count[entry.term_id];
		query[i].tf_idf = entry.tf * log(inverted_index.num_docs / float(max(1, idf)));
		count[i] = idf;
		atomicAdd(d_qnorm, query[i].tf_idf * query[i].tf_idf);
		atomicAdd(d_qnorml1, query[i].tf_idf);

	}
}
