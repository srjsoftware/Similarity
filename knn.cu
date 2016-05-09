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
#include "cuCompactor.cuh"


struct is_bigger_than_threshold
{
	float threshold;
	is_bigger_than_threshold(float thr) : threshold(thr) {};
	__host__ __device__
	bool operator()(const Similarity &reg)
	{
		return (reg.distance > threshold);
	}
};


/*
* We pass the distance function  as a pointer  (*distance)
*/

__host__ int KNN(InvertedIndex inverted_index, vector<Entry> &query, float threshold, Similarity* distances,
	void(*distance)(InvertedIndex, Entry*, int*, Similarity*, int D), struct DeviceVariables *dev_vars, int docid) {

	dim3 grid, threads;
	get_grid_config(grid, threads);

	int num_docs = inverted_index.num_docs;
	int *d_count = dev_vars->d_count, *d_index = dev_vars->d_index, *d_sim = dev_vars->d_sim, *size_doc = dev_vars->d_size_doc;
	int *d_BlocksCount = dev_vars->d_bC, *d_BlocksOffset = dev_vars->d_bO;
	Entry *d_query = dev_vars->d_query;
	Similarity *d_similarity = dev_vars->d_dist, *d_result = dev_vars->d_result;

	gpuAssert(cudaMemcpyAsync(d_query, &query[0], query.size()*sizeof(Entry), cudaMemcpyHostToDevice));
	gpuAssert(cudaMemset(dev_vars->d_sim, 0, num_docs*sizeof(int))); // TODO: usar apenas o necessário

	//get_term_count_and_tf_idf << <grid, threads >> >(inverted_index, d_query, d_count, d_qnorm, d_qnorml1, query.size());
	get_term_count_and_tf_idf << <grid, threads >> >(inverted_index, d_query, d_count, query.size());

	thrust::device_ptr<int> thrust_d_count(d_count);
	thrust::device_ptr<int> thrust_d_index(d_index);
	thrust::inclusive_scan(thrust_d_count, thrust_d_count + query.size(), thrust_d_index);

	jaccardSimilarity(inverted_index, d_query, d_index, d_sim, query.size());
	
	filter_registers<< <grid, threads>> >(d_sim, threshold, query.size(), docid, num_docs, size_doc, d_similarity); //TODO: fazer apenas com o necessário

	int blocksize = 512;
	int numBlocks = cuCompactor::divup(num_docs, blocksize);

	// TODO: fazer apenas com o necessário
	int totalSimilars = cuCompactor::compact2<Similarity>(d_similarity, d_result, num_docs, is_bigger_than_threshold(threshold), blocksize, numBlocks, d_BlocksCount, d_BlocksOffset);

	cudaMemcpy(distances, d_result, sizeof(Similarity)*totalSimilars, cudaMemcpyDeviceToHost);
	
	return totalSimilars;
}

__global__ void get_term_count_and_tf_idf(InvertedIndex inverted_index, Entry *query, int *count, int N) {
	int block_size = N / gridDim.x + (N % gridDim.x == 0 ? 0 : 1);		//Partition size
	int offset = block_size * (blockIdx.x); 				//Beginning of the block
	int lim = min(offset + block_size, N); 					//End of the block
	int size = lim - offset; 						//Block size

	query += offset;
	count += offset;

	for (int i = threadIdx.x; i < size; i += blockDim.x) {
		Entry entry = query[i];

		int idf = inverted_index.d_count[entry.term_id];
		//query[i].tf_idf = entry.tf * log(inverted_index.num_docs / float(max(1, idf)));
		count[i] = idf;
		//atomicAdd(d_qnorm, query[i].tf_idf * query[i].tf_idf);
		//atomicAdd(d_qnorml1, query[i].tf_idf);

	}
}

__global__ void filter_registers(int *sim, float threshold, int querysize, int docid, int N, int *doc_size, Similarity *similars) { // similars + id_doc
	//N -= docid;
	int block_size = N / gridDim.x + (N % gridDim.x == 0 ? 0 : 1);		//Partition size
	int offset = block_size * (blockIdx.x);// + docid; 				//Beginning of the block
	int lim = min(offset + block_size, N);// + docid); 					//End of the block
	int size = lim - offset;

	similars += offset;
	sim += offset;
	doc_size += offset;

	for (int i = threadIdx.x; i < size; i += blockDim.x) {
		float jac = sim[i]/ (float) (querysize + doc_size[i] - sim[i]);

		similars[i].doc_id = offset + i;
		similars[i].distance = jac;
	}
}
