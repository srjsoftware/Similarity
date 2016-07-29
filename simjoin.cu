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

#include "simjoin.cuh"
#include "structs.cuh"
#include "utils.cuh"
#include "inverted_index.cuh"
#include "cuCompactor.cuh"


struct is_bigger_than_threshold
{
	float threshold;
	is_bigger_than_threshold(float thr) : threshold(thr) {};
	__host__ __device__
	bool operator()(const Similarity &reg)
	{
		return (reg.similarity > threshold);
	}
};

struct bigger_than_zero
{
  __host__ __device__
    bool operator()(Similarity &reg)
    {
        return reg.similarity;
    }
};

__host__ int findSimilars(InvertedIndex inverted_index, float threshold, struct DeviceVariables *dev_vars, Similarity* h_result,
		int docid, int queryqtt) {

	dim3 grid, threads;
	get_grid_config(grid, threads);

	int num_docs = inverted_index.num_docs;
	int *docsizes = dev_vars->d_docsizes, *docstarts = dev_vars->d_docstarts;
	int *d_BlocksCount = dev_vars->d_bC, *d_BlocksOffset = dev_vars->d_bO; // arrays used for compression
	Entry *d_query = inverted_index.d_entries;
	Similarity *d_candidates = dev_vars->d_candidates, *d_result = dev_vars->d_result;

	gpuAssert(cudaMemset(d_candidates, 0, queryqtt*num_docs*sizeof(Similarity)));

	candidateFiltering<<<grid, threads>>>(inverted_index, d_query, d_candidates, docid, queryqtt, docstarts, docsizes, threshold);

	int blocksize = 1024;
	int numBlocks = cuCompactor::divup(500*num_docs, blocksize);
	int totalSimilars = cuCompactor::compact2<Similarity>(d_candidates, d_result, 500*num_docs, bigger_than_zero(), blocksize, numBlocks, d_BlocksCount, d_BlocksOffset);

	verify_candidates<<<grid, threads>>>(d_query, docstarts, docsizes, d_result, totalSimilars, threshold);

	if (totalSimilars) cudaMemcpyAsync(h_result, d_result, sizeof(Similarity)*totalSimilars, cudaMemcpyDeviceToHost);

	return totalSimilars;
}

/*__host__ int findSimilars3(InvertedIndex inverted_index, float threshold, struct DeviceVariables *dev_vars, Similarity* h_result,
		int docid, int queryqtt) {
	dim3 grid, threads;
	get_grid_config(grid, threads);

	int num_docs = inverted_index.num_docs;
	int *docsizes = dev_vars->d_docsizes, *docstarts = dev_vars->d_docstarts;
	Entry *d_query = inverted_index.d_entries;
	Similarity *d_result = dev_vars->d_result;

	gpuAssert(cudaMemset(d_candidates, 0, queryqtt*num_docs*sizeof(int)));

	//candidateFiltering<<<grid, threads>>>(inverted_index, d_query, d_candidates, docid, queryqtt, docstarts, docsizes, threshold);

	//verify_candidates<<<grid, threads>>>(d_query, docstarts, docsizes, d_result, totalSimilars, threshold);

	if (totalSimilars) cudaMemcpyAsync(h_result, d_result, sizeof(Similarity)*totalSimilars, cudaMemcpyDeviceToHost);

	return totalSimilars;
}*/

__global__ void candidateFiltering(InvertedIndex inverted_index, Entry *d_query, Similarity *d_candidates, int begin, int query_qtt,
		int *docstart, int *docsizes, float threshold) {

	int block_start, block_end, docid, size, maxsize, midprefix;

	for (int q = 0; q < query_qtt && q < inverted_index.num_docs - 1; q++) { // percorre as queries

		docid = begin + q;
		size = docsizes[docid];
		maxsize = ceil(((float) size)/threshold);
		midprefix = size - ceil((threshold*((float) 2*size)) / (1.0 + threshold)) + 1;//mudar pra maxprefix

		for (int idx = blockIdx.x; idx < midprefix; idx += gridDim.x) { // percorre os termos da query (apenas os que estão no midprefix)

			Entry entry = d_query[idx + docstart[docid]]; // find the term

			block_start = entry.term_id == 0 ? 0 : inverted_index.d_index[entry.term_id-1];
			block_end = inverted_index.d_index[entry.term_id];

			for (int i = block_start + threadIdx.x; i < block_end; i += blockDim.x) { // percorre os documentos que tem aquele termo

				Entry index_entry = inverted_index.d_inverted_index[i]; // obter item
				int entry_size = docsizes[index_entry.doc_id];
				Similarity *sim = &d_candidates[q*inverted_index.num_docs + index_entry.doc_id];

				// somar na distância
				if (index_entry.doc_id > docid && entry_size <= maxsize) {
					int minoverlap = (threshold * ((float ) (entry_size + size)))/(1 + threshold);
					int rem = sim->similarity + entry_size - index_entry.pos; // # of features remaning
					if (rem < minoverlap || sim->similarity + docsizes[docid] - entry.pos < minoverlap) {
						sim->similarity = -1000000;
					} else {
						atomicAdd(&sim->similarity, 1);
						//if (!sim->doc_j) {
							sim->doc_i = docid;
							sim->doc_j = index_entry.doc_id;
						//}
						atomicMax(&sim->ft_i, entry.pos);
						atomicMax(&sim->ft_j, index_entry.pos);
					}
				}
			}
		}
	}
}

__global__ void verify_candidates(Entry *d_query, int *doc_start, int *size_docs, Similarity *d_similarity, int candidates_num, float threshold) {
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int totalThreads = blockDim.x*gridDim.x;

	for (; i < candidates_num; i += totalThreads) {
		Similarity *sim = &d_similarity[i];

		Entry *query = d_query + doc_start[sim->doc_i];
		int querysize = size_docs[sim->doc_i];

		Entry *candidate = d_query + doc_start[sim->doc_j];
		int candidatesize = size_docs[sim->doc_j];

		int minoverlap = (threshold * ((float) (candidatesize + querysize)) / (1 + threshold));
		int j = sim->ft_i + 1; // last feature verified in the query
		int k = sim->ft_j + 1; // last feature verified in the candidate

		while (j < querysize && k < candidatesize && candidatesize + sim->similarity - k >= minoverlap && querysize + sim->similarity - j >= minoverlap) { // TODO: parar quando ver que não dá mais
			if (candidate[k].term_id == query[j].term_id) {
				sim->similarity += 1;
				k++;
				j++;
			} else if (candidate[k].term_id < query[j].term_id) {
				k++;
			} else {
				j++;
			}
		}
		//printf("(%d, %d): inter: %f size1: %d size2: %d   fti: %d ftj: %d\n", d_query[0].doc_id, id, d_similarity[i].distance, querysize, size, ft_i[id], ft_j[id]);
		d_similarity[i].similarity = sim->similarity / ((float) (querysize + candidatesize) - sim->similarity);
	}
}

__host__ int findSimilars2(InvertedIndex inverted_index, float threshold, struct DeviceVariables *dev_vars, Similarity* h_result,
		int docid, int queryqtt) {

	dim3 grid, threads;
	get_grid_config(grid, threads);

	int num_docs = inverted_index.num_docs;
	int *docsizes = dev_vars->d_docsizes, *docstarts = dev_vars->d_docstarts;
	int *d_BlocksCount = dev_vars->d_bC, *d_BlocksOffset = dev_vars->d_bO;
	Entry *d_query = inverted_index.d_entries;
	Similarity *d_similarity = dev_vars->d_candidates, *d_result = dev_vars->d_result;

	gpuAssert(cudaMemset(d_similarity, 0, queryqtt*num_docs*sizeof(Similarity)));

	calculateIntersection<<<grid, threads>>>(inverted_index, d_query, d_similarity, docid, queryqtt, docstarts, docsizes, threshold);

	int blocksize = 1024;
	int numBlocks = cuCompactor::divup(num_docs, blocksize);
	int totalSimilars = cuCompactor::compact2<Similarity>(d_similarity, d_result, queryqtt*num_docs, bigger_than_zero(), blocksize, numBlocks, d_BlocksCount, d_BlocksOffset);

	if (totalSimilars) cudaMemcpyAsync(h_result, d_result, sizeof(Similarity)*totalSimilars, cudaMemcpyDeviceToHost);

	return totalSimilars;
}

__global__ void calculateIntersection(InvertedIndex inverted_index, Entry *d_query, Similarity *d_candidates, int begin, int query_qtt,
		int *docstart, int *docsizes, float threshold) {

	int block_start, block_end, docid, size, maxsize;

	for (int q = 0; q < query_qtt && q < inverted_index.num_docs - 1; q++) { // percorre as queries

		docid = begin + q;
		size = docsizes[docid];
		maxsize = ceil(((float) size)/threshold);
		//midprefix = size - ceil((threshold*((float) 2*size)) / (1.0 + threshold)) + 1;//mudar pra maxprefix

		for (int idx = blockIdx.x; idx < size; idx += gridDim.x) { // percorre os termos da query (apenas os que estão no midprefix)

			Entry entry = d_query[idx + docstart[docid]]; // find the term

			block_start = entry.term_id == 0 ? 0 : inverted_index.d_index[entry.term_id-1];
			block_end = inverted_index.d_index[entry.term_id];

			for (int i = block_start + threadIdx.x; i < block_end; i += blockDim.x) { // percorre os documentos que tem aquele termo

				Entry index_entry = inverted_index.d_inverted_index[i]; // obter item
				int entry_size = docsizes[index_entry.doc_id];
				Similarity *sim = &d_candidates[q*inverted_index.num_docs + index_entry.doc_id];

				// somar na distância
				if (index_entry.doc_id > docid && entry_size <= maxsize && sim->similarity > -1) {
					int minoverlap = (threshold * ((float ) (entry_size + size)))/(1 + threshold);
					int rem = sim->similarity + entry_size - index_entry.pos; // # of features remaning
					if (rem < minoverlap || sim->similarity + docsizes[docid] - entry.pos < minoverlap) {
						sim->similarity = -1000000;
					} else {
						atomicAdd(&sim->similarity, 1);
						if (!sim->doc_j) {
							sim->doc_i = docid;
							sim->doc_j = index_entry.doc_id;
						}
					}
				}
			}
		}
	}
}

__global__ void verify_candidates2(Entry *d_query, int querysize, int *doc_start, int *size_docs, Entry *entries, Similarity *d_similarity, int candidates_num, float threshold, int *ft_i, int *ft_j) {
	int i = threadIdx.x + blockIdx.x*gridDim.x;
	int totalThreads = blockDim.x*blockDim.x;

	for (; i < candidates_num; i += totalThreads) {
		int id = d_similarity[i].doc_j;
		//d_candidates[i].distance = 0;
		Entry *candidate = entries + doc_start[id];
		int size = size_docs[id];
		int minoverlap = (threshold * ((float) (size + querysize)) / (1 + threshold));

		for (int j = ft_j[id] + 1, k = ft_i[id] + 1; j < querysize && k < size && size + d_similarity[i].similarity - k >= minoverlap &&
			querysize +  d_similarity[i].similarity - j >= minoverlap;) { // TODO: parar quando ver que não dá mais
			if (candidate[k].term_id == d_query[j].term_id) {
				d_similarity[i].similarity += 1;
				k++;
				j++;
			} else if (candidate[k].term_id < d_query[j].term_id) {
				k++;
			} else {
				j++;
			}
		}
		//printf("(%d, %d): inter: %f size1: %d size2: %d   fti: %d ftj: %d\n", d_query[0].doc_id, id, d_similarity[i].distance, querysize, size, ft_i[id], ft_j[id]);
		d_similarity[i].similarity = d_similarity[i].similarity / ((float) (querysize + size) - d_similarity[i].similarity);
	}
}

__global__ void calculateJaccardSimilarity(InvertedIndex inverted_index, Entry *d_query, int *index, Similarity *d_sim, int D, int docid, int *sizedoc, int maxsize, float threshold, int *ft_i, int *ft_j) {
	__shared__ int N;

	if (threadIdx.x == 0) {
		N = index[D - 1];	//Total number of items to be queried
	}
	__syncthreads();

	int block_size = N / gridDim.x + (N % gridDim.x == 0 ? 0 : 1);		//Partition size
	int lo = block_size * (blockIdx.x); 								//Beginning of the block
	int hi = min(lo + block_size, N); 								//End of the block
	int size = hi - lo;											// Real partition size (the last one can be smaller)

	int idx = 0;
	int end;
	int minoverlap, rem;

	for (int i = threadIdx.x; i < size; i += blockDim.x) {
		int pos = i + lo;

		while (true) {
			end = index[idx];

			if (end <= pos) {
				idx++;
			}
			else {
				break;
			}
		}

		Entry entry = d_query[idx]; 		//finds out the term
		int offset = end - pos;

		int idx2 = inverted_index.d_index[entry.term_id] - offset;
		Entry index_entry = inverted_index.d_inverted_index[idx2];
		int entry_size = sizedoc[index_entry.doc_id];

		if (index_entry.doc_id > docid && entry_size <= maxsize) {
			minoverlap = (threshold * ((float ) (entry_size + sizedoc[docid])))/(1 + threshold);
			rem = d_sim[index_entry.doc_id].similarity + entry_size - index_entry.pos;

			if (rem < minoverlap || d_sim[index_entry.doc_id].similarity + sizedoc[docid] - entry.pos < minoverlap) {
				d_sim[index_entry.doc_id].similarity = -1000000;
			} else {
				atomicAdd(&d_sim[index_entry.doc_id].similarity, 1);
				if (!d_sim[index_entry.doc_id].doc_j) {
					d_sim[index_entry.doc_id].doc_j = index_entry.doc_id;
				//	atomicAdd(&ft_i[index_entry.doc_j], index_entry.pos);
					//atomicAdd(&ft_j[index_entry.doc_id], entry.pos);
				}
				atomicMax(&ft_i[index_entry.doc_id], index_entry.pos);
				atomicMax(&ft_j[index_entry.doc_id], entry.pos);
			}
		}
	}
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

__global__ void filter_k (unsigned int *dst, const short *src, int *nres, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x)
    {
        if (src[i] > 0)
            dst[atomicAdd(nres, 1)] = i;
    }
}

__global__ void filter_registers(int *sim, float threshold, int querysize, int docid, int N, int *doc_size, Similarity *similars) { // similars + id_doc
	N -= (docid + 1);
	int block_size = N / gridDim.x + (N % gridDim.x == 0 ? 0 : 1);		//Partition size
	int offset = block_size * (blockIdx.x) + docid + 1; 				//Beginning of the block
	int lim = min(offset + block_size, N + docid + 1); 					//End of the block
	int size = lim - offset;

	similars += offset;
	sim += offset;
	doc_size += offset;

	for (int i = threadIdx.x; i < size; i += blockDim.x) {
		float jac = sim[i]/ (float) (querysize + doc_size[i] - sim[i]);

		similars[i].doc_j = offset + i;
		similars[i].similarity = jac;
	}
}
