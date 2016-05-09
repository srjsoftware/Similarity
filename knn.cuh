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

/*
 * knn.cuh
 *
 *  Created on: Dec 4, 2013
 *      Author: silvereagle
 */

#ifndef KNN_CUH_
#define KNN_CUH_

#include "inverted_index.cuh"

__global__ void calculateDistances(InvertedIndex inverted_index, Entry *d_query, int *index, Similarity *dist, int D);

__global__ void bitonicPartialSort(Similarity *dist, Similarity *nearestK, int N, int K);

//__global__ void get_term_count_and_tf_idf(InvertedIndex inverted_index, Entry *query, int *count, float *qnorm, float *qnorml1, int N);
__global__ void get_term_count_and_tf_idf(InvertedIndex inverted_index, Entry *query, int *count, int N);

__host__ int KNN(InvertedIndex inverted_index, std::vector<Entry> &query, float threshold, Similarity* h_nearestK,
	void(*distance)(InvertedIndex, Entry*, int*, Similarity*, int D), struct DeviceVariables *dev_vars, int docid);

__device__ void bitonicPartialSort(Similarity *dist, int N, int K);

__device__ void bitonicPartialMerge(Similarity *dist, Similarity *nearestK, int N, int K);

__device__ void initDistances(Similarity *dist, int offset, int N);

__device__ void calculateDistancesDevice(InvertedIndex inverted_index, Entry *d_query, int *index, Similarity *dist, int D);

__global__ void filter_registers(int *sim, float threshold, int querysize, int docid, int N, int *doc_size, Similarity *similars);

#endif /* KNN_CUH_ */
