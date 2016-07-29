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
 * structs.cuh
 *
 *  Created on: Dec 4, 2013
 *      Author: silvereagle
 */

#ifndef STRUCTS_CUH_
#define STRUCTS_CUH_

struct Entry {
    int doc_id;
    int term_id;
    int tf;
    int pos;

    __host__ __device__ Entry(int doc_id, int term_id, int tf = 0, int pos = 0) : doc_id(doc_id), term_id(term_id), tf(tf), pos(pos) {}

    bool operator < (const Entry& e) const {
        if(doc_id != e.doc_id) return doc_id < e.doc_id;
        return term_id < e.term_id;
    }
};

struct Similarity {
	int doc_i;
    int doc_j;
    float similarity;
    int ft_i;
   int ft_j;

    Similarity() {}
 //   __host__ __device__ Similarity(int doc_i, int doc_j, float distance) : doc_i(doc_i), doc_j(doc_j), similarity(distance) {}
    __host__ __device__ Similarity(int doc_i, int doc_j, float distance, int ft_i, int ft_j) : doc_i(doc_i), doc_j(doc_j), similarity(distance), ft_i(ft_i), ft_j(ft_j) {}

    __host__ __device__ bool operator < (const Similarity &sim) const {
        return similarity > sim.similarity;
    }

    __host__ __device__ bool operator > (const Similarity &sim) const {
        return similarity < sim.similarity;
    }
};

struct DeviceVariables{
	int *d_docstarts, *d_docsizes, *d_bC, *d_bO, *d_fti, *d_ftj;
    Entry *d_query;
    Similarity *d_candidates, *d_result;//, *d_nearestK, *h_nearestK;
    //float *d_qnorms, *d_similars;// [2] =  *d_qnorm, *d_qnorml1;
};

#endif /* STRUCTS_CUH_ */
