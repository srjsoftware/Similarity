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

#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include <vector>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <string>
#include <sstream>
#include <cuda.h>
#include <map>

#include "structs.cuh"
#include "utils.cuh"
#include "inverted_index.cuh"
#include "simjoin.cuh"
#include "cuda_distances.cuh"


#define OUTPUT 1
#define NUM_STREAMS 1


using namespace std;

struct FileStats {
	int num_docs;
	int num_terms;

	vector<int> sizes; // tamanho dos conjuntos
	map<int, int> doc_to_class;
	vector<int> start; // onde cada conjunto começa em entries

	FileStats() : num_docs(0), num_terms(0) {}
};

FileStats readInputFile(string &file, vector<Entry> &entries, vector<Entry> &entriesmid, float threshold);
void processTestFile(InvertedIndex &index, FileStats &stats, string &file, float threshold,
		string distance, stringstream &fileout);


/**
 * Receives as parameters the training file name and the test file name
 */

static int num_tests = 0;
int biggestQuerySize = -1;


int main(int argc, char **argv) {

	if (argc != 6) {
		cerr << "Wrong parameters. Correct usage: <executable> <input_file> <threshold> <cosine | l2 | l1> <output_file> <number_of_gpus>" << endl;
		exit(1);
	}

	int gpuNum;
	cudaGetDeviceCount(&gpuNum);

	if (gpuNum > atoi(argv[5])){
		gpuNum = atoi(argv[5]);
		if (gpuNum < 1)
			gpuNum = 1;
	}
	//cerr << "Using " << gpuNum << "GPUs" << endl;

	// we use 2 streams per GPU
	int numThreads = gpuNum*NUM_STREAMS;

	omp_set_num_threads(numThreads);

#if OUTPUT
	//truncate output files
	ofstream ofsf(argv[4], ofstream::trunc);
	ofsf.close();

	ofstream ofsfileoutput(argv[4], ofstream::out | ofstream::app);
#endif
	vector<string> inputs;// to read the whole test file in memory
	vector<InvertedIndex> indexes;
	indexes.resize(gpuNum);

	double starts, ends;

	string inputFileName(argv[1]);

	printf("Reading file...\n");
	vector<Entry> entries;
	vector<Entry> entriesmid;
	float threshold = atof(argv[2]);

	starts = gettime();
	FileStats stats = readInputFile(inputFileName, entries, entriesmid, threshold);
	ends = gettime();

	printf("Time taken: %lf seconds\n", ends - starts);

	vector<stringstream*> outputString;
	//Each thread builds an output string, so it can be flushed at once at the end of the program
	for (int i = 0; i < numThreads; i++){
		outputString.push_back(new stringstream);
	}

	//create an inverted index for all streams in each GPU
	#pragma omp parallel num_threads(gpuNum)
	{
		int cpuid = omp_get_thread_num();
		cudaSetDevice(cpuid);
		double start, end;

		start = gettime();
		indexes[cpuid] = make_inverted_index(stats.num_docs, stats.num_terms, entriesmid, entries);
		end = gettime();

		#pragma omp single nowait
		printf("Total time taken for insertion: %lf seconds\n", end - start);
	}


	#pragma omp parallel
	{
		int cpuid = omp_get_thread_num();
		cudaSetDevice(cpuid / NUM_STREAMS);


		string distanceFunction(argv[3]);

		FileStats lstats = stats;

		processTestFile(indexes[cpuid / NUM_STREAMS], lstats, inputFileName, threshold,
			distanceFunction, *outputString[cpuid]);
		if (cpuid %  NUM_STREAMS == 0)
			gpuAssert(cudaDeviceReset());

	}

#if OUTPUT
		starts = gettime();
		for (int i = 0; i < numThreads; i++){
			ofsfileoutput << outputString[i]->str();
		}
		ends = gettime();

		printf("Time taken to write output: %lf seconds\n", ends - starts);

		ofsfileoutput.close();
#endif
		return 0;
}

FileStats readInputFile(string &filename, vector<Entry> &entries, vector<Entry> &entriesmid, float threshold) {
	ifstream input(filename.c_str());
	string line;

	FileStats stats;
	int accumulatedsize = 0;
	int doc_id = 0;

	while (!input.eof()) {
		getline(input, line);
		if (line == "") continue;

		num_tests++;
		vector<string> tokens = split(line, ' ');
		biggestQuerySize = max((int)tokens.size() / 2, biggestQuerySize);

		int size = (tokens.size() - 2)/2;
		stats.sizes.push_back(size);
		stats.start.push_back(accumulatedsize); // TODO: salvar na stats ou outro lugar
		accumulatedsize += size;

		int midprefix = get_midprefix(size, threshold);

		for (int i = 2, size = tokens.size(), j = 0; i + 1 < size; i += 2, j++) {
			int term_id = atoi(tokens[i].c_str());
			int term_count = atoi(tokens[i + 1].c_str());
			stats.num_terms = max(stats.num_terms, term_id + 1);
			entries.push_back(Entry(doc_id, term_id, term_count, j));

			if (j < midprefix + 1) {
				entriesmid.push_back(Entry(doc_id, term_id, term_count, j));
			}
		}
		doc_id++;
	}

	stats.num_docs = num_tests;

	input.close();

	return stats;
}

void allocVariables(DeviceVariables *dev_vars, float threshold, int num_docs, Similarity** h_similarity, int batch) {
	dim3 grid, threads;

	get_grid_config(grid, threads);

	gpuAssert(cudaMalloc(&dev_vars->d_candidates, batch*num_docs * sizeof(Similarity))); // distance between all the docs and the query doc
	gpuAssert(cudaMalloc(&dev_vars->d_result, batch*num_docs * sizeof(Similarity))); // compacted similarities between all the docs and the query doc
	gpuAssert(cudaMalloc(&dev_vars->d_docstarts, num_docs * sizeof(int))); // count of elements in common
	gpuAssert(cudaMalloc(&dev_vars->d_docsizes, num_docs * sizeof(int))); // size of all docs
	*h_similarity = (Similarity*)malloc(batch*num_docs * sizeof(Similarity));

	int blocksize = 1024;
	int numBlocks = num_docs*batch / blocksize + (num_docs*batch % blocksize ? 1 : 0);

	gpuAssert(cudaMalloc(&dev_vars->d_bC,sizeof(int)*(numBlocks + 1)));
	gpuAssert(cudaMalloc(&dev_vars->d_bO,sizeof(int)*numBlocks));

}

void freeVariables(DeviceVariables *dev_vars, InvertedIndex &index, Similarity** distances){
	cudaFree(dev_vars->d_candidates);
	cudaFree(dev_vars->d_result);
	cudaFree(dev_vars->d_docstarts);
	cudaFree(dev_vars->d_docsizes);
	cudaFree(dev_vars->d_bC);
	cudaFree(dev_vars->d_bO);

	free(*distances);

	if (omp_get_thread_num() % NUM_STREAMS == 0){
		cudaFree(index.d_count);
		cudaFree(index.d_index);
		cudaFree(index.d_inverted_index);
	}
}

void processTestFile(InvertedIndex &index, FileStats &stats, string &filename, float threshold,
		string distance, stringstream &outputfile) {

	//#pragma omp single nowait
	printf("Processing input file %s...\n", filename.c_str());

	DeviceVariables dev_vars;
	Similarity* h_result;
	int batch = 500;

	allocVariables(&dev_vars, threshold, index.num_docs, &h_result, batch);

	cudaMemcpyAsync(dev_vars.d_docsizes, &stats.sizes[0], index.num_docs * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(dev_vars.d_docstarts, &stats.start[0], index.num_docs * sizeof(int), cudaMemcpyHostToDevice);

	double start = gettime();
	int num_test_local = 0;

	#pragma omp for
	for (int docid = 0; docid < index.num_docs - 1; docid += batch) {

		int totalSimilars = findSimilars(index, threshold, &dev_vars, h_result, docid, batch);
		num_test_local += batch;

		for (int i = 0; i < totalSimilars; i++) {
#if OUTPUT
			if (h_result[i].similarity >= threshold) {
				outputfile << "(" << h_result[i].doc_i  << ", " << h_result[i].doc_j << "): " << h_result[i].similarity << endl;
			}
#endif
		}
	}

	freeVariables(&dev_vars, index, &h_result);
	int threadid = omp_get_thread_num();

	printf("Entries in device %d stream %d: %d\n", threadid / NUM_STREAMS, threadid %  NUM_STREAMS, num_test_local);

	#pragma omp barrier

	double end = gettime();

	//#pragma omp master
	//printf("Total num tests %d\n", num_tests);

	#pragma omp master
	{
		printf("Time taken for %d queries: %lf seconds\n\n", stats.num_docs, end - start);
	}

}
