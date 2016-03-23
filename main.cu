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
#include "knn.cuh"
#include "cuda_distances.cuh"


#define OUTPUT 1


using namespace std;

struct FileStats {
	int num_docs;
	int num_terms;

	map<int, int> doc_to_class;

	FileStats() : num_docs(0), num_terms(0) {}
};

FileStats readInputFile(string &file, vector<Entry> &entries, vector<string>& inputs);
void processTestFile(InvertedIndex &index, FileStats &stats, vector<string>& input,
	string &file, float threshold, string distance, stringstream &fileout);
bool makeQuery(InvertedIndex &inverted_index, FileStats &stats, string &line, float threshold, 
	void(*distance)(InvertedIndex, Entry*, int*, Similarity*, int D), Similarity* distances,
	stringstream &fileout, DeviceVariables *dev_vars);

int get_class(string token);


/**
 * Receives as parameters the training file name and the test file name
 */

static int num_tests = 0;
static int correct_l2 = 0, correct_cosine = 0, correct_l1 = 0;
static int wrong_l2 = 0, wrong_cosine = 0, wrong_l1 = 0;
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
	cerr << "Using " << gpuNum << "GPUs" << endl;

	// we use 2 streams per GPU
	int numThreads = gpuNum* NUM_STREAMS;

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

	starts = gettime();
	FileStats stats = readInputFile(inputFileName, entries, inputs);
	ends = gettime();

	printf("time taken: %lf seconds\n", ends - starts);
	//fprintf(stderr,"sizeof Entry %u , sizeof Similarity %u\n",sizeof(Entry), sizeof(Similarity));

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
		indexes[cpuid] = make_inverted_index(stats.num_docs, stats.num_terms, entries);
		end = gettime();

		#pragma omp single nowait
		printf("Total time taken for insertion: %lf seconds\n", end - start);
	}


	#pragma omp parallel 
	{
		int cpuid = omp_get_thread_num();
		cudaSetDevice(cpuid / NUM_STREAMS);

		float threshold = atof(argv[2]);
		string distanceFunction(argv[3]);

		FileStats lstats = stats;

		processTestFile(indexes[cpuid / NUM_STREAMS], lstats, inputs, inputFileName, threshold,
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

		printf("time taken to write output: %lf seconds\n", ends - starts);

		ofsfileoutput.close();
#endif
		return 0;
}

FileStats readInputFile(string &filename, vector<Entry> &entries, vector<string>& inputs) {
	ifstream input(filename.c_str());
	string line;

	FileStats stats;

	while (!input.eof()) {
		getline(input, line);
		if (line == "") continue;

		inputs.push_back(line);
		num_tests++;
	}
	
	stats.num_docs = num_tests;
	
	for (int doc_id = 0; doc_id < num_tests; doc_id++) {
		line = inputs[doc_id];
		
		vector<string> tokens = split(line, ' ');
		biggestQuerySize = max((int)tokens.size() / 2, biggestQuerySize);

		stats.doc_to_class[doc_id] = get_class(tokens[1]);

		for (int i = 2, size = tokens.size(); i + 1 < size; i += 2) {
			int term_id = atoi(tokens[i].c_str());
			int term_count = atoi(tokens[i + 1].c_str());
			stats.num_terms = max(stats.num_terms, term_id + 1);
			entries.push_back(Entry(doc_id, term_id, term_count));
		}
	}

	input.close();

	return stats;
}

void allocVariables(DeviceVariables *dev_vars, float threshold, int num_docs, Similarity** distances){
	dim3 grid, threads;

	get_grid_config(grid, threads);

	gpuAssert(cudaMalloc(&dev_vars->d_dist, num_docs * sizeof(Similarity)));
	gpuAssert(cudaMalloc(&dev_vars->d_query, biggestQuerySize * sizeof(Entry)));
	gpuAssert(cudaMalloc(&dev_vars->d_index, biggestQuerySize * sizeof(int)));
	gpuAssert(cudaMalloc(&dev_vars->d_count, biggestQuerySize * sizeof(int)));
	gpuAssert(cudaMalloc(&dev_vars->d_qnorms, 2 * sizeof(float)));

	*distances = (Similarity*)malloc(num_docs * sizeof(Similarity));

}

void freeVariables(DeviceVariables *dev_vars, InvertedIndex &index, Similarity** distances){
	cudaFree(dev_vars->d_dist);
	cudaFree(dev_vars->d_query);
	cudaFree(dev_vars->d_index);
	cudaFree(dev_vars->d_count);
	cudaFree(dev_vars->d_qnorms);

	free(*distances);

	if (omp_get_thread_num() % NUM_STREAMS == 0){
		cudaFree(index.d_count);
		cudaFree(index.d_index);
		cudaFree(index.d_inverted_index);
		cudaFree(index.d_norms);
		cudaFree(index.d_normsl1);
	}
}

void processTestFile(InvertedIndex &index, FileStats &stats, vector<string>& input_t,
	string &filename, float threshold, string distance, stringstream &outputfile) {

	int num_test_local = 0, i;

	//#pragma omp single nowait
	printf("Processing input file %s...\n", filename.c_str());

	DeviceVariables dev_vars;
	Similarity* distances;

	allocVariables(&dev_vars, threshold, index.num_docs, &distances);	

	double start = gettime();

	#pragma omp for
	for (i = 0; i < input_t.size(); i++){

		num_test_local++;

		if (distance == "cosine" || distance == "both") {
			if (makeQuery(index, stats, input_t[i], threshold, CosineDistance, distances, outputfile, &dev_vars)) {
				#pragma omp atomic
				correct_cosine++;
			}
			else {
				#pragma omp atomic
				wrong_cosine++;
			}
		}

		if (distance == "l2" || distance == "both") {

			if (makeQuery(index, stats, input_t[i], threshold, EuclideanDistance, distances, outputfile, &dev_vars)) {
				#pragma omp atomic
				correct_l2++;
			}
			else {
				#pragma omp atomic
				wrong_l2++;
			}
		}

		if (distance == "l1" || distance == "both") {
			if (makeQuery(index, stats, input_t[i], threshold, ManhattanDistance, distances, outputfile, &dev_vars)) {
				#pragma omp atomic
				correct_l1++;
			}
			else {
				#pragma omp atomic
				wrong_l1++;
			}
		}

		input_t[i].clear();
	}

	freeVariables(&dev_vars, index, &distances);
	int threadid = omp_get_thread_num();

	printf("Entries in device %d stream %d: %d\n", threadid / NUM_STREAMS, threadid %  NUM_STREAMS, num_test_local);

	#pragma omp barrier

	double end = gettime();

	#pragma omp master
	printf("Total num tests %d\n", num_tests);

	#pragma omp master
	{
		printf("Time taken for %d queries: %lf seconds\n\n", num_tests, end - start);

		if (distance == "cosine" || distance == "both") {
			printf("Cosine similarity\n");
			printf("Correct: %d Wrong: %d\n", correct_cosine, wrong_cosine);
			printf("Accuracy: %lf%%\n\n", double(correct_cosine) / double(num_tests));
		}

		if (distance == "l2" || distance == "both") {
			printf("L2 distance\n");
			printf("Correct: %d Wrong: %d\n", correct_l2, wrong_l2);
			printf("Accuracy: %lf%%\n\n", double(correct_l2) / double(num_tests));
		}

		if (distance == "l1" || distance == "both") {
			printf("L1 distance\n");
			printf("Correct: %d Wrong: %d\n", correct_l1, wrong_l1);
			printf("Accuracy: %lf%%\n\n", double(correct_l1) / double(num_tests));
		}
	}

}

bool makeQuery(InvertedIndex &inverted_index, FileStats &stats, string &line, float threshold,
	void(*distance)(InvertedIndex, Entry*, int*, Similarity*, int D), Similarity *distances,
	stringstream &outputfile, DeviceVariables *dev_vars) {

	vector<Entry> query;
	vector<string> tokens = split(line, ' ');

	int docid = atoi(tokens[0].c_str());

	for (int i = 2, size = tokens.size(); i + 1 < size; i += 2) {
		int term_id = atoi(tokens[i].c_str());
		int term_count = atoi(tokens[i + 1].c_str());

		query.push_back(Entry(0, term_id, term_count));
	}

	//Creates an empty document if there are no terms
	if (query.empty()) {
		query.push_back(Entry(0, 0, 0));
	}

	KNN(inverted_index, query, threshold, distances, distance, dev_vars, docid);
	
	float qnorm, qnorml1, qnorms[2];

	cudaMemcpyAsync(qnorms, dev_vars->d_qnorms, 2 * sizeof(float), cudaMemcpyDeviceToHost);

	// TODO modificar para outras distâncias
	qnorm = qnorms[0];
	qnorml1 = qnorms[1];
	float qnormeucl = qnorm;
	float qnormcos = sqrt(qnorm);
	if (qnormcos == 0)qnormcos = 1; //avoid NaN

	for (int i = docid + 1; i < inverted_index.num_docs; i++) {
		if (distances[i].distance/qnormcos > threshold) {
#if OUTPUT
			outputfile << "(" << docid << ", " << distances[i].doc_id << "): " << distances[i].distance/qnormcos << endl;
#endif
		}
	}

	return 0;
}

int get_class(string token) {
	vector<string> class_tokens = split(token, '=');

	if (class_tokens.size() == 1) {
		return atoi(class_tokens[0].c_str());
	}
	else {
		return atoi(class_tokens[1].c_str());
	}
}
