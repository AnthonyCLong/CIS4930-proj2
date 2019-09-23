/* ==================================================================
	Programmer: Yicheng Tu (ytu@cse.usf.edu)
	The basic SDH algorithm implementation for 3D data
	To compile: nvcc SDH.c -o SDH in the rc machines
   ==================================================================
*/

/*==================================================================
Anthony Long 
U53579009
Project 1, Fall 2019
run on the c4 machine 7

to run:
$ module load apps/cuda/7.5
$ nvcc proj1-anthonylong.cu -o proj1
$ /proj1 10000 500.0
  ==================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define BOX_SIZE	23000 /* size of the data box on one dimension            */

/* descriptors for single atom in the tree */
typedef struct atomdesc 
{
	double x_pos;
	double y_pos;
	double z_pos;
} atom;

typedef struct hist_entry
{
	//float min;
	//float max;
	unsigned long long d_cnt;   /* need a long long type as the count might be huge */
} bucket;



bucket *GPUhistogram;	/* list of all buckets in the GPUhistogram   */
bucket *CPUhistogram;	/* list of all buckets in the CPUhistogram   */	
long long PDH_acnt;		/* total number of data points            */
int num_buckets;		/* total number of buckets in the GPUhistogram */
double PDH_res;			/* value of w                             */
atom *atom_list;		/* list of all data points                */


double p2p_distanceCPU(int ind1, int ind2) {
	
	double x1 = atom_list[ind1].x_pos;
	double x2 = atom_list[ind2].x_pos;
	double y1 = atom_list[ind1].y_pos;
	double y2 = atom_list[ind2].y_pos;
	double z1 = atom_list[ind1].z_pos;
	double z2 = atom_list[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}

//__devicce__ means function is executed on GPU, rather than the CPU
__device__ double p2p_distance(atom *p, int ind1, int ind2) 
{
	double x1 = p[ind1].x_pos;
	double x2 = p[ind2].x_pos;
	double y1 = p[ind1].y_pos;
	double y2 = p[ind2].y_pos;
	double z1 = p[ind1].z_pos;
	double z2 = p[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}

/* 
	brute-force SDH solution in a single CPU thread 
*/
int PDH_baselineCPU() {
	int i, j, h_pos;
	double dist;
	
	for(i = 0; i < PDH_acnt; i++) {
		for(j = i+1; j < PDH_acnt; j++) {
			//printf("i: %d, j: %d \n", i, j);
			dist = p2p_distanceCPU(i,j);
			h_pos = (int) (dist / PDH_res);
			//printf("dis: %f, W: %f, pos: %d \n", dist, PDH_res, h_pos);

			CPUhistogram[h_pos].d_cnt++;
		} 
	}
	return 0;
}

__global__ void PDH_baseline(bucket *hist, atom *atomList, double width, int size) 
{
	int i, pos;
	double dis;
	
	//i is computed by taking the correct block times the dimention of the block + the thread offset
	i = (blockIdx.x * blockDim.x) + threadIdx.x;

	/*iterates through each n-1 point pairs, using p2p distance 
	to find distance, divide it by the w, and use atomic add to incriment the correct bucket in the histogram*/
	for (int j = i+1; j < size; ++j) 
	{
		dis = p2p_distance(atomList, i, j);
		pos = (int) (dis / width);
		atomicAdd( &hist[pos].d_cnt, 1);
		__syncthreads();
	}
}

void output_histogram(bucket *hist){
	int i; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", hist[i].d_cnt);
		total_cnt += hist[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
	printf("\n");
}

void output_differences(bucket *hist1, bucket *hist2){
	int i; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		int diff = 	hist1[i].d_cnt - hist2[i].d_cnt;
		printf("%15lld ", diff);
		total_cnt += diff;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n Total differences:%lld \n", total_cnt);
		else printf("| ");
	}
	printf("\n");
}


int main(int argc, char const *argv[])
{
	PDH_acnt = atoi(argv[1]);	// Number of atoms
	PDH_res = atof(argv[2]);	// Input Distance: W

	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;

	//sizeof for malloc allocation
	size_t histSize = sizeof(bucket)*num_buckets;
	size_t atomSize = sizeof(atom)*PDH_acnt;

	//host histograms and atomlist
	GPUhistogram = (bucket *)malloc(histSize);
	CPUhistogram = (bucket *)malloc(histSize);
	atom_list = (atom *)malloc(atomSize);

	srand(1);
	/* generate data following a uniform distribution */
	for(int i = 0;  i < PDH_acnt; i++) 
	{
		atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}

	//device histogram and atom list
	bucket *d_histogram = NULL;
	atom *d_atom_list = NULL;

	//cuda malloc
	cudaMalloc((void**) &d_histogram, histSize);
	cudaMalloc((void**) &d_atom_list, atomSize);

	//cuda mem copy to device
	cudaMemcpy(d_histogram, GPUhistogram, histSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_atom_list, atom_list, atomSize, cudaMemcpyHostToDevice);

	//cpu baseline and gpu baseline
	PDH_baselineCPU();
	PDH_baseline <<<ceil(PDH_acnt/32.0), 32>>> (d_histogram, d_atom_list, PDH_res, PDH_acnt);

	//output of cpu, gpu, and differences
	cudaMemcpy(GPUhistogram, d_histogram, histSize, cudaMemcpyDeviceToHost);
	printf("CPU:");
	output_histogram(CPUhistogram);
	printf("GPU:");
	output_histogram(GPUhistogram);
	printf("Differences:");
	output_differences(CPUhistogram, GPUhistogram );

	//cuda mem free
	cudaFree(d_histogram);
	cudaFree(d_atom_list);

	//free allocated
	free(GPUhistogram);
	free(CPUhistogram);
	free(atom_list);

	cudaDeviceReset();

	return 0;
}









