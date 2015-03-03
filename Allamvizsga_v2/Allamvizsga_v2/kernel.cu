#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <conio.h>
#include "handle_error.h"
#include <fstream>
#include <sstream>
using namespace std;
#define BLOCK_SIZE 256
float *hA;
//#define infRate 1.6
__global__ void expand(float *A, float *C, int N)
{
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	float sum = 0.0f;

		for (int n = 0; n < N; ++n)
			sum += A[row*N + n] * A[n*N + col];
		C[row*N + col] = sum;
}
__global__ void normalize2(int N, float *hA)
{

	int row = blockIdx.x*blockDim.x + threadIdx.x;
	
		double sum = 0;
		for (int n = 0; n < N; ++n)
			sum += hA[row*N + n];
		for (int n = 0; n < N; ++n)
			hA[row*N + n] = hA[row*N + n] / sum;	

}


__global__ void symmetrize2(int N, float *hA)
{
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	float newVal;	
		if (row>col)
		{
			newVal = sqrt(hA[row*N + col] * hA[col*N + row]);
			if (newVal < 0.0001) newVal = 0;
			hA[row*N + col] = newVal;
			hA[col*N + row] = newVal;
		}

}



__global__ void inflate2(int N, float *hA)
{
	float infRate = 1.5;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	hA[row*N + col] = pow(hA[row*N + col], infRate);

}


void writer(int N, float* C, float * dA, int size)
{
	HANDLE_ERROR(cudaMemcpy(C, dA, size, cudaMemcpyDeviceToHost));
	for (int row = 0; row < N; row++)
	{
		for (int col = 0; col < N; col++){
			cout << C[row*N + col] << " ";
		}
		cout << endl;
	}
}
void LoadProteinMatrix(string fname)
{
	const char* msg = fname.c_str();
	int o;
	FILE* F = fopen(msg, "rb");
	fread(&o, sizeof(int), 1, F);
	fread(&o, sizeof(int), 1, F);
	int ProtNR = 0;
	//job = (SCOP95job*) malloc(sizeof(SCOP95job));
	int* protein;
	int count = 0;
	ProtNR = o;
	protein = (int*)malloc(ProtNR * sizeof(int));
	count += fread(protein, sizeof(int), ProtNR, F);
	ProtNR = 895;
	int extendedProtNr = 0;
	if (ProtNR % 256 != 0)
	{
		for (int i = ProtNR;; ++i)
		{
			if (i % 256 == 0)
			{
				extendedProtNr = i;
				break;
			}
		}
	}
	else extendedProtNr = ProtNR;

	cout << count << endl;
	count = 0;
	//hA = (float**)malloc(ProtNR*sizeof(float*));
	hA = new float[extendedProtNr*extendedProtNr];// int prot = ProtNR*ProtNR;
	count += fread(hA, sizeof(float), ProtNR*ProtNR, F);

	if (count < ProtNR*ProtNR)
	{
		cout << "Matrix loading failed.";
	}
	for (int nr = ProtNR*ProtNR; nr < extendedProtNr*extendedProtNr; nr++)
	{
		if (nr % (extendedProtNr+1) == 0)
			hA[nr] = 1.0f;
		else
			hA[nr] = 0.0f;

	}
	
	
}
void filecreator(string filename, float *hA)
{

	ofstream myfile;
	myfile.open(filename);
	for (int i = 0; i < 1024 * 1024; ++i)
	{
		if (i % 1024 == 0) myfile << "\n";
		myfile << hA[i] << " ";
	}
	myfile.close();
}
int main(int argc, char *argv[])
{
	int N;
	string filename = "";
	int q = 4;
	N = q*BLOCK_SIZE;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float time;
	cout << "Matrix size: " << N << "x" << N << endl;

	// Allocate memory on the host
	float *hC;

	LoadProteinMatrix("1024.pm");
	int size = N*N*sizeof(float);    // Size of the memory in bytes
	float *dA, *dC;

	dim3 inf_b(16 * q, 16 * q);
	dim3 inf_t(16, 16);
	dim3 norm_b(q, 1);
	dim3 norm_t(256, 1);

	int count = 50;
	
	cudaEventRecord(start, 0);
	//	filecreator("in.txt", hA);
	cudaSetDevice(0);
	/*
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0);
	cout<<prop.name<<endl;
	cout << "Max threads/block"<<endl;
	cout <<prop.maxThreadsPerBlock<<endl;
	cout << "Processor Count"<<endl;
	cout <<prop.multiProcessorCount<<endl;
	cout << "maxGridSize"<<endl;
	cout <<prop.maxGridSize[0]<<endl;
	cout <<prop.maxGridSize[1]<<endl;
	cout <<prop.maxGridSize[2]<<endl;
	cout <<prop.maxGridSize[3]<<endl;
	cout <<"Compute capability: "<<endl;
	cout <<prop.major<<" - "<<prop.minor<<endl;
	*/

	// Allocate memory on the device
	
	HANDLE_ERROR(cudaMalloc(&dA, size));
	HANDLE_ERROR(cudaMalloc(&dC, size));
	// Copy matrices from the host to device
	HANDLE_ERROR(cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaFree(dA));
	HANDLE_ERROR(cudaMalloc(&dA, size));
	HANDLE_ERROR(cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dC, hA, size, cudaMemcpyHostToDevice));
	
	do
	{
		count--;

		inflate2 << <inf_b, inf_t >> >(N, dA);
		HANDLE_ERROR(cudaDeviceSynchronize());
		normalize2 << <norm_b, norm_t >> >(N, dA);
		HANDLE_ERROR(cudaDeviceSynchronize());
		symmetrize2 << <inf_b, inf_t >> >(N, dA);
		
		HANDLE_ERROR(cudaDeviceSynchronize());
		normalize2 << <norm_b, norm_t >> >(N, dA);
		HANDLE_ERROR(cudaDeviceSynchronize());
		symmetrize2 << <inf_b, inf_t >> >(N, dA);
		HANDLE_ERROR(cudaDeviceSynchronize());
		
		normalize2 << <norm_b, norm_t >> >(N, dA);
		HANDLE_ERROR(cudaDeviceSynchronize());
		expand << <inf_b, inf_t >> >(dA, dC, N);
		HANDLE_ERROR(cudaDeviceSynchronize());
		HANDLE_ERROR(cudaMemcpy(dA, dC, size, cudaMemcpyDeviceToDevice));
		/*writer(N,C,dA,size);
		cout <<"in cikle:";
		cin>>m;
		if (m==0) break;*///string str;          //The string
		//ostringstream temp;  //temp as in temporary
		//temp << 50 - count;
		//str = temp.str();
		//HANDLE_ERROR(cudaMemcpy(hA, dA, size, cudaMemcpyDeviceToHost));
		//filename = str;
		//filename += ".txt";
		//filecreator(filename, hA);
	} while (count >= 0);
	//writer(N,C,dA,size);
	HANDLE_ERROR(cudaEventRecord(stop, 0)); // Trigger Stop event
	HANDLE_ERROR(cudaEventSynchronize(stop)); // Sync events (BLOCKS till last (stop inthis case) has been recorded!)
	float elapsedTime = 0.0f; // Initialize elapsedTime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("Execution Time: %f millisecond\n", elapsedTime); // Print Elapsedtime

	// Now copy the GPU result back to CPU
	//    cudaDeviceSynchronize();

	HANDLE_ERROR(cudaFree(dA));

	HANDLE_ERROR(cudaFree(dC));
	return 0;
}

