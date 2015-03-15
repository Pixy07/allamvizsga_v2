#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include "windows.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <conio.h>
#include <stdlib.h>
#include "math.h"
#include <algorithm>

//#include "sslc.h"
const int KETSZAZOTVENHAT = 256;

struct NonZeroElement
{
	int row;
	int col;
	short value;
};

struct Leaf
{
	int klass;
	int fold;
	int super;
	int family;
	int first;
	int last;
};

struct SSMitem
{
	int nrNZ;
	int currNZbuffsize;
	float* NonZero_sij;
	float* NonZero_tij;
	int* NonZero_col;
	int* nrNZinrow;
	int* rowNZheads;
};



// global variables
	float infRate;
	float eps;
	int i,j,k;
	int nrProt;

	// normalizacio soran a sorok osszegeit tartalmazo tomb
	float* rowSum;

	// egyetlen kiteritett nem ritka sor a matrixbol, amit az expansion szamol kicurr_NZbuffsize
	// ebbol annyi db kell, ahany szalon szamolunk, pl 256 vagy tobb
	float* fleto_matyi;
	
	// ketto db ritka matrix struktura
	//SSMitem current;
	//SSMitem future;

	int* init_nrNZ;
	int* nrNZ;
//	int curr_nrNZ;
//	int curr_NZbuffsize;
	float* curr_NonZero_sij;
	int* curr_NonZero_tij;
	int* curr_NonZero_col;
	int* curr_nrNZinrow;
	int* curr_rowNZheads;

//	int future_nrNZ;
//	int future_NZbuffsize;
	float* future_NonZero_sij;
	int* future_NonZero_col;
	int* future_nrNZinrow;
	int* future_rowNZheads;

	int* Tiempo; 
	int* nzStat; 

	LARGE_INTEGER Frekk, Start, End;
// end of global

void startTime()
{	
	QueryPerformanceFrequency(&Frekk);
	QueryPerformanceCounter(&Start);
}

int endTime()
{
	QueryPerformanceCounter(&End);
	return (int)(1000.0*(double(End.QuadPart-Start.QuadPart)/double(Frekk.QuadPart)));
}

void initialize(int nProt, int nNZ, FILE* Fnz, float err, float epsilon, int mbSize)
{

}

__global__ void gpu_normalize_future(int *curr_NonZero_tij,float* future_NonZero_sij,int* future_NonZero_col,int* future_nrNZinrow,int* future_rowNZheads)
{
	int row = blockIdx.x*blockDim.x + threadIdx.x;
	float sum = 0;

	int index = future_rowNZheads[row];
	int count = future_nrNZinrow[row];

	for (int i=0; i<count; i++)
	{
		sum += future_NonZero_sij[index];
		++index;
	}

	index = future_rowNZheads[row];
	for (int i=0; i<count; i++)
	{
		future_NonZero_sij[index] /= sum;
		curr_NonZero_tij[index] = -1;
		++index;
	}
}

__global__ void gpu_normalize_curr(float* curr_NonZero_sij,int* curr_NonZero_tij,int* curr_NonZero_col,int* curr_nrNZinrow,int* curr_rowNZheads)
{
	int row = blockIdx.x*blockDim.x + threadIdx.x;
	float sum = 0;

	int index = curr_rowNZheads[row];
	int count = curr_nrNZinrow[row];

	for (int i=0; i<count; i++)
	{
		sum += curr_NonZero_sij[index];
		++index;
	}

	index = curr_rowNZheads[row];
	for (int i=0; i<count; i++)
	{
		curr_NonZero_sij[index] /= sum;
		++index;
	}
}

__global__ void gpu_up_symmetrize(float* curr_NonZero_sij, int* curr_NonZero_tij, int* curr_NonZero_col, int* curr_nrNZinrow, int* curr_rowNZheads)
{
	int row = blockIdx.x*blockDim.x + threadIdx.x;
	float newVal = 0;

	int index = curr_rowNZheads[row];
	int count = curr_nrNZinrow[row];

	for (int i=0; i<count; i++) 
		if(row < curr_NonZero_col[index])
		{
			int col = curr_NonZero_col[index];
			
			int index2 = curr_rowNZheads[col];
			int count2 = index2 + curr_nrNZinrow[col];
			int done = 0;
			do
			{
				if (curr_NonZero_col[index2] == row)
				{
					// itt szimmetrizelunk
					newVal = sqrt(curr_NonZero_sij[index] * curr_NonZero_sij[index2]);
					if (newVal < 0.001) newVal = 0.0;
					curr_NonZero_sij[index] = newVal;
					curr_NonZero_sij[index2] = newVal;
					curr_NonZero_tij[index] = index2;
					curr_NonZero_tij[index2] = index;
					++done;
				}
				++index2;
			}
			while (done==0 && curr_NonZero_col[index2] < col && index2<count2);
			if (done==0)
				curr_NonZero_sij[index] = 0;
			++index;
		}
}

__global__ void gpu_down_symmetrize(float* curr_NonZero_sij, int* curr_NonZero_tij, int* curr_NonZero_col, int* curr_nrNZinrow, int* curr_rowNZheads)
{
	int row = blockIdx.x*blockDim.x + threadIdx.x;
	int col;
	float newVal = 0;

	int start = curr_rowNZheads[row];
	int count = curr_nrNZinrow[row];
	int last = start + count;

	for (int i=start; i<last; i++) 
		if(row > curr_NonZero_col[i])
		{
			col = curr_NonZero_col[i];
			if (curr_NonZero_tij[i] < 0)  
				curr_NonZero_sij[i]=0;
			else
			{
				newVal = sqrt(curr_NonZero_sij[i] * curr_NonZero_sij[curr_NonZero_tij[i]]);
				if (newVal < 0.001) newVal = 0.0;
				curr_NonZero_sij[i] = newVal;
				curr_NonZero_sij[curr_NonZero_tij[i]] = newVal;
			}
		}
}

__global__ void gpu_killzeros(float* curr_NonZero_sij, int* curr_NonZero_tij, int* curr_NonZero_col, int* curr_nrNZinrow, int* curr_rowNZheads)
{
	int row = blockIdx.x*blockDim.x + threadIdx.x;

	int start = curr_rowNZheads[row];
	int count = curr_nrNZinrow[row];
	int last = start + count;
	int index = start;

	for (int i=start; i<last; i++) 
	{
		if (curr_NonZero_sij[i]>0)
		{
			if (index<i)
			{
				curr_NonZero_sij[index] = curr_NonZero_sij[i];
				curr_NonZero_col[index] = curr_NonZero_col[i];
			}
			++index;
		}
	}
	curr_nrNZinrow[row] = index-start;
}

__global__ void gpu_expand(float* curr_NonZero_sij, int* curr_NonZero_tij, int* curr_NonZero_col, int* curr_nrNZinrow, int* curr_rowNZheads, float * fleto_matyi, int* nrNZ, float* future_NonZero_sij,int* future_NonZero_col,int* future_nrNZinrow,int* future_rowNZheads)
{
	int row = blockIdx.x*blockDim.x + threadIdx.x;
	int thr = threadIdx.x;
	int fleto_offset = nrNZ[2] * thr;

	int start = curr_rowNZheads[row];
	int count = curr_nrNZinrow[row];
	int last = start + count;
	for (int i=start; i<last; i++) 
	{
		float w = curr_NonZero_sij[i];
		int col = curr_NonZero_col[i];

		int start2 = curr_rowNZheads[col];
		int count2 = curr_nrNZinrow[col];
		int last2 = start2 + count2;
		for (int j=start; j<last; j++) 
		{
			fleto_matyi[fleto_offset+curr_NonZero_col[j]] += w * curr_NonZero_sij[j];
		}
	}	

	float sum = 0;
	int db = 0;
	for (int i=0; i<nrNZ[2]; ++i) 
		sum += fleto_matyi[fleto_offset+i];
	for (int i=0; i<nrNZ[2]; ++i) 
	{
		float w = fleto_matyi[fleto_offset+i]/sum;
		if (w > 0.0005f) 
		{
			++db;
			fleto_matyi[fleto_offset+i] = w;
		}
		else
			fleto_matyi[fleto_offset+i] = 0;
	}

	future_nrNZinrow[row] = db;
	// exlcude all other threads
	future_rowNZheads[row] = nrNZ[1];
	__syncthreads();
	nrNZ[1] += db;
	__syncthreads();
	// endof
	int index = future_rowNZheads[row];
	for (int i=0; i<nrNZ[2]; ++i)
	{
		float w = fleto_matyi[fleto_offset+i];
		if (w>0)
		{
			future_NonZero_sij[index] = w;
			future_NonZero_col[index] = i;
			++index;
			fleto_matyi[fleto_offset+i] = 0;
		}
	}



}

__global__ void gpu_inflate(float* future_NonZero_sij, int* future_nrNZinrow, int* future_rowNZheads)
{
	float infRate = 1.5;
	int row = blockIdx.x*blockDim.x + threadIdx.x;

	int index = future_rowNZheads[row];
	int count = future_nrNZinrow[row];

	for (int i=0; i<count; i++)
	{
		future_NonZero_sij[index] = pow(future_NonZero_sij[index],infRate);
		++index;
	}
}


void done()
{
	free(curr_NonZero_sij);
	free(curr_NonZero_tij);
	free(curr_NonZero_col);
	free(curr_nrNZinrow);
	free(curr_rowNZheads);
	free(future_NonZero_sij);
//	free(future_NonZero_tij);
	free(future_NonZero_col);
	free(future_nrNZinrow);
	free(future_rowNZheads);
	free(rowSum);
	free(fleto_matyi);
}

int* makeOutput(int nrCycles)
{
	int* res = (int*)malloc(sizeof(int)*nrProt);
	int count = 0;
	int nrClu = 0;
	int largest = 0;
	int cluSize;
	int* clusters = (int*)malloc(sizeof(int)*nrProt);
	int* oldBuff = (int*)malloc(sizeof(int)*nrProt);
	int* newBuff = (int*)malloc(sizeof(int)*nrProt);
	int nrOld, nrNew;
	for (int i=0; i<nrProt; ++i) clusters[i]=-1;

	while (count<nrProt)
	{

		int o = 0;
		while (clusters[o]>=0) ++o;

		oldBuff[0] = o;
		nrOld = 1;
		cluSize = 1;

		clusters[o] = nrClu;
		res[count] = -o;
		++count;

		while (nrOld>0)
		{
			nrNew = 0;
			for (int old = 0; old<nrOld; ++old)
			{
				for (int j=0; j<future_nrNZinrow[oldBuff[old]]; ++j) 
				{
					int pos = future_rowNZheads[oldBuff[old]]+j;
					if (future_NonZero_sij[pos]>0 && clusters[future_NonZero_col[pos]]<0)
					{
						clusters[future_NonZero_col[pos]] = nrClu;
						res[count] = future_NonZero_col[pos];
						++count;
						++cluSize;
						newBuff[nrNew++] = future_NonZero_col[pos];
					}
				}
			}
			nrOld = nrNew;
			for (int j=0; j<nrNew; ++j) oldBuff[j] = newBuff[j];	
		}		
		if (cluSize > largest) largest = cluSize;
		++nrClu;
	}
	free(oldBuff);
	free(newBuff);
	free(clusters);
	printf("Largest cluster contains %d items.\n", largest);

	int sumAll = 0;
	for (int o=0; o<nrCycles; ++o) 
	{
		sumAll += Tiempo[o];
	}

	printf("Total runtime: %d msec \n",sumAll);
	printf("Average runtime pre iteration: %d msec \n",sumAll/nrCycles);
	printf("When the Checksum stabilizes, there's no need for further iterations. \n");

	count = 0;
	FILE* F = fopen("result.txt","wt");
	fprintf(F,"[");
	do
	{
		if (res[count]<0) fprintf(F," ]\n[");
		fprintf(F," %d",abs(res[count++]));
	}
	while (count<nrProt);
	fprintf(F," ]\n");
	fclose(F);

	return res;
}

//
//int* run(int q, int nrCycles = 50)
//{
//	
//	return reketye;
//}



int main()
{
	//This value defines the size of buffers used by the algorithm.
	//If the amount of nonzeros in the matrix exceeds the buffer size, the execution crashes.
	//Automatic estimation is used when MEGAITEMS is set to 0.
	const int MEGAITEMS = 0;
	
	//Markov clustering parameters. 
	//Inflation rate. Should be larger than 1.3
	const float err = 1.5f;

	//Nonzeros below this threshold are rounded to zero. Don't use below 0.001
	const float eps = 0.001f;

	//an input file with 10k proteins
	FILE* inpF = fopen("C:/Users/Cuda Programming/Desktop/alomvizsga/Allamvizsga_v2/Allamvizsga_v2/blast11944.sm","rb");

	int head[3];
	fread(&head,sizeof(int),3,inpF);
	printf("Header data: \n  %d proteins, \n  %d families (ground truth), \n  %d expected nonzeros.\n\n",head[0],head[1],head[2]);
	
	int q = (head[0]+KETSZAZOTVENHAT-1)/KETSZAZOTVENHAT;

	// if ground truth is not available, head[1] should be 0
	if (head[1]>0)
	{
		Leaf* leafBuffer = (Leaf*)malloc(sizeof(Leaf)*head[1]);
		fread(leafBuffer,sizeof(Leaf),head[1],inpF);
		free(leafBuffer);
	}
	// input file is open and the next byte to read is the record of the first nonzero in the input matrix

	printf("Executing Markov clustering, parameters: eps = %.6f; r=%.2f\n",eps,err);
	//initialize(, head[2], inpF, err, eps, MEGAITEMS);
	__int64* nzBuffer = (__int64*)malloc(sizeof(__int64)* 0x40000);

	nrProt = head[0];
	int nNZ= head[2];
	FILE* Fnz = inpF; 
		int mbSize = MEGAITEMS;
	int extra = (KETSZAZOTVENHAT - (nrProt % KETSZAZOTVENHAT)) % KETSZAZOTVENHAT;
	nrProt += extra;
	infRate = err;

	int NZbuffsize;

	if (mbSize>0)
	{
		NZbuffsize = mbSize * 0x100000;
	}
	else
	{
		int z = 5 * nNZ;
		z = (z / 0x100000 + 1) * 0x100000;
		NZbuffsize = z;
	}

	cudaMalloc(&nrNZ, 3 * sizeof(int));
	cudaMalloc(&curr_NonZero_sij, NZbuffsize*sizeof(float));
	cudaMalloc(&curr_NonZero_tij, NZbuffsize*sizeof(int));
	cudaMalloc(&curr_NonZero_col, NZbuffsize*sizeof(int));
	cudaMalloc(&curr_nrNZinrow, nrProt*sizeof(int));
	cudaMalloc(&curr_rowNZheads, nrProt*sizeof(int));

	cudaMalloc(&future_NonZero_sij, NZbuffsize*sizeof(float));
	cudaMalloc(&future_NonZero_col, NZbuffsize*sizeof(int));
	cudaMalloc(&future_nrNZinrow, nrProt*sizeof(int));
	cudaMalloc(&future_rowNZheads, nrProt*sizeof(int));

	cudaMalloc(&rowSum, nrProt*sizeof(float));
	cudaMalloc(&fleto_matyi, KETSZAZOTVENHAT*nrProt*sizeof(float)); // LE KELL NULLAZNI!!!!
	cudaMemset(fleto_matyi, 0, KETSZAZOTVENHAT*nrProt*sizeof(float));
	float* init_sij = (float*)malloc(NZbuffsize*sizeof(float));
	int* init_col = (int*)malloc(NZbuffsize*sizeof(int));
	int* init_nrNZinrow = (int*)malloc(nrProt*sizeof(int));
	int* init_rowNZheads = (int*)malloc(nrProt*sizeof(int));
	init_nrNZ = (int*)calloc(3, sizeof(int));

	init_nrNZ[2] = nrProt;

	int index = 0;
	int row = 0;
	int count = 0;

	init_rowNZheads[row] = 0;
	while (!feof(Fnz))
	{
		count = fread(nzBuffer, sizeof(__int64), 0x40000, Fnz);

		for (int index = 0; index<count; ++index)
		{
			int nzrow = nzBuffer[index] / 0x10000000000;
			int nzcol = (nzBuffer[index] / 0x10000) % 0x1000000;
			int nzval = nzBuffer[index] % 0x10000;

			if (nzrow != row)
			{
				init_nrNZinrow[row] = init_nrNZ[1] - init_rowNZheads[row];
				++row;
				init_rowNZheads[row] = init_nrNZ[1];
			}
			if (nzrow == row)
			{
				init_sij[init_nrNZ[1]] = nzval;
				init_col[init_nrNZ[1]] = nzcol;
				++init_nrNZ;
			}
		}
		init_nrNZinrow[row] = init_nrNZ[1] - init_rowNZheads[row];
	}
	fclose(Fnz);
	printf("Nonzero elements loaded: %d\n", init_nrNZ[1]);

	for (int i = 0; i<extra; ++i)
	{
		future_NonZero_sij[init_nrNZ[1]] = 1;
		future_NonZero_col[init_nrNZ[1]] = nrProt - extra + i;
		future_nrNZinrow[nrProt - extra + i] = 1;
		future_rowNZheads[nrProt - extra + i] = init_nrNZ[1];
		init_nrNZ[1]++;
	}

	cudaMemcpy(future_NonZero_sij, init_sij, init_nrNZ[1] * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(future_NonZero_col, init_col, init_nrNZ[1] * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(future_nrNZinrow, init_nrNZinrow, nrProt*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(future_rowNZheads, init_rowNZheads, nrProt*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(nrNZ, init_nrNZ, 3 * sizeof(int), cudaMemcpyHostToDevice);

	free(init_sij);
	free(init_col);
	free(init_nrNZinrow);
	free(init_rowNZheads);

	Tiempo = (int*)malloc(sizeof(int)* 1000);
	nzStat = (int*)malloc(sizeof(int)* 1000);

	//you can command the number of loops performed. 50 is the default.
	//int* output = run(q /* 60 */);
	int nrCycles = 50;
	int zero = 0;
	dim3 block(q, 1);
	dim3 thread(256, 1);
	gpu_normalize_future << <block, thread >> >(curr_NonZero_tij, future_NonZero_sij,  future_NonZero_col, future_nrNZinrow, future_rowNZheads);
	int MAXCYCLES = nrCycles;
	if (MAXCYCLES<20) MAXCYCLES = 20;
	if (MAXCYCLES>999) MAXCYCLES = 999;
	int cycle;


	for (cycle = 0; cycle<MAXCYCLES; ++cycle)
	{
		startTime();
		gpu_inflate << <block, thread >> >(future_NonZero_sij,  future_nrNZinrow, future_rowNZheads);
		gpu_normalize_future << <block, thread >> >(curr_NonZero_tij, future_NonZero_sij, future_NonZero_col,  future_nrNZinrow,future_rowNZheads);
		gpu_up_symmetrize << <block, thread >> >(curr_NonZero_sij, curr_NonZero_tij,  curr_NonZero_col, curr_nrNZinrow, curr_rowNZheads);
		gpu_normalize_curr << <block, thread >> >( curr_NonZero_sij,  curr_NonZero_tij,  curr_NonZero_col, curr_nrNZinrow,  curr_rowNZheads);
		gpu_down_symmetrize << <block, thread >> >(curr_NonZero_sij, curr_NonZero_tij,  curr_NonZero_col, curr_nrNZinrow, curr_rowNZheads);
		gpu_normalize_curr << <block, thread >> >( curr_NonZero_sij,  curr_NonZero_tij,  curr_NonZero_col,  curr_nrNZinrow,  curr_rowNZheads);

		init_nrNZ[1] = 0;
		cudaMemcpy(nrNZ, init_nrNZ, 3 * sizeof(int), cudaMemcpyHostToDevice);

		gpu_expand << <block, thread >> >(curr_NonZero_sij, curr_NonZero_tij, curr_NonZero_col, curr_nrNZinrow, curr_rowNZheads, fleto_matyi, nrNZ,future_NonZero_sij, future_NonZero_col,  future_nrNZinrow,  future_rowNZheads);

		Tiempo[cycle] = endTime();
		//nzStat[cycle] = future_nrNZ;

		//float sqSum = 0.0f;
		//for (int i=0; i<future_nrNZ; ++i) sqSum+=future_NonZero_sij[i]*future_NonZero_sij[i];

		printf("%d msec in cycle %d --- \n", Tiempo[cycle], cycle + 1);
	}

	int* reketye = makeOutput(cycle);
	done();

	return 0;
}