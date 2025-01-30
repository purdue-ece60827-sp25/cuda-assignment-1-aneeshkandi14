
#include "cudaLib.cuh"
#include "cpuLib.h"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < size)
		y[i] = scale * x[i] + y[i];
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here
	// std::cout << "Lazy, you are!\n";
	// std::cout << "Write code, you must\n";

	float * a, * b, * c;
	float * a_d, * c_d;

	// Memory allocation in CPU
	a = (float *) malloc(vectorSize * sizeof(float));
	b = (float *) malloc(vectorSize * sizeof(float));
	c = (float *) malloc(vectorSize * sizeof(float));

	if (a == NULL || b == NULL || c == NULL) {
		printf("Unable to malloc memory ... Exiting!");
		return -1;
	}

	// Initiating vectors
	vectorInit(a, vectorSize);
	vectorInit(b, vectorSize);
	float scale = 2.0f;

	// Memory allocation in GPU
	cudaMalloc((void **) &a_d, vectorSize * sizeof(float));
	cudaMalloc((void **) &c_d, vectorSize * sizeof(float));

	// Copy memory from host to device
	cudaMemcpy(a_d, a, vectorSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(c_d, b, vectorSize * sizeof(float), cudaMemcpyHostToDevice);
	
	#ifndef DEBUG_PRINT_DISABLE 
		printf("\n Adding vectors : \n");
		printf(" scale = %f\n", scale);
		printf(" a = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", a[i]);
		}
		printf(" ... }\n");
		printf(" b = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", b[i]);
		}
		printf(" ... }\n");
	#endif

	saxpy_gpu<<<ceil(float(vectorSize)/1024), 1024>>>(a_d, c_d, scale, vectorSize);
	
	cudaMemcpy(c, c_d, vectorSize * sizeof(float), cudaMemcpyDeviceToHost);	

	#ifndef DEBUG_PRINT_DISABLE 
		printf(" c = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", c[i]);
		}
		printf(" ... }\n");
	#endif

	int errorCount = verifyVector(a, b, c, scale, vectorSize);
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";

	cudaFree(a_d);
	cudaFree(c_d);

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int hitCount = 0;

	// Setup RNG
	curandState_t rng;
	curand_init(clock64(), idx, 0, &rng);

	float x, y;
	if(idx < pSumSize){
		for (uint64_t i = 0; i < sampleSize; i++){
			x = curand_uniform(&rng);
			y = curand_uniform(&rng);

			if ( int(x * x + y * y) == 0 )
				++ hitCount;
		}
	
		pSums[idx] = hitCount;
	}
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	uint64_t partialHitCount = 0;
	uint64_t n = pSumSize / reduceSize;
	if(idx < reduceSize){
		for(int i = 0; i < n; i++)
			partialHitCount += pSums[idx + i];

		totals[idx] = partialHitCount;
	}
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;
	uint64_t totalHitCount = 0;
	//      Insert code here
	// std::cout << "Sneaky, you are ...\n";
	// std::cout << "Compute pi, you must!\n";
	uint64_t * pSum_d, * totals, * totals_d;
	cudaMalloc((void **) &pSum_d, generateThreadCount * sizeof(uint64_t));
	cudaMalloc((void **) &totals_d, reduceThreadCount * sizeof(uint64_t));
	totals = (uint64_t *) malloc(reduceThreadCount * sizeof(uint64_t));

	// Kernel launch
	generatePoints<<<ceil(float(generateThreadCount) / 1024), 1024>>>(pSum_d, generateThreadCount, sampleSize);
	reduceCounts<<<ceil(float(reduceThreadCount) / 1024), 1024>>>(pSum_d, totals_d, generateThreadCount, reduceThreadCount);

	cudaMemcpy(totals, totals_d, reduceThreadCount * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	cudaFree(pSum_d);
	cudaFree(totals_d);

	for (int i = 0; i < reduceThreadCount; i++)
		totalHitCount += totals[i];

	approxPi = ((double) totalHitCount / sampleSize) / generateThreadCount;
	approxPi = approxPi * 4.0f;
	
	return approxPi;
}
