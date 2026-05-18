#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Simple matrix multiply kernel: C = A * B
__global__ void matmul(const float *A, const float *B, float *C, int M, int N, int K) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
    	if (row < M && col < N) {
		float sum = 0.0f;
		for (int i = 0; i < K; i++) {
			sum += A[row * K + i] * B[i * N + col];
		}
		C[row * N + col] = sum;
	}
}

void printMatrix(const float *mat, int rows, int cols) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			printf("%6.2f ", mat[i * cols + j]);
		}
		printf("\n");
	}
}

int main() {
	int M = 4, K = 2, N = 4;  // dimensions (C = A(MxK) * B(KxN))

	size_t sizeA = M * K * sizeof(float);
	size_t sizeB = K * N * sizeof(float);
	size_t sizeC = M * N * sizeof(float);

	// Allocate host memory
	float *h_A = (float*)malloc(sizeA);
	float *h_B = (float*)malloc(sizeB);
	float *h_C = (float*)malloc(sizeC);

	// Initialize data
	for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
	for (int i = 0; i < K * N; i++) h_B[i] = 1.0f;

	// Allocate device memory
	float *d_A, *d_B, *d_C;
	cudaMalloc((void**)&d_A, sizeA);
	cudaMalloc((void**)&d_B, sizeB);
	cudaMalloc((void**)&d_C, sizeC);

	// Copy data to device
	cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

	// Launch kernel
	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((N + 15) / 16, (M + 15) / 16);
	matmul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);

	// Copy result back
	cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

	// Print result
	printf("Matrix A:\n");
	printMatrix(h_A, M, K);
	printf("Matrix B:\n");
	printMatrix(h_B, K, N);
	printf("Result matrix C:\n");
	printMatrix(h_C, M, N);

	// Cleanup
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}

