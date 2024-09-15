
/*#include <stdio.h>
__global__ void GputAdd(float* A, float* B, float* C, int N)
{
    printf("1");
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

void HostAdd(float* A, float* B, float* C, int N){
    for(int i=0;i<N;i++)
        C[i]=A[i]+B[i];
}

void init(float*arr,int N){
    for(int i=0;i<N;i++){
        arr[i]=(float)( rand() & 0xFF )/10.0f;
    }
}

void show(float *arr, int N){
    for(int i=0;i<N;i++){
        printf("%5.2f  ",arr[i]);
    }
}
void check(float* A, float* B, int N){
    bool flag=1;
    double eps=1.0E-8;;
    for(int i=0;i<N;i++){
        if(abs(A[i]-B[i])>eps){
            flag=0;
            printf("arrays don't match\n");
            printf("error in %d point: gpu %5.2f cpu %5.2f \n",i,A[i],B[i]);
            break;
        }
    }
    if(flag)
        printf("arrays match\n");
    
}

// Host code
int main()
{
    int N = 16;
    size_t size = N * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* gpu_res = (float*)malloc(size);
    float* host_res = (float*)malloc(size);

    // Initialize input vectors
    init(h_A,N);
    init(h_B,N);

    // Allocate vectors in device memory
    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);
    float* d_C;
    cudaMalloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 4;
    int blocksPerGrid =4;

    GputAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(gpu_res, d_C, size, cudaMemcpyDeviceToHost);

    HostAdd(h_A,h_B,host_res,N);

    show(gpu_res,N);
    printf("\n");
    show(host_res,N);
    check(gpu_res,host_res,N);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(gpu_res);
    free(host_res);
}
*/


#include <stdio.h>
__global__ void helloFromGPU (void){
    printf("Hello World from GPU!\n");
}
int main(void)
{
    // hello from cpu
    printf("Hello World from CPU!\n");
    helloFromGPU <<<10, 1>>>();
    cudaDeviceReset();
    return 0;
}